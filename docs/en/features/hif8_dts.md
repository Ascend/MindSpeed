# HiF8 DTS (Delayed Tensor Scaling) Training

## Background and Challenges

HiF8 (Hybrid Float8) is a dynamic low-precision floating-point data format with 1 sign bit plus dynamic dot bits, exponent bits, and mantissa bits, with a maximum representable value of 2E15. Compared with the fixed-format E4M3 / E5M2, HiF8 balances a larger numerical representation range and higher precision by dynamically adjusting the bit-width allocation, making it suitable for a broader range of model structures and training scenarios.

In HiF8 low-precision training, the strategy for computing the scaling factor directly determines the numerical stability of training. Delayed Tensor Scaling (DTS) is a delayed scaling strategy based on the historical amax: the recipe collects amax in the early stage of training and updates the scale factor at fixed intervals in the steady-state stage. Because DTS relies on historical statistics, when NaN/Inf occurs during training (such as abnormal batch data or gradient explosion), the contaminated amax history causes the scale factor of subsequent steps to become inaccurate, which in turn triggers continuous numerical overflow and eventually leads to training crash.

## Solution

MindSpeed provides the **Step Recovery** feature for HiF8 DTS training, which transactionally wraps `train_step` at the framework layer:

1. **NaN/Inf detection**: Before the optimizer commit, check the loss, grad norm, and optimizer overflow flag to determine whether the current step is valid.
2. **All-rank failure synchronization**: Synchronize the failure decision and failure reason across the world group via `all_reduce(MAX)` to ensure all ranks retry consistently, avoiding topology inconsistency caused by the optimizer all-reduce of pipeline-only ranks not being covered.
3. **Transient state cleanup**: Clear gradients, the optimizer overflow flag, and pending asynchronous communication, but **do not touch** the parameter updates, optimizer momentum/variance, scheduler state, global step, and consumed-sample counters (these are committed only after a successful attempt).
4. **HiF8 amax history reset**: Call the `FP8GlobalStateManager.reset_fp8_amax_history()` interface of TransformerEngineNPU to clear the contaminated amax history, so that subsequent steps restart from a clean scale factor.
5. **RNG state restoration**: Restore the random number state of Python, NumPy, Torch CPU/Device, and the Megatron TP RNG tracker, so that dropout / random mask / MoE routing are reproduced as much as possible on retry.
6. **Batch replay**: Cache the microbatches consumed by the original attempt and replay the same batch on retry, ensuring the continuity of the training data stream. Two modes are supported:
   - **ExternalReplayAdapter**: The framework layer caches the microbatches produced by the iterator, suitable for scenarios where the rerun state machine is not activated. Supports a `None` data_iterator (pipeline scenarios).
   - **NativeRerunReplayAdapter**: Implements replay based on Megatron `RerunDataIterator.saved_microbatches`, suitable for scenarios where the rerun state machine is activated.
7. **Retry once**: After cleanup and reset, re-execute `train_step` using the restored RNG and the replayed batch. If the retry still fails, output a structured diagnostic log and raise a `RuntimeError`.

The framework layer **does not maintain or read** the HiF8 warmup / CTS / DTS internal state; the only interaction point with TransformerEngineNPU is `reset_fp8_amax_history()`.

## Additional Overhead of Normal Steps

After Step Recovery is enabled, each normal step (without NaN/Inf) incurs **two** additional small `all_reduce(MAX)` collective communications:

| Synchronization Point | Timing | Payload Size | Description |
|--------|------|-------------|------|
| `after_prepare_grads` | After the optimizer `prepare_grads()` returns | 2 x int32 | Synchronize the found_inf flag and the failure reason |
| `before_step_with_ready_grads` | Before the optimizer `step_with_ready_grads()` is called | 2 x int32 | Synchronize the NaN/Inf detection result of loss/grad_norm |

The payload of each `all_reduce` is only 2 int32 values (8 bytes), so the overhead is negligible. These two synchronization points ensure that when any rank detects an abnormality, all ranks can stop and retry consistently before the optimizer commit.

In scenarios other than the `hif8_delayed` recipe, the wrapper short-circuits directly to the original `train_step`, with no additional overhead.

## Application Scenario

- When performing HiF8 DTS low-precision training with `--fp8-format hif8 --fp8-recipe hif8_delayed`, Step Recovery is enabled by default.
- Scenarios where occasional NaN/Inf during training (such as abnormal data or gradient explosion) needs to be automatically recovered without interrupting training.
- Scenarios that require continuity of the training data stream and do not want to skip batches.

## Usage

### Basic Enabling

HiF8 DTS training is enabled through the following parameter combination, and Step Recovery is automatically activated accordingly:

```bash
--transformer-impl transformer_engine
--fp8-format hif8
--fp8-recipe hif8_delayed
```

### Explicitly Disabling Step Recovery

To disable Step Recovery (for example, for debugging or comparing with a baseline), add:

```bash
--no-hif8-step-recovery
```

### HiF8 DTS Configuration Parameters

The HiF8 DTS recipe provides the following tunable parameters, which are used to control the amax collection and scale factor update behavior:

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--hif8-input-margin` | int | 11 | Number of guard bits for input/activation tensors, recommended range 9-11 |
| `--hif8-weight-margin` | int | 12 | Number of guard bits for weight tensors, recommended range 11-12 |
| `--hif8-grad-margin` | int | 11 | Number of guard bits for gradient tensors, recommended range 9-11 |
| `--hif8-amax-collect-interval` | int | 5 | Length of the warmup stage (number of iterations). During this period, amax is collected at every step; recommended range 5-20 |
| `--hif8-scale-update-interval` | int | 10 | Interval for amax history collection and scale factor update in the steady-state stage (number of iterations) |
| `--hif8-amax-history-len` | int | 128 | Length of the amax history buffer, recommended values 64 / 128 / 256 |
| `--no-hif8-step-recovery` | flag | False | Disable HiF8 NaN/Inf Step Recovery |

## Usage Impact

- **Training continuity**: After detecting NaN/Inf, Step Recovery retries the same batch instead of skipping it, ensuring the continuity of the training data stream and avoiding convergence drift.
- **Numerical stability**: By resetting the HiF8 amax history, it eliminates the impact of the contaminated scale factor on subsequent steps and avoids cascading failures.
- **Determinism**: By restoring the complete RNG state, the retry attempt reproduces the random behavior of the original attempt as much as possible (dropout, mask, routing, etc.).
- **Overhead**: Additional overhead is incurred only when NaN/Inf is detected (cleanup, reset, RNG restoration, batch replay, and one extra forward/backward pass). The additional overhead of a normal step is only two 2 x int32 `all_reduce(MAX)` operations, which is negligible. The wrapper short-circuits directly in scenarios other than `hif8_delayed`.
- **Retry limit**: Each step is retried at most once. If the retry still fails, training is aborted and a structured diagnostic log is output (including iteration, reason, rank, loss, and grad_norm).

## Parameter Combination Restrictions

The `hif8_delayed` recipe is only allowed to be used in combination with `--fp8-format hif8`. Combining other fp8-format values (such as `e4m3`) with `hif8_delayed` reports an error during the parameter validation stage.

<table><thead>
  <tr>
    <th width='120'>Feature</th>
    <th>Combination</th>
    <th>Supported</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="3">HiF8 DTS training</td>
    <td rowspan="3">--transformer-impl transformer_engine
    <br> --fp8-format hif8
    <br> --fp8-recipe hif8_delayed </td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="3">HiF8 DTS + Step Recovery</td>
    <td rowspan="3">--transformer-impl transformer_engine
    <br> --fp8-format hif8
    <br> --fp8-recipe hif8_delayed
    <br>(Enabled by default, no additional parameters required)</td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="3">HiF8 DTS + Explicitly Disabling Step Recovery</td>
    <td rowspan="3">--transformer-impl transformer_engine
    <br> --fp8-format hif8
    <br> --fp8-recipe hif8_delayed
    <br> --no-hif8-step-recovery </td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
</table>

## Related Feature References

- **[Megatron Transformer-engine](transformer_engine.md)**: An overall introduction to the TE module, including the HiF8 data format, low-precision training scaling strategies, and a description of the TE module features.
