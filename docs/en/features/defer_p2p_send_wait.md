# Deferring the Forward P2P Send Wait in VPP

## Function and Principle

When output pseudo-release is enabled, the interleaved pipeline schedule of Megatron Core 0.18 waits for the current
forward send to complete before releasing. This step may establish a communication-to-computation dependency early, shrinking the overlap window between the P2P send and subsequent
computation.

MindSpeed's `defer-p2p-send-wait` defers the wait for the current send until after the next managed
forward send is submitted; the last send is finalized when this schedule exits. The receive wait is still handled by the original schedule,
and the computation order, communication content, and number of communications remain unchanged.

This optimization targets the native VPP interleaved schedule on the NPU, and its main goal is to improve communication-computation overlap.
It does not promise a reduction in peak memory, and it is not a memory optimization controlled by the MoE fast ops switch.

## Usage

`--defer-p2p-send-wait` is enabled by default and is an optimization level 2 feature.
For it to actually take effect, `optimization_level >= 2` is required and all applicability conditions below must be met.
Explicitly adding this parameter does not bypass the compatibility checks; use `--no-defer-p2p-send-wait` to disable it.

The following is an example of a PP/VPP configuration added to an existing valid training command; the remaining model, parallelism, and training
parameters still need to be set according to the actual task:

```bash
--optimization-level 2 \
--num-layers 16 \
--pipeline-model-parallel-size 2 \
--num-layers-per-virtual-pipeline-stage 4
```

In this example, each PP rank has two virtual pipeline stages. Do not set another VPP/layout
specification method at the same time, and do not add `--no-overlap-p2p-communication`. The final configuration must also satisfy
`batch_p2p_comm=False` and `deallocate_pipeline_outputs=True`.

When comparing the switch, keep the original configuration and add only the following to the disabled group:

```bash
--no-defer-p2p-send-wait
```

## Applicability Conditions and Automatic Fallback

After complete argument validation, the enabled state is retained only if all of the following conditions are met:

| Condition | Requirement |
| --- | --- |
| Pipeline parallelism | `pipeline_model_parallel_size > 1` |
| Virtual pipeline | Derived `virtual_pipeline_model_parallel_size > 0` |
| P2P communication overlap | `overlap_p2p_comm=True` |
| Batch P2P | `batch_p2p_comm=False` |
| Output pseudo-release | `deallocate_pipeline_outputs=True` |
| Schedule selection | `schedules_method=None`, using the native interleaved schedule |
| Incompatible features | None of the switches below are enabled |

The incompatible switches are:

- `moe_fb_overlap`
- `use_multiparameter_pipeline_model_parallel`
- `recompute_in_bubble`, `recompute_in_advance`
- `tp_2d`
- `variable_seq_lengths`
- `optimize_send_recv_comm`
- `dist_train`

When a condition is not met, the validation automatically sets `args.defer_p2p_send_wait` to `False` and continues
with the original schedule. The PP/VPP values may use convertible numeric strings, and if they cannot be converted the feature is also disabled.
For example, a configuration with PP2 but VPP not enabled does not apply this optimization.

Patch installation occurs before complete argument validation. Registration only checks the user switch, to avoid treating a not-yet-derived VPP
or an early string argument as incompatible. Each schedule entry reads the final switch once; when it is disabled, the original schedule is called directly,
and the send is not taken over.

During execution, when not in a managed schedule, when P2P overlap is not enabled, when there is no output to send, when the output is not
a single NPU Tensor, or when the communication configuration uses batch P2P / disables output pseudo-release, this invocation
goes directly through the original communication interface.

The automatic fallback only handles these pre-determinable applicability conditions; it does not swallow errors in the original training configuration, nor does it
catch exceptions after a send has already been submitted and re-execute the communication.

## Lifecycle of the Send Storage

Megatron's output pseudo-release replaces the `.data` of the output tensor. Saving only the same Tensor's
Python reference cannot protect the original send storage, so the implementation manages asynchronous sends in the following order:

1. Before submission, create independent Tensor metadata via `detach()` that holds the original storage;
   do not copy the send payload, and do not additionally hold its autograd graph.
2. Call the original `send_forward_recv_forward` and take over the returned `send_next` Work.
   Remove only this item from the handles returned to the scheduler, and keep all receive waits.
3. After the current send is submitted, call `wait()` on the previous managed send, and then release the corresponding holder.
   In normal steady-state operation, keep the most recent send; between submitting a new send and finalizing the previous one, two sends are briefly held.
4. `Work.wait()` may only add an event dependency to the current NPU stream, so before discarding the holder,
   call `record_stream()` on the send tensor to protect the storage until the work on the related stream completes.
5. When the schedule exits normally, returns from forward-only, or an exception occurs, finalize the remaining sends via `finally`
   and restore the schedule context.

The normal path does not add any payload copies, communication calls, or full-device synchronizations. If the submission process throws an exception and fails to return
a Work, the cleanup path uses a device synchronization to confirm the storage is safe; if the wait or cleanup fails, the error continues to propagate,
and the related storage is retained until the process exits. An explicit error is also raised when the original interface does not return the expected `send_next` Work.

## Validation and Performance Observation

When comparing, fix the model, PP/VPP layout, number of microbatches, precision, recomputation configuration, and random seed, then run
the default enabled group and the explicitly disabled group separately:

- Check the value of `defer_p2p_send_wait` in the final arguments to confirm the optimization actually takes effect.
- Compare the loss, gradients, or parameter updates to confirm the training results meet the tolerance of the precision in use.
- After warmup, compare the steady-state step time and throughput; observe the overlap of the forward send and subsequent computation in the timeline.
- Also record peak allocated / reserved. Extending the lifetime of the send storage may increase local
  memory usage, so you cannot infer a memory decrease or a guaranteed throughput improvement merely from the wait being deferred.

The benefit depends on the pipeline layout, communication duration, and the amount of overlapable computation, and requires validation on actual multi-card NPUs.

## Implementation Location

<!-- - Argument and compatibility validation:
  [`defer_p2p_send_wait.py`](../../../mindspeed/features_manager/pipeline_parallel/defer_p2p_send_wait.py)
- Schedule wrapping and send storage management:
  [`defer_p2p_send_wait.py`](../../../mindspeed/core/pipeline_parallel/defer_p2p_send_wait.py) -->

The implementation is installed by MindSpeed on Megatron's schedule and P2P interfaces, and the patch does not need to be repeatedly
installed in TENPU.
