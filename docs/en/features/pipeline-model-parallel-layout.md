# Megatron Custom Pipeline Layout

<!-- md-trans-meta sourceCommit=94e8ac1c0d003c914fcb5b2fd734093aaee76a51 translatedAt=2026-08-17T08:08:13.728Z pushedAt=2026-08-17T08:17:07.887Z -->

## Background and Challenges

Megatron's Pipeline Parallelism (PP) and Virtual Pipeline Parallelism (VPP) split a model evenly by the number of decoder layers by default. For conventional dense models, this approach usually meets training requirements; however, in models that include embedding, loss, MTP, or decoder layers with different computational costs, even splitting may cause imbalanced computational loads across different pipeline stages, thereby increasing pipeline bubbles and reducing overall throughput.

For example, when MTP is enabled or embedding and loss computations are incorporated into pipeline stages, the first and last stages may bear additional computation; in MoE models, the number of decoder layers in different stages may also need to be split non-uniformly according to the model structure or performance profiling results. In such cases, relying solely on `--num-layers-per-virtual-pipeline-stage` cannot express the specific layer types and layer counts of each stage.

## Solution

MindSpeed supports customizing the layer types held by each stage in PP/VPP, enabled through the `--pipeline-model-parallel-layout` parameter.

This feature can:

- Explicitly specify the pipeline stages where embedding, decoder, MTP, and loss layers reside.
- Support different VPP chunks on the same PP rank holding different numbers of decoder layers.
- Automatically derive the VPP size based on the number of stages in the layout.
- Work with `--moe-fb-overlap` (MoE cross-microbatch forward/backward communication overlap) in restricted scenarios.

## Layout String Format

`--pipeline-model-parallel-layout` uses a single string to describe all pipeline stages, with stages separated by `|`. The string is expanded in forward computation order: first list all PP stages on VPP rank 0, then all PP stages on VPP rank 1, and so on.

The supported layer types are as follows:

| Character | Meaning |
| --- | --- |
| `E` | Embedding layer |
| `t` | Transformer decoder layer |
| `m` | MTP layer |
| `L` | Loss layer |

The format rules are as follows:

- `|` is used to separate stages.
- `,` is used only to improve readability and is ignored during parsing.
- `x*N` means repeating a single character, for example, `t*3` is equivalent to `ttt`.
- `(pattern)*N` means repeating a layout segment, for example, `(tt|)*2` is equivalent to `tt|tt|`.
- Consecutive `||` can represent an empty stage, but empty decoder chunks are not supported when used together with `--moe-fb-overlap`.

## Use Cases

This feature applies to the following scenarios:

- You need to explicitly place embedding, loss, or MTP layers on specified pipeline stages.
- The computational load is uneven across different pipeline stages, and you need to balance the load by using a non-uniform number of decoder layers.
- In VPP scenarios, you need different chunks on the same rank to have different numbers of layers.
- In MoE models, you may want to use custom PP/VPP splitting in combination with `--moe-fb-overlap`.

## Usage

Add the `--pipeline-model-parallel-layout` parameter to the startup script to enable this feature.

Take a model with `PP=2`, `VPP=2`, 8 decoder layers, and 1 MTP layer as an example:

```shell
--pipeline-model-parallel-size 2
--pipeline-model-parallel-layout Ett|tt|ttt|tmL
--num-layers 8
--mtp-num-layers 1
```

This layout contains 4 stages, and `4 / PP = 2`, so `VPP=2` is automatically inferred. The corresponding stage distribution is as follows:

| PP rank | VPP rank 0 | VPP rank 1 |
| --- | --- | --- |
| 0 | `Ett` | `ttt` |
| 1 | `tt` | `tmL` |

The total number of decoder layers is `2 + 2 + 3 + 1 = 8`, the total number of MTP layers is 1, and embedding and loss each appear once.

If only PP is used without VPP, the number of stages in the layout should equal `--pipeline-model-parallel-size`. For example:

```shell
--pipeline-model-parallel-size 2
--pipeline-model-parallel-layout Etttt|ttttL
--num-layers 8
```

## Using with `--moe-fb-overlap`

When `--pipeline-model-parallel-layout` and `--moe-fb-overlap` are enabled at the same time, MindSpeed uses layout-aware VPP scheduling logic and supports inconsistent decoder layer counts across different chunks on the same rank. In this scenario, forward-backward overlap is executed according to the actual number of layer graphs of the current forward chunk and backward chunk; when the layer counts on the two sides differ, overlap is first performed on the pairable layers, and then the remaining layers are processed.

A typical MoE configuration is as follows:

```shell
--pipeline-model-parallel-size 2
--pipeline-model-parallel-layout Ett|tt|ttt|tmL
--num-layers 8
--mtp-num-layers 1
--expert-model-parallel-size 2
--expert-tensor-parallel-size 1
--num-experts 8
--moe-grouped-gemm
--moe-token-dispatcher-type alltoall
--moe-fb-overlap
```

When using this combination, the original constraints of `--moe-fb-overlap` must be satisfied, for example, using the `alltoall` dispatcher, enabling `--moe-grouped-gemm`, setting `--expert-tensor-parallel-size=1`, and `--expert-model-parallel-size > 1`.

## Usage Constraints

The following constraints apply when using `--pipeline-model-parallel-layout`:

1. The number of stages in `layout` must be divisible by `--pipeline-model-parallel-size`.
2. `layout` must contain exactly one embedding layer and one loss layer.
3. The number of decoder layers in `layout` must match `--num-layers`.
4. If MTP is used, the number of `m` in `layout` must match `--mtp-num-layers`, and decoder layers must be placed before MTP layers.
5. Encoder layers are not currently supported.
6. It cannot be configured together with `--num-layers-per-virtual-pipeline-stage` or `--num-virtual-stages-per-pipeline-rank`.
7. It cannot be used together with `--pipeline-num-transformer-layers`, `--noop-layers`, or `--schedules-method dualpipev`.
8. It is not currently supported to be used together with `--recompute-in-bubble` or `--recompute-in-advance`.
9. When the layout derives VPP, using it together with `--optimize-send-recv-comm` is not supported yet.
10. When used together with `--moe-fb-overlap`, empty decoder chunks are not supported yet, and the combination of `--noop-layers + --pipeline-model-parallel-layout + --moe-fb-overlap` is not supported yet either.

## Effects

With a custom pipeline layout, you can place embedding, loss, MTP, and decoder layers on different stages according to their actual computational load, reducing load imbalance and pipeline waiting in PP/VPP. For MoE scenarios, this feature can also work with `--moe-fb-overlap` to continue performing cross-microbatch forward-backward communication overlap under non-uniform chunk layer counts, thereby improving the flexibility of pipeline scheduling.
