# Megatron MoE alltoall dispatcher分支内存优化

## 背景与挑战

MoE中动态内存比较大，在使用overlap策略后，动态内存会进一步提高，内存墙问题严重。此时，使用普通重计算会因为粒度较粗，加重重计算导致的性能问题。

## 解决方案

针对这种场景，使用重通信，细粒度的重计算和针对性swap进行内存节省，采用计算掩盖重通信和swap，将重计算与未掩盖通信进行隐藏。
当前支持 `alltoall` dispatcher。

- level0 在专家计算部分进行重计算。
- 此处MLP亦包含共享专家部分。
- 在`alltoall`分支中，进行了probs重计算的前移，进一步提高内存节约的效果。

## 使用方法

**后端限制：** 必须设置 `--transformer-impl transformer_engine`，不支持 `local`。

设置 `--moe-zero-memory level0`，并开启 `--moe-alltoall-overlap-comm` 或 `--moe-fb-overlap`。该优化作用于所有 MoE 层。

- 使用 `--moe-fb-overlap` 时，需满足 [FB overlap 使用约束](megatron-moe-fb-overlap.md#使用约束)。
- `level0` 不能与 `--moe-apply-probs-on-input` 或 `--moe-latent-size` 同时使用。

## 适用场景

适用于需要重计算的 MoE `alltoall` dispatcher 场景。
