# Alltoall Dispatcher 分支优化

## 背景与挑战

### 1. repeat_interleave 并行

在 Alltoall dispatcher 分支中，调用了 repeat_interleave 算子，此算子只使用了单个 block dim 在单个下发流上进行串行计算，且耗时较长，算子的输出也是在 alltoall、permute、alltoallv 之后才用到。

### 2. 计算通信并行

在 alltoall 分支中的 permutation 函数最后会进行 allgather 操作，对所有 tokens 被切分的 H 维进行补全，然后再对数据分块进行专家计算。此项操作为串行操作，但各专家间的 tokens 并没有存在依赖关系，可修改为并行操作。

## 解决方案

### 1. repeat_interleave 并行

通过新建一条下发流，将 repeat_interleave 算子调用分到新的流上，在 block dim 资源充足的情况下，可进行两个算子的并行计算，节省耗时。

### 2. 计算通信并行

可按照每个专家需要的 tokens 进行切分，然后逐个对 tokens 进行 allgather 通信 + 专家计算，由于第一个专家计算只依赖第一个通信，专家之间无依赖关系，因此在做第一个专家计算的时候可同步进行第二专家的通信，达到计算和通信并行。

## 使用场景

**后端限制：** 开启 Grouped GEMM、Alltoall 通信隐藏或 FB overlap 时，必须设置 `--transformer-impl transformer_engine`。

在使用 mcore MoE 的场景下，开启了 `--moe-token-dispatcher-type alltoall`。

## 使用方法

`--moe-token-dispatcher-type` 的类型支持情况如下：

| 类型 | 支持情况 |
| --- | --- |
| `alltoall` | 支持；可按约束选择 Alltoall 通信隐藏、FB overlap、MC2 或 permute 融合。 |
| `allgather` | 支持，默认类型；使用专用的 [Allgather 配置](megatron-moe-allgather-dispatcher.md)，不支持动态序列长度。 |
| `flex`（`deepep/hybridep`） | 不支持，尚未完成 NPU 适配。 |

设置 `--moe-token-dispatcher-type alltoall` 选择 Alltoall dispatcher。计算与通信重叠的配置见 [Alltoall 通信隐藏](megatron-moe-alltoall-overlap-comm.md)。

如需使用 `--moe-alltoall-mc2`，须为 dropless 场景，设置 `--expert-tensor-parallel-size 1`，且不设置 `--moe-expert-capacity-factor`。

## 使用效果

开启后可降低训练时长，提高性能。
