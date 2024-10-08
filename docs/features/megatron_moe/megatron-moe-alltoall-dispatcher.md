# Alltoall Dispatcher 分支优化

## 问题分析
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
在使用 mcore MoE 的场景下，开启了 `--moe-token-dispatcher-type alltoall`。

## 使用方法
开启参数 `--moe-permutation-async-comm`。

## 场景限制
由于开启 `--moe-grouped-gemm` 后，专家计算被单一算子合并，因此计算通信并行优化会失效。
