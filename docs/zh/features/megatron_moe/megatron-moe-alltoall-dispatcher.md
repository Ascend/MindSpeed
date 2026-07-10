# Alltoall Dispatcher 分支优化

## 背景与挑战

**repeat_interleave并行**

在Alltoall dispatcher分支中，调用了repeat_interleave算子，此算子只使用了单个block dim在单个下发流上进行串行计算，且耗时较长，算子的输出也是在alltoall、permute和alltoallv之后才用到。

**计算通信并行**

在alltoall分支中的permutation函数最后会进行allgather操作，对所有tokens被切分的H维进行补全，然后再对数据分块进行专家计算。此项操作为串行操作，但各专家间的tokens并没有存在依赖关系，可修改为并行操作。

## 解决方案

**repeat_interleave并行**

通过新建一条下发流，将repeat_interleave算子调用分到新的流上，在block dim资源充足的情况下，可进行两个算子的并行计算，节省耗时。

**计算通信并行**

可按照每个专家需要的tokens进行切分，然后逐个对tokens进行allgather通信和专家计算，由于第一个专家计算只依赖第一个通信，专家之间无依赖关系，因此在做第一个专家计算的时候可同步进行第二专家的通信，达到计算和通信并行。

## 使用场景

在使用mcore MoE的场景下，开启了`--moe-token-dispatcher-type alltoall`。

## 使用方法

开启参数`--moe-permutation-async-comm`。

> [!NOTE]
>
> 由于开启`--moe-grouped-gemm`后，专家计算被单一算子合并，因此计算通信并行优化会失效。

## 使用效果

开启后可降低训练时长，提高性能。
