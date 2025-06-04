## ND_MatMul

## 问题分析

传统的1d张量并行中，输入数据仅在张量并行组内简单复制，造成单卡静态内存较大；同时，attention和ffn的partial输出结果都需要做一次all_reduce，这一部分通信开销较大。

<img src="../../sources/images/megatron_tp.png" alt="megatron_tp" style="zoom:50%;" />

## 解决方案

针对attention和ffn中的矩阵乘，将矩阵乘的参数同时进行行和列切分，即mp=mp_row*mp_col，同时在一个张量并行组内将输入x列切mp份，每张卡只保留输入数据的1/mp，通过插入更小规模的all_gather和reduce_scatter通信算子保证计算的准确性。算法原理图如下：

![nd_matmul](../../sources/images/nd_matmul.png)

## 使用方法

设置`--use-nd-matmul`，打开ND_MatMul特性的总开关。

设置`--nd1-dim1-size`，默认为1，需要确保`--nd1-dim1-size`能够被`--tensor-model-parallel-size`整除。

设置`--nd2-dim1-size`，默认为1，需要确保`--nd2-dim1-size`能够被`--tensor-model-parallel-size`整除。

示例：`--tensor-model-parallel-size`为32，`--nd1-dim1-size`可以设置为2、4、8、16，`--nd2-dim1-size`可以设置为2、4、8、16，出于性能考虑(建议`--nd1-dim1-size`或者`--nd2-dim1-size`大于等于8)，可配置`--nd1-dim1-size`为8、`--nd2-dim1-size`为4。

## 使用效果

降低单卡显存占用效果明显，在`--nd1-dim1-size`或者`--nd2-dim2-size`较大(>8)时，相比megatron TP性能提升。