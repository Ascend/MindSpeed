# swiglu融合优化

## 背景与挑战
SwiGLU（Swish-Gated Linear Unit，Swish门控线性单元激活函数）常见于在LLaMA、LLaMA2、Baichuan等大型语言模型中激活层。然而，由于PyTorch标准库中缺乏直接支持SwiGLU的算子接口，模型通常会以一系列基础算子组合的方式来实现SwiGLU功能，这种方式的执行效率并不理想。

## 解决方法
为应对这一挑战，MindSpeed对SwiGLU操作进行了融合优化，将其封装为一个高性能的融合算子，显著减少了数据在内存间的传输次数以及临时数据的存储需求。算子接口见[link](../ops/swiglu.md)。

## 使用场景
当模型设计中采用了SwiGLU作为MLP层的激活函数，并且训练脚本中已包含以下配置项：
`--swiglu`

## 使用方法
启用SwiGLU融合优化，需在训练脚本中加入以下参数配置：
`--use-fused-swiglu`

mcore分支下仅支持使能该融合算子。

## 使用效果 
在LLaMA2-7B模型下，通过启用融合优化后的SwiGLU算子，内存节省约16.6%，性能提升约4.7%，不仅可以有效降低内存消耗，还能大幅提升模型训练的运行效率。

