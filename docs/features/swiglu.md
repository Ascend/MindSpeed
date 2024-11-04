# swiglu融合优化

## 问题分析
swiglu常见于LLaMA、LLaMA2、Baichuan等大模型中的激活层，由于torch侧没有提供swiglu算子的接口，因此在模型中通常是以小算子的形式出现，这种形式的执行效率相对较低。

## 解决方法
MindSpeed将swiglu操作合并成一个融合算子，减少数据传输和临时存储。算子接口见[link](../ops/swiglu.md)。

## 使用场景
模型使用swiglu作为MLP层激活函数，脚本中设置了`--swiglu`。

## 使用方法
设置`--use-fused-swiglu`即可调用swiglu融合算子。mcore分支下仅支持使能该融合算子。

## 使用效果 
开启融合算子可以节省内存，提升性能。

