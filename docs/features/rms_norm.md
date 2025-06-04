# rms_norm融合优化
## 问题分析
rms_norm常见于LLaMA、LLaMA2、Baichuan等LLM模型中用于归一化，由于torch侧没有提供rms_norm算子的接口，因此在模型中通常是以自定义的形式出现，这种形式的执行效率相对较低。

## 解决方法
MindSpeed将rms_norm操作合并成一个算子，减少数据传输和临时存储。算子接口见[link](../ops/rms_norm.md)。

## 使用场景
模型使用rms_norm作为归一化方式，脚本中设置了`--normalization RMSNorm`。

## 使用方法
设置`--use-fused-rmsnorm`即可调用rms_norm融合算子。mcore分支下仅支持使能该融合算子。

## 使用效果 
开启融合算子可以节省内存，提升性能。
