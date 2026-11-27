# RMSNorm融合优化

## 背景与挑战

在诸如LLaMA、LLaMA2和Baichuan等大语言模型（LLM）中，RMSNorm（Root Mean Square Norm）作为一种归一化技术被广泛应用。然而，由于PyTorch框架本身未直接提供RMSNorm算子接口，模型中常以自定义形式实现RMSNorm，这在一定程度上影响了执行效率。

## 解决方法

MindSpeed针对上述情况，对RMSNorm算子进行了融合优化，将其集成到单个算子中，有效减少了数据传输次数和临时内存需求。算子接口见[rms_norm](../ops/rms_norm.md)。

## 使用场景

当模型选用RMSNorm作为其归一化方法，并且训练脚本中已包含以下配置时：
`--normalization RMSNorm`。

## 使用方法

启用RMSNorm融合优化，需在训练脚本中加入以下参数配置：
`--use-fused-rmsnorm`
需要注意的是，Mcore分支下该算子仅支持使能融合特性。

## 使用效果 

在LLaMA2-7B模型下，启用融合算子RMSNorm后，内存节省约12%，性能提升约2.7%，不仅能够有效节省内存资源，还能提升模型训练效率。
