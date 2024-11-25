# MoE 负载感知内存均衡算法

## 问题分析

MoE在预训练前期负载均衡 aux_loss 未起作用时，token 在专家层的分配不均会导致全局内存负载不均衡问题，并引入大量碎片内存，导致显存波动巨大，并呈现逐步递增的趋势，大集群训练下更容易出现OOM。

## 优化方案

根据模型设定参数（DP/TP/SeqLength/NumExpert等），设定token分布不均的判定阈值，当超过该阈值后执行重计算，避免产生内存不均衡的激活值。

## 使用限制

1. 使用时**建议**开启`--moe-router-load-balancing-type aux_loss`，这样会使得训练过程中token分布快速趋向于平衡。
2. 开启全局重计算后该功能无效。
3. 仅支持`--moe-token-dispatcher-type`是all-gather时可用。

## 使用方法

- 必选参数：脚本中加入`--moe-adaptive-recompute-activation`即可。

- 可选参数：如果希望调节判定阈值，则修改`--moe-adaptive-recompute-activation-scale`即可，默认值为2.0，表示阈值为平均分配在每个专家上的序列的两倍长度。
