# Efficient-MOE
## 1. MOE token dropless性能优化

### 问题分析

现有MoE无token丢弃方案在训练过程中实时all reduce通信全局最大专家容量作为所有rank的容量，引入频繁的all reduce开销

### 优化方案

采用滑窗预测，无需每一个step都进行all reduce通信，all reduce通信频次降低为1/window_size，对于超过预测值场景，使用实际最大专家容量值替换预测capacity。

#### 优化思路:

1. MoE模型训练过程中capacity具有一定的连续性，维护一个滑动窗口来保存近几次统计的capacity来预估下一个窗口的capacity。
2. 在Gate中不再每一个step都直接进行all reduce统计全局最大capacity，而是各个进程先判断当前的capacity能否满足不丢弃token，通过reduce统计全局的判断信息，若都能满足则无需进行all reduce通信，否则进行all reduce通信取得实际max capacity.
<p align="center"> <img src="../../sources/images/moe_dynamic_padding_a.png" height="300px" width="600px"></p>

### 使用场景

在MoE模型中的无token丢弃训练场景，提升训练速度。

### 使用方法

设置`--moe-no-drop`: 表示开启MoE无token丢弃训练模式，Top1 Gate &Top2 Gate均已支持， 请搭配aux loss/sinkhorn负载均衡方式使用，避免无token丢弃场景负载均衡情况劣化严重

设置`--moe-dynamic-padding`: 表示开启MoE无token丢弃训练优化，需要搭配`--moe-no-drop`同时开启，
附加功能

设置`--moe-use-sinkhorn`: 表示开启sinkhorn负载均衡功能


### 使用效果

在保持精度的同时提升训练速度。

训练模型：Mixtral(4层)

精度对比图如下：
<p align="center"> <img src="../../sources/images/moe_dynamic_padding_b.png" height="300px" width="800px"></p>

top2 多种并行方式 提速效果：
<p align="center"> <img src="../../sources/images/moe_dynamic_padding_c.png" height="400px" width="800px"></p>

top1 多种并行方式 提速效果：
<p align="center"> <img src="../../sources/images/moe_dynamic_padding_d.png" height="400px" width="800px"></p>

同时开启此优化减少显存占用3%：
<p align="center"> <img src="../../sources/images/moe_dynamic_padding_e.png" height="500px" width="800px"></p>


## 2. MoE 负载感知内存均衡算法

### 问题分析

MOE在预训练前期负载均衡aux_loss未起作用时，token在专家层的分配不均会导致全局内存负载不均衡问题，并引入大量碎片内存，导致显存波动巨大，并呈现逐步递增的趋势，大集群训练下更容易出现OOM。

### 优化方案

根据模型设定参数（DP/TP/SeqLength/NumExpert等），设定token分布不均的判定阈值，当超过该阈值后执行重计算，避免产生内存不均衡的激活值。

### 使用限制

1. 使用时**建议**开启`--moe-router-load-balancing-type aux_loss`，这样会使得训练过程中token分布快速趋向于平衡。
2. 开启全局重计算后该功能无效。

### 使用方法

- 必选参数：脚本中加入`--moe-adaptive-recompute-activation`即可。

- 可选参数：如果希望调节判定阈值，则修改`--moe-adaptive-recompute-activation-scale`即可，默认值为2.0，表示阈值为平均分配在每个专家上的序列的两倍长度。
