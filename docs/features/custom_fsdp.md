# Megatron 全分片数据并行（Fully Sharded Data Parallel, FSDP）
## 背景与挑战
随着大模型权重增加，需要进一步提高显存利用效率。之前的Zero-1操作只对优化器的状态在DP域内进行切分，但是对模型权重和梯度并未做切分。这使得模型权重和梯度占用静态显存的大头。FSDP能将权重和梯度也在DP域内切分，从而进一步降低静态显存的大小。

## 解决思路
令每个DP rank只含参数的分片，在每一块权重前向前，在DP域内做All Gather，前向结束后释放得到的权重，仅保留分片。在反向前先All Gather，获得完整的权重，反向后做Reduce Scatter，对所有DP rank的梯度进行加和，同时只保留该DP rank对应的部分分片。

## 使用场景
在DP>1时，模型权重占显存较多，想要进一步切分权重和梯度以节省显存

## 使用方法
为了开启全分片数据并行，需要加入以下配置
```bash
--use-custom-fsdp
--data-parallel-sharding-strategy optim_grads_params
--no-gradient-accumulation-fusion
--use-distributed-optimizer
```
需要关闭CUDA_MAX_CONNECTIONS

```bash
unset CUDA_MAX_CONNECTIONS
```

## 使用影响
模型和权重被进一步切分，显存下降。但是因为每次前向反向都新增了通信，性能会下降。

## 注意事项
MindSpeed适配该特性基本功能，不建议与仓上其他特性组合，使用方法[参考脚本](../../tests_extend/system_tests/feature_tests/custom_fsdp.sh)