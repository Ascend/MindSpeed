# megatron0.7.0版本长稳测试出现grad norm为nan
## 问题现象
在megatron0.7.0版本中，采用mindspeed自定义`--tokenizer-type PretrainedFromHF`， 长稳测试一定步数后发现loss抖动异常最终出现grad norm为nan的问题，报错示例如下：
```
2024-09-18 11:14:247 iteration	427/	5000  consumed samples:	6832	elapsed time per iteration (
ms): 209.8 | Learning rate: 1.229919E-06 | global batch size:   16 | Lm loss: 8.567080E+00 | loss scale: 1.0 | gr
ad norm: 35.518 | number of skipped iterations:   О | number of nan iterations: 0
[2024-09-18 11:14:25] iteration	428/   5000] consumed samples:	6848	elapsed time per iteration (
ms): 210.5 | Learning rate: 1.229826E-06 | global batch size: _ 16 | lm loss: 7.180392E+00 | loss scale: 1.0 | gr
ad norm: 36.838 ] number of skipped iterations:   О | number of nan iterations:
Traceback (most recent call last):
File "pretrain_gpt.py”, line 247, in <module>
pretrain(
File "/home/Megatron-LM/megatron/training/training.py”, Line 274, in pretrain
iteration, num floating point operations so far = train(
File "/home/Megatron-LM/megatron/training/training.py”, Line 1027, in train
train step(forward step func,
File "/home/Megatron-LM/megatron/training/training.py”, Line 550, in train_step
losses reduced = forward backward func(
File "/home/Megatron-LM/megatron/core/pipeline parallel/schedules.py”, line 1400, in forward backward
pipelining without interleaving
config.finalize model grads func(
File "/home/Megatron-LM/megatron/core/distributed/finalize model_grads.py”, Line 113, in finalize mode
l grads
model chunk.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/distributed data parallel.py”, Line 248, in finish_g
rad sync
buffer.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and_grad buffer.py”, Line 513, in finish_grad
sync
bucket.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and_grad buffer.py”, Line 151, in finish_grad
sync
self.start grad sync()
File “/home/Megatron-LM/megatron/core/distributed/param and grad buffer.py”, Line 114, in start_grad_s
ync
assert not norm.isnan( ), (
AssertionError: Rank 13: found NaN in local grad norm in backward pass before data-parallel communication collectie
ve. Device: 5, node: node-15-11
```

## 问题根因

1. 问题场景使用的数据集生成时，增加了`--append-eod`参数，这会让每个数据sample末尾增加一个eos结束标志位；
2. megatron0.7.0对数据集提取过程增加了pad功能（在`class GPTDataset`类中），`PretrainedFromHF`模式下，会将pad标志位与eos标志位配成相同值（`pad_token_id == eos_token_id`）。loss_mask中会去掉pad标志位，但实际去掉的都是eos标志位。
3. 以上两个原因综合导致了grad norm为nan的问题，这个问题是megatron原生问题，相同配置下实测GPU中也会报错。


## 解决方案

在`--tokenizer-type PretrainedFromHF`模式下，不使用`--append-eod`生成数据集