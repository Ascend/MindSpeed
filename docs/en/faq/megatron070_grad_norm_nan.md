# GradNorm Becomes NaN in Long-Duration Stability Test of Megatron-LM 0.7.0

## Symptom

In Megatron-LM 0.7.0, when using the mindspeed custom `--tokenizer-type PretrainedFromHF`, after a certain number of steps in a long-duration stability test, the loss fluctuates abnormally and eventually the grad norm becomes NaN. An example error is shown below:

```bash
2024-09-18 11:14:247 iteration 427/ 5000  consumed samples: 6832 elapsed time per iteration (
ms): 209.8 | Learning rate: 1.229919E-06 | global batch size:   16 | Lm loss: 8.567080E+00 | loss scale: 1.0 | gr
ad norm: 35.518 | number of skipped iterations:   О | number of nan iterations: 0
[2024-09-18 11:14:25] iteration 428/   5000] consumed samples: 6848 elapsed time per iteration (
ms): 210.5 | Learning rate: 1.229826E-06 | global batch size: _ 16 | lm loss: 7.180392E+00 | loss scale: 1.0 | gr
ad norm: 36.838 ] number of skipped iterations:   О | number of nan iterations:
Traceback (most recent call last):
File "pretrain_gpt.py", line 247, in <module>
pretrain(
File "/home/Megatron-LM/megatron/training/training.py", Line 274, in pretrain
iteration, num floating point operations so far = train(
File "/home/Megatron-LM/megatron/training/training.py", Line 1027, in train
train step(forward step func,
File "/home/Megatron-LM/megatron/training/training.py", Line 550, in train_step
losses reduced = forward backward func(
File "/home/Megatron-LM/megatron/core/pipeline parallel/schedules.py", line 1400, in forward backward
pipelining without interleaving
config.finalize model grads func(
File "/home/Megatron-LM/megatron/core/distributed/finalize model_grads.py", Line 113, in finalize mode
l grads
model chunk.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/distributed data parallel.py", Line 248, in finish_g
rad sync
buffer.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and_grad buffer.py", Line 513, in finish_grad
sync
bucket.finish grad sync()
File "/home/Megatron-LM/megatron/core/distributed/param and_grad buffer.py", Line 151, in finish_grad
sync
self.start grad sync()
File “/home/Megatron-LM/megatron/core/distributed/param and grad buffer.py", Line 114, in start_grad_s
ync
assert not norm.isnan( ), (
AssertionError: Rank 13: found NaN in local grad norm in backward pass before data-parallel communication collectie
ve. Device: 5, node: node-15-11
```

## Root Cause

1. When generating the dataset used in the problem scenario, the `--append-eod` parameter was added, which appends an eos end-of-sequence token to the end of each data sample.
2. Megatron 0.7.0 added a padding feature to the dataset extraction process (in the `class GPTDataset` class). In `PretrainedFromHF` mode, the pad token and eos token are set to the same value (`pad_token_id == eos_token_id`). The loss_mask removes the pad token, but what is actually removed are all eos tokens.
3. The combination of the above two causes leads to the grad norm being NaN. This is a native issue in Megatron, and the same error occurs on actual GPUs under identical configurations.

## Solution

In `--tokenizer-type PretrainedFromHF` mode, do not use `--append-eod` to generate the dataset.
