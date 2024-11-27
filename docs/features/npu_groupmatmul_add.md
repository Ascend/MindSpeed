# npu_groupmatmul_add融合优化

## 问题分析
MOE大模型训练中，MOE专家开启了梯度累加功能，但累加效率较慢，梯度累加中的 Add 算子占比较高

## 解决方法
MindSpeed将gmm操作和add操作合并成一个融合算子。算子接口见[link](../ops/npu_groupmatmul_add.md)。

## 使用场景
带有MOE的大模型且启用了gmm融合算子（脚本中开启`--moe-grouped-gemm`）, 可使用。

## 使用方法
先安装CANN-NNAL并初始化添加环境
例如：CANN-NNAL默认安装路径
source /usr/local/Ascend/nnal/atb/set_env.sh 

加上`--gemm-gradient-accumulation-fusion `即可调用npu_groupmatmul_add_fp32融合算子。

## 使用效果 
开启融合算子，使用脚本[link](../../tests_extend/system_tests/gpt/pretrain_gpt_megatron_moe_8k.sh)，AlltoAll分支，专家数设置为64，hidden-size减半情况下，性能可提升10.5%。

## 使用限制
1.npu_groupmatmul_add_fp32暂不支持mfu统计

2.融合算子与小算子之间存在精度差异，精度差异的原因是：
小算子dtype变化过程：`bf16*bf16=fp32->bf16->fp32+fp32=fp32`
融合算子dtype变化过程：`bf16*bf16=fp32+fp32=fp32`
差异点在于融合算子dtype做了升精度的操作，故导致精度与小算子存在差异