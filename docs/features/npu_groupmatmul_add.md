# matmul_add融合优化

## 问题分析
MOE大模型训练中，专家开启了梯度累加功能，但累加效率较慢，梯度累加中的 Add 算子占比较高

## 解决方法
MindSpeed将Gmm操作和add操作合并成一个融合算子。算子接口见[link](../ops/npu_groupmatmul_add_fp32.md)。

## 使用场景
带有MOE的大模型均使用。

## 使用方法
先安装CANN-NNAL并初始化添加环境
例如：CANN-NNAL默认安装路径
source /usr/local/Ascend/nnal/atb/set_env.sh 

去掉`--gemm-gradient-accumulation-fusion `即可调用npu_groupmatmul_add_fp32融合算子。

## 使用效果 
开启融合算子，使用脚本[link](../../tests_extend/system_tests/gpt/pretrain_gpt_megatron_moe_8k.sh)，AlltoAll分支，专家数设置为64，性能可提升,8.5%。

## 使用限制
npu_groupmatmul_add_fp32暂不支持mfu统计

