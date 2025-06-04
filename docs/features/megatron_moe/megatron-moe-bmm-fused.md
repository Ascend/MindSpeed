# Megatron MoE BMM

## 问题分析

针对MoE的drop and pad场景，所有专家上tokens数量相同，使用bmm融合算子(融合前后的通信操作)替换gmm算子能达到更好的效果。

## 解决方案

通过调用bmm通算融合算子(alltoall_allgather_bmm和bmm_reducescatter_alltoall)替换gmm算子及前后的通信操作，达到加速效果。

## 使用方法
在drop and pad场景

前置条件需要设置`--moe-grouped-gemm`

设置`--moe-bmm-mc2`: 表示通过bmm的融合算子计算。

## 使用效果 
在ep=8的场景下，开启融合算子替换，性能提升2%左右。

## 使用限制
1.仅支持megatron_moe的alltoall分支，且开启tp和ep。

2.仅支持昇腾Atlas A3 AI处理器。

3.不支持`--moe-tp-extend-ep`和`--moe-alltoall-overlap-comm`特性。

4.Megatron 010及以上版本不支持。
