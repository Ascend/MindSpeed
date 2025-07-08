# Megatron MoE Grouped GEMM (GMM)

## 背景与挑战

针对MoE单卡多专家计算，存在细碎的专家计算操作与通信，通过Grouped GeMM（Grouped General Matrix Multiplication）算子对多专家计算进行合并，提升MoE单卡多专家训练性能。

## 解决方案

通过调用 gmm 融合算子，对多个专家计算进行融合，达到加速效果。

## 使用方法

设置`--moe-grouped-gemm`: 表示开启Grouped GEMM计算。
支持MoE allgather, alltoall, alltoall_seq dispatcher.

## 效果说明

典型场景：

- EP变小导致单卡专家数量增大 & DeepSeek MoE专家数量较多等场景。
- DeepSeek MoE finegrained expert单个专家较小 & FFN规模不大 & TP变大导致单卡切分的计算变小。

1. 随着FFN规模提升，计算不再细碎，单专家计算效率提升，Grouped GEMM 收益变小。

表1：grok模型FFN大小和性能加速对比

|ffn_hidden_size| 32768 | 16384| 8192| 4096|
|--|--|--|--|--|
|baseline|2280|1780|1537|1446|
|GEMM|2416|1719|1448|1331|
|性能提升|-5.30%|3.53%|6.12%|8.60%|


2. TP越大，EP越小，收益更大。
   
表2：Mixtral8*7B模型配置不同性能收益

|配置| tp4 ep2 16expert | tp4 ep2 8expert | tp2 ep4 16expert| tp2 ep4 8expert|
|--|--|--|--|--|
|baseline|27969|20127|11976|13981|
|GEMM|19415|17361|11049|14290|
|性能提升|44.06%|17.93%|8.39%|-2.19%|

## 注意事项
1. megatron原生不支持`--moe-grouped-gemm`在开启`--bf16`的场景下使用。
2. 通过`--moe-grouped-gemm`参数使能Grouped GEMM计算时会调用npu_gmm融合算子

算子输入输出格式如下：
```
y = npu_gmm(x, weight, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None)
```

输入：
- x：必选输入，为tensor，数据类型float16, bfloat16, float32
- weight：必选输入，为tensor，数据类型float16, bfloat16, float32
- bias：可选输入，为tensor，数据类型float16, float32, 默认值为None。训练场景下，仅支持bias为None
- group_list：可选输入，数据类型list[int64], tensor，默认值为None。
- group_type：可选输入，数据类型int64，代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则group_type取值-1：不分组，0：m轴分组，1：n轴分组，2：k轴分组，默认值为0。
- gemm_fusion:可选输入，为bool，数据类型True，False，用于反向累加梯度的时候使能GMM+ADD融合算子，默认值为False。
- original_weight:可选输入，为tensor，数据类型float16, bfloat16, float32，用于获取view之前的weight的main_grad用于GMM+ADD中梯度累加功能，默认值为None。

输出：
- y：必选输出，数据类型float16, bfloat16, float32

该算子在非量化场景仅支持以下输入输出类型的组合，若将参数指定为其他不支持类型可能会导致算子报错，影响训练效率：

| x        | weight   | bias    | group_list         | group_type | gemm_fusion | original_weight | y                                 |
|----------|----------|---------|--------------------|------------|-------------|-----------------|-----------------------------------|
| float16  | float16  | float16 | list[int64]或tensor | int64      | bool        | float16         | float16                           |
| bfloat16 | bfloat16 | float32 | list[int64]或tensor | int64      | bool        | bfloat16        | bfloat16                          |
| float32  | float32  | float32 | list[int64]或tensor | int64      | bool        | float32         | float32（仅x、weight、y都为单tensor场景支持） |