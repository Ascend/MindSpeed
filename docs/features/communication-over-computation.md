# 计算通信并行 CoC (Communication Over Computation)

## 问题分析

大模型训练过程中，其ColumnParallelLinear和RowParallelLinear部分的前反向均存在相互毗邻、顺序依赖的计算通信组合，计算为Matmul，而通信则为AllReduce（不开启序列并行）或AllGather和ReduceScatter（开启序列并行）。这些计算通信的组合因为存在顺序依赖（即后一个的输入是前一个输出），常常被串行执行，但这时候计算和通信流都存在一定的空闲等待时间，该过程的执行效率没有被最大化。

## 解决方案

通过将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖。

### 解决思路

#### Python脚本侧实现
将张量进行进一步切分（2/4/8份），通过Python脚本的方式实现每个子tensor之间计算和通信的并行，从而增大计算和通信流的利用率；


#### 融合算子实现
基于MTE远端内存访问能力，以融合大Kernel方式在算子实现的内部将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖；

## 使用场景
该特性目前主要用于训练场景，当Attention模块和MLP模块串行执行且计算通信存在顺序依赖与位置毗邻关系时适用。

使用Python脚本侧实现时，对Matmul左矩阵的m轴有一定要求，必须是切分数（2/4/8）的倍数，且不适用于计算与通信片段耗时相差较大的情况。需要注意的是，脚本侧实现在切分矩阵、切分数量较大时，容易出现host bound问题，从而不能得到预期的收益。支持ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER三个通信场景，支持灵活设置先通信或先计算。

对于计算通信融合算子，目前已支持：
1. MATMUL_ALL_REDUCE融合算子（先计算后通信）及其确定性计算；
2. MATMUL_REDUCE_SCATTER融合算子（先计算后通信）及其确定性计算；
3. ALL_GATHER_MATMUL, ALL_GATHER_MATMUL_V2融合算子（先通信后计算）（V2版本接口支持ALL_GATHER中间结果获取）；
4. 量化场景：MATMUL_ALL_REDUCE融合算子支持fp16格式的w8A16伪量化，粒度包含per tensor / per channel / per group；

## 使用方法 —— 在MindSpeed中进行整网训练

计算通信并行优化算法通过在训练脚本(MindSpeed/tests_extend/xxx/xxx.sh)中配置环境变量来进行使能，需要安装mindspeed。

当前计算通信并行有两种实现方法：python脚本使能、融合算子使能，两者选其一即可。两个方式都需要替换原Megatron框架中的ColumnParallelLinear和RowParallelLinear这两个class的forward函数，替换脚本已经根据MindSpeed指定Megatron版本进行编码和适配，位于mindspeed/core/tensor_parallel/lcal_coc/目录下。

请根据需要选择下列两种场景中的一个进行使用。

设置--use-ascend-coc使能计算通信并行功能，使用方式通过如下变量进行设置：

### 1. 使用通过Python脚本使能的计算通信并行特性

```shell
--use-ascend-coc 
--coc-parallel-num 2 # 或者4，或者8
```

### 2. 使用通过融合算子使能的计算通信并行特性
注意：计算通信并行融合算子需要安装ATB后才能使用！

ATB安装方法：

- 二进制包安装：安装CANN-NNAL包之后, source /usr/local/Ascend/nnal/atb/set_env.sh
```shell
--use-ascend-coc
--coc-fused-kernel # 注意：当前只支持TP=8的场景！
```

融合算子的环境变量拥有更高优先级，即当 coc-parallel-num > 1 且 使能coc-fused-kernel时，前者不会生效。


## CFG自定义方法

用户可以自定义mindspeed/core/tensor_parallel/lcal_coc/user_config.py中的coc_cfgs字典，来达到自定义COC的部分配置。

【只对通过Python脚本使能的计算通信并行实现适用】
'matmul_soc_friendly'：是否对输入matmul的张量做transpose/padding操作，使其以NPU亲和的shape进入Matmul算子从而获得一定性能提升，默认为True；
'customized_coc': 自定义指定shape的matmul的COC切分份数，默认为{}。如果需要设置指定shape的matmul的CoC切分份数为1（不开COC）或与coc-parallel-num不同的值，可以按照这个例子设置：
'customized_coc': {"[16384, 5120, 1920]": 8, "[16384, 1920, 5120]": 1}

【只对通过融合算子使能的计算通信并行实现适用】
'enable_coc_in_column_backward': 是否在ColumnParllelLinear的反向中使用COC（ColumnParallelLinear的反向中本来就有非互相依赖的计算通信并行），默认为False；

【对脚本实现和融合算子实现都适用】
'recompute_all_gather': 是否在ColumnParallelLinear的反向中重新计算all gather，默认为True。若为False，则将从前向保存all gather结果到反向，会减少反向计算时间但是会增加训练过程中的峰值内存占用；

## COC融合算子使用效果

在BLOOM 7B模型中获得端到端性能收益约3.20%，在BLOOM 176B模型中获得端到端性能收益约5.47%，在LLAMA2 70B模型中获得端到端性能收益约7.85%。精度相对误差控制在2%的范围内。

## 注意事项：

1. 计算通信并行优化特性不能与TP重计算通信优化特性同时使用，因为存在机制上的冲突；
