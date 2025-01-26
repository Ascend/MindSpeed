# 简介

MindSpeed 是针对华为昇腾设备的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后再通过集合通信对结果进行汇总。

昇腾提供 MindSpeed 加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

# 安装

### 1. 安装依赖

在安装**MindSpeed**之前，请参考[版本配套表](#版本配套表)，安装最新昇腾软件栈。

<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>软件安装指南</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="3">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>




### 2. 安装 MindSpeed

下载源码安装：

 ```shell
 git clone -b core_r0.7.0 https://gitee.com/ascend/MindSpeed.git
 pip install -e MindSpeed
 ```

如需使用Ascend Transformer Boost（ATB）加速库算子，请先安装CANN-NNAL并初始化添加环境，例如：
 ```shell
#CANN-NNAL默认安装路径为：/usr/local/Ascend/nnal
#运行CANN-NNAL默认安装路径下atb文件夹中的环境配置脚本set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 
 ```

### 3. 获取 Megatron-LM 并指定分支

 ```shell
 # 目前版本基于core_r0.7.0的release版本
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_r0.7.0
 ```

# 快速上手

以 GPT 模型为例：

1. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行`import mindspeed.megatron_adaptor`

    ```diff
     import os
     import torch
    +import mindspeed.megatron_adaptor
     from functools import partial
     from typing import Union
    ```

2. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在model_provider函数中删除`assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"`。
    ```diff
    else:
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    ```

3. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。
    ```shell
    bash examples/pretrain_gpt_distributed.sh
    ```

# 自定义优化级别
MindSpeed提供了多层次的优化解决方案，分为三个层级，用户可根据实际需求灵活启用任意层级。高层级兼容低层级的能力，确保了整个系统的稳定性和扩展性。
用户可以通过设置`--optimization-level {优化层级}`参数来自定义开启的优化层级。该参数支持以下值：
- `0`：基础兼容层L0，提供Megatron-LM框架对NPU的支持，确保无缝集成。该层包含基础功能集patch，保证可靠性和稳定性，为高级优化奠定基础。
- `1`：亲和性增强层L1（兼容L0能力），集成高性能融合算子库，结合昇腾亲和的计算优化，充分释放昇腾算力，显著提升计算效率。
- `2`（默认值）：自研加速算法层L2（兼容L1,L0能力），集成了多项自主研发的核心技术成果，提供全面的性能优化。


# 特性介绍
MindSpeed特性由七大模块组成，分别为：megetron特性支持、并行策略特性、内存优化特性、亲和计算特性、通信优化特性、关键场景特性以及多模态特性。其中【Prototype】表示原型特性，暂未商用发布。

当前大模型训练主要使用bf16数据类型，以下特性若无特殊声明原则上兼容fp16，如遇到fp16场景下相关问题请联系MindSpeed团队或提交issue,我们会快速响应。
## Megatron特性支持
| 特性                           | 介绍                                                        |
|------------------------------|-----------------------------------------------------------|
| Megatron 数据并行                | [link](docs/features/data-parallel.md)                    |
| Megatron 张量并行                | [link](docs/features/tensor-parallel.md)                  |
| Megatron 流水并行                | [link](docs/features/pipeline-parallel.md)                |
| Megatron 虚拟流水并行              | [link](docs/features/virtual-pipeline-parallel.md)        |
| Megatron 分布式优化器              | [link](docs/features/distributed-optimizer.md)            |
| Megatron 序列并行                | [link](docs/features/sequence-parallel.md)                |
| Megatron 异步DDP               | [link](docs/features/async-ddp.md)                        |
| Megatron 权重更新通信隐藏            | [link](docs/features/async-ddp-param-gather.md)           |
| Megatron 重计算                 | [link](docs/features/recomputation.md)                    |

## 并行策略特性
| 特性                           | 介绍                                                        |
|------------------------------|-----------------------------------------------------------|
| Ascend Ulysses 长序列并行                | [link](docs/features/ulysses-context-parallel.md)         |
| Ascend Ring Attention 长序列并行         | [link](docs/features/ring-attention-context-parallel.md)  |
| Ascend 混合长序列并行           | [link](docs/features/hybrid-context-parallel.md)          |
| Ascend 自定义空操作层                              | [link](docs/features/noop-layers.md)     |

## 内存优化特性
| 特性                           | 介绍                                                        |
|------------------------------|-----------------------------------------------------------|
| Ascend 自适应选择重计算              | [link](docs/features/adaptive-recompute.md)               |
| Ascend 激活函数重计算               | [link](docs/features/activation-function-recompute.md)    |
| Ascend 重计算流水线独立调度            | [link](docs/features/recompute_independent_pipelining.md) |
| Ascend Mask归一                | [link](docs/features/generate-mask.md)                    |
| Ascend BF16 参数副本复用           | [link](docs/features/reuse-fp32-param.md)                 |
| Ascend swap_attention            | [link](docs/features/swap_attention.md)                   |
| Ascend Norm重计算                  | [link](docs/features/norm-recompute.md)          |

## 亲和计算特性
| 特性                                         | 介绍                                                           |
|--------------------------------------------|--------------------------------------------------------------|
| Ascend rms_norm 融合算子                       | [link](docs/features/rms_norm.md)                            |
| Ascend swiglu 融合算子                         | [link](docs/features/swiglu.md)                              |
| Ascend rotary_embedding 融合算子               | [link](docs/features/rotary-embedding.md)                    |
| Ascend flash attention 融合算子                | [link](docs/features/flash-attention.md)                     |
| Ascend Moe Token Permute and Unpermute 融合算子 | [link](docs/features/moe-token-permute-and-unpermute.md)     |
| Ascend npu_matmul_add_fp32梯度累加融合算子         | [link](docs/features/npu_matmul_add.md)                      |
| 【Prototype】Ascend Moe BMM通算融合算子                       | [link](docs/features/megatron_moe/megatron-moe-bmm-fused.md) |
| 【Prototype】Ascend 计算通信并行优化                 | [link](docs/features/communication-over-computation.md)      |
| 【Prototype】Ascend MC2                      | [link](docs/features/mc2.md)                                 |
| 【Prototype】Ascend fusion_attention_v2      | [link](docs/features/fusion-attn-v2.md)                      |

## 通信优化特性
| 特性                           | 介绍                                                        |
|------------------------------|-----------------------------------------------------------|
| Ascend nano-pipe流水线并行        | [link](docs/features/nanopipe-pipeline-parallel.md)       |
| 【Prototype】Ascend 高维张量并行                | [link](docs/features/tensor-parallel-2d.md)                           |
| Ascend Gloo 存档落盘优化                | [link](docs/features/hccl-replace-gloo.md)                           |

## Mcore MoE特性
| 特性                                                       | 介绍                                                                               |
|----------------------------------------------------------|----------------------------------------------------------------------------------|
| Ascend Megatron MoE GMM                                         | [link](docs/features/megatron_moe/megatron-moe-gmm.md)                                                      |
| Ascend Megatron MoE Allgather Dispatcher 性能优化 | [link](docs/features/megatron_moe/megatron-moe-allgather-dispatcher.md)                                     |
| Ascend Megatron MoE Alltoall Dispatcher 性能优化  | [link](docs/features/megatron_moe/megatron-moe-alltoall-dispatcher.md)                                      |
| Ascend Megatron MoE TP拓展EP | [link](docs/features/megatron_moe/megatron-moe-tp-extend-ep.md)                                             |
| Ascend 共享专家                     | [link](docs/features/shared-experts.md)                                          |
| 【Prototype】Ascend Megatron MoE 负载感知内存均衡算法                           | [link](docs/features/megatron_moe/megatron-moe-adaptive-recompute-activation.md) |

## DeepSpeed MoE特性
| 特性                                 | 介绍                                                   |
|------------------------------------|------------------------------------------------------|
| DeepSpeed MoE                   | [link](docs/features/deepspeed_moe/deepspeed-moe.md) |
| Ascend DeepSpeed MoE token 重排性能优化  | [link](docs/features/deepspeed_moe/deepspeed-moe-token-rearrange.md)             |
| Ascend DeepSpeed MoE dropless 性能优化 | [link](docs/features/deepspeed_moe/deepspeed-moe-efficient-moe.md) |
| Ascend MLP 通信隐藏              | [link](docs/features/pipeline-experts.md)                 |
| Ascend Ampipe流水通信隐藏          | [link](docs/features/ampipe.md)                     |

## 关键场景特性
| 特性                              | 介绍                                                   |
|---------------------------------|------------------------------------------------------|
| Ascend EOD Reset训练场景 | [link](docs/features/eod-reset.md)                   |
| 【Prototype】Ascend alibi         | [link](docs/features/alibi.md)                       |

## 多模态特性
| 特性                                   | 介绍                                                 |
|--------------------------------------|----------------------------------------------------|
| Ascend PP支持动态形状                            | [link](docs/features/variable_seq_lengths.md)     |
| Ascend PP支持多参数传递                           | [link](docs/features/multi_parameter_pipeline.md)     |
| 【Prototype】Ascend fused ema adamw优化器 | [link](docs/features/fused_ema_adamw_optimizer.md) |
| 【Prototype】Ascend conv3d 序列并行        | [link](docs/features/conv3d_sequence_paralle.md) |

## 其它特性
| 特性                                         | 介绍                                                        |
|--------------------------------------------|-----------------------------------------------------------|
| Ascend TFLOPS计算                            | [link](docs/features/ops_flops_cal.md)                    |
| Ascend Auto Tuning 并行策略自动搜索系统              | [link](docs/features/auto_tuning.md)                     |
| 【Prototype】Ascend 确定性计算                    | [link](docs/features/npu_deterministic.md)                    |
| 【Prototype】Ascend Flex-Parallel 多维并行自动调优算法 | [link](docs/features/Automatic_Parallelism.md)  |

# 自定义算子
部分自定义算子设置为公开接口，公开接口设置说明请参照MindSpeed安全声明中的[公开接口声明](SECURITYNOTE.md#公开接口声明)，具体对外接口细节参照以下算子对应的手册链接。
| 算子                                         | 介绍                                                  |
|--------------------------------------------|-----------------------------------------------------|
| npu_dropout_add_layer_norm                 | [link](docs/ops/npu_dropout_add_layer_norm.md)      |
| npu_rotary_position_embedding              | [link](docs/ops/npu_rotary_position_embedding.md)   |
| fusion_attention                | [link](docs/ops/fusion_attention.md)                |
| rms_norm                        | [link](docs/ops/rms_norm.md)                        |
| swiglu                          | [link](docs/ops/swiglu.md)                          |
| npu_mm_all_reduce_add_rms_norm  | [link](docs/ops/npu_mm_all_reduce_add_rms_norm.md)  |
| npu_mm_all_reduce_add_rms_norm_ | [link](docs/ops/npu_mm_all_reduce_add_rms_norm_.md) |
| npu_gmm             | [link](docs/ops/gmm.md)  |
| npu_grouped_mat_mul_all_reduce  | [link](docs/ops/npu_grouped_mat_mul_all_reduce.md)  |
| 【Prototype】lcal_coc                        | [link](docs/ops/lcal_coc.md)                        |
| 【Prototype】ffn                             | [link](docs/ops/ffn.md)                             |
| 【Prototype】npu_fused_moe_token_permute     | [link](docs/ops/npu_fused_moe_token_permute.md)     |
| 【Prototype】npu_fused_moe_token_unpermute   | [link](docs/ops/npu_fused_moe_token_unpermute.md)    |
| 【Prototype】npu_ring_attention_update  | [link](docs/ops/npu_ring_attention_update.md)  |
| 【Prototype】npu_matmul_add_fp32 | [link](docs/ops/npu_matmul_add.md)  |
| 【Prototype】npu_groupmatmul_add_fp32 | [link](docs/ops/npu_groupmatmul_add.md)  |
| 【Prototype】npu_all_to_all_all_gather_bmm  | [link](docs/ops/npu_all_to_all_all_gather_bmm.md)  |
| 【Prototype】npu_bmm_reduce_scatter_all_to_all  | [link](docs/ops/npu_bmm_reduce_scatter_all_to_all.md)  |
| 【Prototype】quant_gmm  | [link](docs/ops/quant_gmm.md)  |
| 【Prototype】npu_apply_fused_ema_adamw  | [link](docs/ops/npu_apply_fused_ema_adamw.md)  |

# MindSpeed中采集Profile数据

MindSpeed支持命令式开启Profile采集数据，命令配置介绍如下：

| 配置命令                    | 命令含义                                                                              | 
|-------------------------|-----------------------------------------------------------------------------------|
| --profile               | 打开profile开关                                                                       |
| --profile-step-start    | 配置开始采集步, 未配置时默认为10, 配置举例: --profile-step-start 30                                 |
| --profile-step-end      | 配置结束采集步, 未配置时默认为12, 配置举例: --profile-step-end 35                                   |
| --profile-level         | 配置采集等级, 未配置时默认为level0, 可选配置: level0, level1, level2, 配置举例: --profile-level level1 |
| --profile-with-cpu      | 打开cpu信息采集开关                                                                       |
| --profile-with-stack    | 打开stack信息采集开关                                                                     |
| --profile-with-memory   | 打开memory信息采集开关, 配置本开关时需打开--profile-with-cpu                                       |
| --profile-record-shapes | 打开shapes信息采集开关                                                                    |
| --profile-save-path     | 配置采集信息保存路径, 未配置时默认为./profile_dir, 配置举例: --profile-save-path ./result_dir          |
| --profile-ranks         | 配置待采集的ranks，未配置时默认为-1，表示采集所有rank的profiling数据，配置举例: --profile-ranks 0 1 2 3, 需注意: 该配置值为每个rank在单机/集群中的全局值   |

# 版本配套表

**PyTorch Extension**版本号采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**PyTorch Extension**匹配的PyTorch版本，后者用于匹配CANN版本，详细匹配如下：

| MindSpeed版本             | Megatron版本      | PyTorch版本   | torch_npu版本 | CANN版本  | Python版本                               | 硬件型态     |
|-------------------------|-----------------|------------- |-------------|---------|----------------------------------------|----------|
| master（主线）              | Core 0.7.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| core_r0.7.0（主线）         | Core 0.7.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| core_r0.6.0（主线）         | Core 0.6.0      |   2.1.0     | 在研版本        | 在研版本    | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.0_core_r0.7.0（商用）   | Core 0.7.0      |  2.1.0     | 6.0.0       | 8.0.0   | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.0_core_r0.6.0（商用）   | Core 0.6.0      |  2.1.0     | 6.0.0       | 8.0.0   | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC3_core_r0.7.0（商用） | Core 0.7.0      |  2.1.0     | 6.0.RC3     | 8.0.RC3 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC3_core_r0.6.0（商用） | Core 0.6.0      |  2.1.0     | 6.0.RC3     | 8.0.RC3 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC2（商用）             | Core 0.6.0      |  2.1.0     | 6.0.RC2     | 8.0.RC2 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |
| 1.0.RC1（商用）             | commitid bcce6f |  2.1.0     | 6.0.RC1     | 8.0.RC1 | Python3.8.x, Python3.9.x, Python3.10.x | Atlas 200T A2 Box16,  Atlas 800T A2,  Atlas 900 A2 PODc |

[昇腾辅助软件](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中有更多关于PyTorch和CANN的版本信息。

# 分支维护策略

MindSpeed版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |

# MindSpeed版本维护策略

| **MindSpeed版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**           | **EOL日期** |
|---------------------|-----------|---------|------------|--------------------|-----------|
| 1.0.0_core_r0.7.0   |  常规版本  | 开发      | 2024/12/30 | 预计2025/6/30起无维护	   |           |
| 1.0.0_core_r0.6.0   |  常规版本  | 开发      | 2024/12/30 | 预计2025/6/30起无维护	   |           |
| 1.0.RC3_core_r0.7.0 |  常规版本  | 维护      | 2024/09/30 | 预计2025/3/30起无维护	   |           |
| 1.0.RC3_core_r0.6.0 |  常规版本  | 维护      | 2024/09/30 | 预计2025/3/30起无维护	   |           |
| 1.0.RC2             |  常规版本  | 维护      | 2024/06/30 | 预计2024/12/30起无维护	   |           |
| 1.0.RC1             |  常规版本  | 停止维护  | 2024/03/30 | 2024/9/30起无维护           |           |


# 安全声明

[MindSpeed 安全声明](SECURITYNOTE.md)

# 常见问题

| 现象                                 | 介绍                                    |
|------------------------------------|---------------------------------------|
| Data helpers 数据预处理出错               | [link](docs/faq/data_helpers.md)      |
| Torch extensions 编译卡住              | [link](docs/faq/torch_extensions.md)  |
| megatron0.7.0版本长稳测试出现grad norm为nan | [link](docs/faq/megatron070_grad_norm_nan.md)  |

# 致谢

MindSpeed-Core 由华为公司的下列部门联合贡献 ：

华为公司：

- 昇腾计算产品部
- 计算算法部
- 计算软件平台部 
- 计算技术开发部
- 公共开发部：NAIE
- 网络技术实验室

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-Core！
