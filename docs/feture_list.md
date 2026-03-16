# 特性列表

 本手册描述MindSpeed Core相关特性
 
**表 1**  特性列表
 
|特性类型|特性名称|Released|PyTorch框架支持情况|MindSpore框架支持情况|
|--|--|-|-|--|
|Megatron特性|[Megatron 数据并行](features/data-parallel.md)|✅|✅|-|
|Megatron特性|[Megatron 张量并行](features/tensor-parallel.md)|✅|✅|✅|
|Megatron特性|[Megatron 流水并行](features/pipeline-parallel.md)|✅|✅|✅|
|Megatron特性|[Megatron 虚拟流水线并行](features/virtual-pipeline-parallel.md)|✅|✅|✅|
|Megatron特性|[Megatron 分布式优化器](features/distributed-optimizer.md)|✅|✅|✅|
|Megatron特性|[Megatron 序列并行](features/sequence-parallel.md)|✅|✅|✅|
|Megatron特性|[Megatron 异步DDP](features/async-ddp.md)|✅|✅|❌|
|Megatron特性|[Megatron 权重更新通信隐藏](features/async-ddp-param-gather.md)|✅|✅|✅|
|Megatron特性|[Megatron 重计算](features/recomputation.md)|✅|✅|✅|
|Megatron特性|[Megatron 分布式权重](features/dist_ckpt.md)|✅|✅|-|
|Megatron特性|[Megatron 全分片并行](features/custom_fsdp.md)|✅|✅|暂不支持开启pp及--reuse-fp32-param参数配置|
|Megatron特性|[Megatron Transformer Engine](features/transformer_engine.md)|✅|✅|❌|
|Megatron特性|[Megatron Multi-head Latent Attention](features/multi-head-latent-attention.md)|✅|✅|❌|
|并行策略特性|[Ascend Ulysses 长序列并行](features/ulysses-context-parallel.md)|✅|✅|✅|
|并行策略特性|[Ascend Ring Attention 长序列并行](features/ring-attention-context-parallel.md)|✅|✅|✅|
|并行策略特性|[Ascend Double Ring Attention 长序列并行](features/double-ring.md)|✅|✅|❌|
|并行策略特性|[Ascend 混合长序列并行](features/hybrid-context-parallel.md)|✅|✅|❌|
|并行策略特性|[Ascend 自定义空操作层](features/noop-layers.md)|✅|✅|✅|
|并行策略特性|[Ascend DualPipeV](features/dualpipev.md)|✅|✅|暂不支持--dualpipev-dw-detach参数配置|
|内存优化特性|[Ascend 激活函数重计算](features/activation-function-recompute.md)|✅|✅|✅|
|内存优化特性|[Ascend 重计算流水线独立调度](features/recompute_independent_pipelining.md)|✅|✅|❌|
|内存优化特性|[Ascend Mask归一](features/generate-mask.md)|✅|✅|❌|
|内存优化特性|[Ascend BF16 参数副本复用](features/reuse-fp32-param.md)|✅|✅|✅|
|内存优化特性|[Ascend swap_attention](features/swap_attention.md)|✅|✅|❌|
|内存优化特性|[Ascend Norm重计算](features/norm-recompute.md)|✅|✅|✅|
|内存优化特性|[Ascend Hccl Buffer 自适应](features/hccl-group-buffer-set.md)|✅|✅|❌|
|内存优化特性|[Ascend Swap Optimizer](features/swap-optimizer.md)|✅|✅|✅|
|内存优化特性|[Virtual Optimizer](features/virtual-optimizer.md)|✅|✅|❌|
|亲和计算特性|[Ascend rms_norm 融合算子](features/rms_norm.md)|✅|✅|✅|
|亲和计算特性|[Ascend swiglu 融合算子](features/swiglu.md)|✅|✅|✅|
|亲和计算特性|[Ascend rotary_embedding 融合算子](features/rotary-embedding.md)|✅|✅|❌|
|亲和计算特性|[Ascend flash attention](features/flash-attention.md)|✅|✅|✅|
|亲和计算特性|[Ascend Moe Token Permute and Unpermute 融合算子](features/moe-token-permute-and-unpermute.md)|✅|✅|✅|
|亲和计算特性|[Ascend npu_matmul_add_fp32 梯度累加融合算子](features/npu_matmul_add.md)|✅|✅|✅|
|亲和计算特性|[Ascend 计算通信并行优化](features/communication-over-computation.md)|❌|✅|✅|
|亲和计算特性|[Ascend MC2](features/mc2.md)|❌|✅|❌|
|亲和计算特性|[Ascend fusion_attention_v2](features/fusion-attn-v2.md)|❌|✅|❌|
|通信优化特性|[Ascend Gloo 存档落盘优化](features/hccl-replace-gloo.md)|✅|✅|❌|
|通信优化特性|[Ascend 高维张量并行](features/tensor-parallel-2d.md)|✅|✅|❌|
|Mcore MoE特性|[Ascend Megatron MoE GMM](features/megatron_moe/megatron-moe-gmm.md)|✅|✅|✅|
|Mcore MoE特性|[Ascend Megatron MoE Allgather Dispatcher 性能优化](features/megatron_moe/megatron-moe-allgather-dispatcher.md)|✅|✅|❌|
|Mcore MoE特性|[Ascend Megatron MoE Alltoall Dispatcher 性能优化](features/megatron_moe/megatron-moe-alltoall-dispatcher.md)|✅|✅|✅|
|Mcore MoE特性|[Ascend Megatron MoE TP拓展EP](features/megatron_moe/megatron-moe-tp-extend-ep.md)|✅|✅|✅|
|Mcore MoE特性|[Megatron MoE alltoall dispatcher分支通信隐藏优化](features/megatron_moe/megatron-moe-alltoall-overlap-comm.md)|❌|✅|✅|
|Mcore MoE特性|[Megatron MoE allgather dispatcher分支通信隐藏优化](features/megatron_moe/megatron-moe-allgather-overlap-comm.md)|✅|✅|❌|
|Mcore MoE特性|[Ascend 共享专家](features/shared-experts.md)|✅|✅|✅|
|Mcore MoE特性|[1F1B Overlap](features/megatron_moe/megatron-moe-fb-overlap.md)|✅|✅|❌|
|Mcore MoE特性|[专家并行动态负载均衡(数参互寻)](features/balanced_moe.md)|✅|✅|❌|
|关键场景特性|[Ascend EOD Reset训练场景](features/eod-reset.md)|✅|✅|✅|
|关键场景特性|[Ascend alibi](features/alibi.md)|❌|✅|✅|
|多模态特性|[Ascend fused ema adamw优化器](features/fused_ema_adamw_optimizer.md)|❌|✅|❌|
|多模态特性|[Ascend PP支持动态形状](features/variable_seq_lengths.md)|✅|✅|✅|
|多模态特性|[Ascend PP支持多参数传递](features/multi_parameter_pipeline.md)|✅|✅|❌|
|多模态特性|[Ascend PP支持多参数传递和动态形状](features/multi_parameter_pipeline_and_variable_seq_lengths.md)|✅|✅|❌|
|多模态特性|[Ascend 非对齐线性层](features/unaligned_linear.md)|✅|✅|❌|
|多模态特性|[Ascend 非对齐Ulysses长序列并行](features/unaligned-ulysses-context-parallel.md)|✅|✅|❌|
|其它特性|[Ascend TFLOPS计算](features/ops_flops_cal.md)|✅|✅|✅|
|其它特性|[Ascend Auto Settings 并行策略自动搜索系统](features/auto_settings.md)|❌|✅|❌|
|其它特性|[Ascend 确定性计算](features/npu_deterministic.md)|❌|✅|✅|
|其它特性|[Ascend MindStudio Training Tools 精度对照](features/npu_datadump.md)|❌|✅|❌|

> [!NOTE]  
> 上表中的“Released”代表商用版本已发布，“✅”代表支持，“❌”代表不支持
