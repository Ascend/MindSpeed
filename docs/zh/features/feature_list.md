# 特性总览

本手册汇总了MindSpeed Core相关特性，具体分类如下所示。

**表 1**  特性列表

<table>
  <thead>
    <tr>
      <th>特性类型</th>
      <th>特性名称</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="13">Megatron特性</td>
      <td><a href="../features/data-parallel.md">Megatron数据并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/tensor-parallel.md">Megatron张量并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/pipeline-parallel.md">Megatron流水并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/virtual-pipeline-parallel.md">Megatron虚拟流水线并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/distributed-optimizer.md">Megatron分布式优化器</a></td>
    </tr>
    <tr>
      <td><a href="../features/sequence-parallel.md">Megatron序列并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/async-ddp.md">Megatron异步DDP</a></td>
    </tr>
    <tr>
      <td><a href="../features/async-ddp-param-gather.md">Megatron权重更新通信隐藏</a></td>
    </tr>
    <tr>
      <td><a href="../features/recomputation.md">Megatron重计算</a></td>
    </tr>
    <tr>
      <td><a href="../features/dist_ckpt.md">Megatron分布式权重</a></td>
    </tr>
    <tr>
      <td><a href="../features/custom_fsdp.md">Megatron全分片并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/transformer_engine.md">Megatron Transformer Engine</a></td>
    </tr>
    <tr>
      <td><a href="../features/multi-head-latent-attention.md">Megatron Multi-head Latent Attention</a></td>
    </tr>
    <tr>
      <td rowspan="6">并行策略特性</td>
      <td><a href="../features/ulysses-context-parallel.md">Ascend Ulysses长序列并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/ring-attention-context-parallel.md">Ascend Ring Attention长序列并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/double-ring.md">Ascend Double Ring Attention长序列并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/hybrid-context-parallel.md">Ascend混合长序列并行</a></td>
    </tr>
    <tr>
      <td><a href="../features/noop-layers.md">Ascend自定义空操作层</a></td>
    </tr>
    <tr>
      <td><a href="../features/dualpipev.md">Ascend DualPipeV</a></td>
    </tr>
    <tr>
      <td rowspan="9">内存优化特性</td>
      <td><a href="../features/activation-function-recompute.md">Ascend激活函数重计算</a></td>
    </tr>
    <tr>
      <td><a href="../features/recompute_independent_pipelining.md">Ascend重计算流水线独立调度</a></td>
    </tr>
    <tr>
      <td><a href="../features/generate-mask.md">Ascend Mask归一</a></td>
    </tr>
    <tr>
      <td><a href="../features/reuse-fp32-param.md">Ascend BF16参数副本复用</a></td>
    </tr>
    <tr>
      <td><a href="../features/swap_attention.md">Ascend swap_attention</a></td>
    </tr>
    <tr>
      <td><a href="../features/norm-recompute.md">Ascend Norm重计算</a></td>
    </tr>
    <tr>
      <td><a href="../features/hccl-group-buffer-set.md">Ascend Hccl Buffer自适应</a></td>
    </tr>
    <tr>
      <td><a href="../features/swap-optimizer.md">Ascend Swap Optimizer</a></td>
    </tr>
    <tr>
      <td><a href="../features/virtual-optimizer.md">Virtual Optimizer</a></td>
    </tr>
    <tr>
      <td>亲和计算特性</td>
      <td><a href="../features/flash-attention.md">Ascend Flash Attention</a></td>
    </tr>
    <tr>
      <td rowspan="2">通信优化特性</td>
      <td><a href="../features/hccl-replace-gloo.md">Ascend Gloo存档落盘优化</a></td>
    </tr>
    <tr>
      <td><a href="../features/tensor-parallel-2d.md">Ascend高维张量并行</a></td>
    </tr>
    <tr>
      <td rowspan="8">Mcore MoE特性</td>
      <td><a href="../features/megatron_moe/megatron-moe-gmm.md">Ascend Megatron MoE GMM</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-allgather-dispatcher.md">Ascend Megatron MoE Allgather Dispatcher性能优化</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-alltoall-dispatcher.md">Ascend Megatron MoE Alltoall Dispatcher性能优化</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-tp-extend-ep.md">Ascend Megatron MoE TP拓展EP</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-allgather-overlap-comm.md">Megatron MoE Allgather Dispatcher分支通信隐藏优化</a></td>
    </tr>
    <tr>
      <td><a href="../features/shared-experts.md">Ascend共享专家</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-fb-overlap.md">1F1B Overlap</a></td>
    </tr>
    <tr>
      <td><a href="../features/balanced_moe.md">专家并行动态负载均衡（数参互寻）</a></td>
    </tr>
    <tr>
      <td>关键场景特性</td>
      <td><a href="../features/eod-reset.md">Ascend EOD Reset训练场景</a></td>
    </tr>
    <tr>
      <td rowspan="5">多模态特性</td>
      <td><a href="../features/variable_seq_lengths.md">Ascend PP支持动态形状</a></td>
    </tr>
    <tr>
      <td><a href="../features/multi_parameter_pipeline.md">Ascend PP支持多参数传递</a></td>
    </tr>
    <tr>
      <td><a href="../features/multi_parameter_pipeline_and_variable_seq_lengths.md">Ascend PP支持多参数传递和动态形状</a></td>
    </tr>
    <tr>
      <td><a href="../features/unaligned_linear.md">Ascend非对齐线性层</a></td>
    </tr>
    <tr>
      <td><a href="../features/unaligned-ulysses-context-parallel.md">Ascend非对齐Ulysses长序列并行</a></td>
    </tr>
    <tr>
      <td>其它特性</td>
      <td><a href="../features/ops_flops_cal.md">Ascend TFLOPS计算</a></td>
    </tr>
  </tbody>
</table>
