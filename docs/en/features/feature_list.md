# Feature List

This document describes the features related to MindSpeed Core.

**Table 1** Feature list

<table>
  <thead>
    <tr>
      <th>Feature Type</th>
      <th>Feature Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">Megatron Features</td>
      <td><a href="../features/data-parallel.md">Megatron Data Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/tensor-parallel.md">Megatron Tensor Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/pipeline-parallel.md">Megatron Pipeline Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/virtual-pipeline-parallel.md">Megatron Virtual Pipeline Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/distributed-optimizer.md">Megatron Distributed Optimizer</a></td>
    </tr>
    <tr>
      <td><a href="../features/sequence-parallel.md">Megatron Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/async-ddp.md">Megatron Asynchronous DDP</a></td>
    </tr>
    <tr>
      <td><a href="../features/async-ddp-param-gather.md">Megatron Weight Update Communication Overlap</a></td>
    </tr>
    <tr>
      <td><a href="../features/recomputation.md">Megatron Recomputation</a></td>
    </tr>
    <tr>
      <td><a href="../features/custom_fsdp.md">Megatron Fully Sharded Data Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/transformer_engine.md">Megatron Transformer Engine</a></td>
    </tr>
    <tr>
      <td><a href="../features/multi-head-latent-attention.md">Megatron Multi-head Latent Attention</a></td>
    </tr>
    <tr>
      <td rowspan="6">Parallel Strategies</td>
      <td><a href="../features/ulysses-context-parallel.md">Ascend Ulysses Long-Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/ring-attention-context-parallel.md">Ascend Ring Attention Long-Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/noop-layers.md">Ascend Custom No-Op Layer</a></td>
    </tr>
    <tr>
      <td><a href="../features/dualpipev.md">Ascend DualPipeV</a></td>
    </tr>
    <tr>
      <td><a href="../features/kvallgather-context-parallel.md">KVAllGather Long-Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/pipeline-model-parallel-layout.md">Megatron Custom Pipeline Layout</a></td>
    </tr>
    <tr>
      <td rowspan="10">Memory Optimization</td>
      <td><a href="../features/activation-function-recompute.md">Ascend Activation Function Recomputation</a></td>
    </tr>
    <tr>
      <td><a href="../features/generate-mask.md">Ascend Mask Generation</a></td>
    </tr>
    <tr>
      <td><a href="../features/reuse-fp32-param.md">Ascend BF16 Parameter Replica Reuse</a></td>
    </tr>
    <tr>
      <td><a href="../features/swap_attention.md">Ascend swap_attention</a></td>
    </tr>
    <tr>
      <td><a href="../features/norm-recompute.md">Ascend Norm Recomputation</a></td>
    </tr>
    <tr>
      <td><a href="../features/hccl-group-buffer-set.md">Ascend HCCL Group Buffer Set</a></td>
    </tr>
    <tr>
      <td><a href="../features/swap-optimizer.md">Ascend Swap Optimizer</a></td>
    </tr>
    <tr>
      <td><a href="../features/virtual-optimizer.md">Virtual Optimizer</a></td>
    </tr>
    <tr>
      <td><a href="../features/param-and-grad-buffer-pad.md">Parameter and Gradient Buffer Alignment</a></td>
    </tr>
    <tr>
      <td><a href="../features/swap-layer-input.md">Transformer Layer Input Offloading</a></td>
    </tr>
    <tr>
      <td rowspan="10">Affinity Computation</td>
      <td><a href="../features/rms_norm.md">Ascend rms_norm Fusion Operator</a></td>
    </tr>
    <tr>
      <td><a href="../features/swiglu.md">Ascend swiglu Fusion Operator</a></td>
    </tr>
    <tr>
      <td><a href="../features/rotary-embedding.md">Ascend rotary_embedding Fusion Operator</a></td>
    </tr>
    <tr>
      <td><a href="../features/flash-attention.md">Ascend Flash Attention</a></td>
    </tr>
    <tr>
      <td><a href="../features/moe-token-permute-and-unpermute.md">Ascend Moe Token Permute and Unpermute Fusion Operator</a></td>
    </tr>
    <tr>
      <td><a href="../features/npu_matmul_add.md">Ascend npu_matmul_add_fp32 Gradient Accumulation Fusion Operator</a></td>
    </tr>
    <tr>
      <td><a href="../features/communication-over-computation.md">Ascend Communication Over Computation</a></td>
    </tr>
    <tr>
      <td><a href="../features/mc2.md">Ascend MC2</a></td>
    </tr>
    <tr>
      <td><a href="../features/hif8_dts.md">HiF8 DTS (Delayed Tensor Scaling) Training</a></td>
    </tr>
    <tr>
      <td><a href="../features/te-comparsion-online.md">Low-Precision Online Comparison</a></td>
    </tr>
    <tr>
      <td rowspan="2">Communication Optimization</td>
      <td><a href="../features/hccl-replace-gloo.md">Ascend Gloo Snapshot Persistence Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/async-log-allreduce.md">Async Log Allreduce</a></td>
    </tr>
    <tr>
      <td rowspan="9">Mcore MoE Features</td>
      <td><a href="../features/megatron_moe/megatron-moe-gmm.md">Ascend Megatron MoE GMM</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-allgather-dispatcher.md">Ascend Megatron MoE Allgather Dispatcher Performance Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-alltoall-dispatcher.md">Ascend Megatron MoE Alltoall Dispatcher Performance Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-alltoall-overlap-comm.md">Communication Overlap for Megatron MoE AlltoAll Dispatcher</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-allgather-overlap-comm.md">Communication Overlap for Megatron MoE AlltoAll Dispatcher</a></td>
    </tr>
    <tr>
      <td><a href="../features/shared-experts.md">Ascend Shared Experts</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-fb-overlap.md">1F1B Overlap</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-zero-memory.md">Megatron MoE Alltoall Dispatcher Memory Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-fix-router.md">MoE Fixed Routing</a></td>
    </tr>
    <tr>
      <td rowspan="4">Key Scenario</td>
      <td><a href="../features/eod-reset.md">Ascend EOD Reset Training</a></td>
    </tr>
    <tr>
      <td><a href="../features/deepseek-sparse-attention.md">DeepSeek Sparse Attention (DSA)</a></td>
    </tr>
    <tr>
      <td><a href="../features/gated-delta-net.md">GatedDeltaNet (GDN)</a></td>
    </tr>
    <tr>
      <td><a href="../features/preprocess-eod-data.md">Pack Mode Pre-training Dataset Processing</a></td>
    </tr>
    <tr>
      <td rowspan="7">Others</td>
      <td><a href="../features/ops_flops_cal.md">Ascend TFLOPS Calculation</a></td>
    </tr>
    <tr>
      <td><a href="../features/npu_deterministic.md">Ascend Deterministic Computing</a></td>
    </tr>
    <tr>
      <td><a href="../features/npu_datadump.md">Ascend MindStudio Training Tools for Precision Comparison</a></td>
    </tr>
    <tr>
      <td><a href="../features/muon-optimizer.md">Muon Optimizer</a></td>
    </tr>
    <tr>
      <td><a href="../features/aiqos.md">AI QoS Differentiated Scheduling</a></td>
    </tr>
    <tr>
      <td><a href="../features/cache-get-batch-config.md">get_batch Config Cache</a></td>
    </tr>
    <tr>
      <td><a href="../features/pretrained-from-hf-tokenizer.md">PretrainedFromHF Tokenizer</a></td>
    </tr>
  </tbody>
</table>
