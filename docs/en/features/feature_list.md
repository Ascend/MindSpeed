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
      <td rowspan="13">Megatron Features</td>
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
      <td><a href="../features/dist_ckpt.md">Megatron Distributed Weight</a></td>
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
      <td><a href="../features/double-ring.md">Ascend Double Ring Attention Long-Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/hybrid-context-parallel.md">Ascend Hybrid Long-Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td><a href="../features/noop-layers.md">Ascend Custom No-Op Layer</a></td>
    </tr>
    <tr>
      <td><a href="../features/dualpipev.md">Ascend DualPipeV</a></td>
    </tr>
    <tr>
      <td rowspan="9">Memory Optimization</td>
      <td><a href="../features/activation-function-recompute.md">Ascend Activation Function Recomputation</a></td>
    </tr>
    <tr>
      <td><a href="../features/recompute_independent_pipelining.md">Ascend Independent Pipelining of Recomputation</a></td>
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
      <td>Affinity Computation</td>
      <td><a href="../features/flash-attention.md">Ascend Flash Attention</a></td>
    </tr>
    <tr>
      <td rowspan="2">Communication Optimization</td>
      <td><a href="../features/hccl-replace-gloo.md">Ascend Gloo Snapshot Persistence Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/tensor-parallel-2d.md">Ascend High-Dimensional Tensor Parallelism</a></td>
    </tr>
    <tr>
      <td rowspan="8">Mcore MoE Features</td>
      <td><a href="../features/megatron_moe/megatron-moe-gmm.md">Ascend Megatron MoE GMM</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-allgather-dispatcher.md">Ascend Megatron MoE Allgather Dispatcher Performance Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-alltoall-dispatcher.md">Ascend Megatron MoE Alltoall Dispatcher Performance Optimization</a></td>
    </tr>
    <tr>
      <td><a href="../features/megatron_moe/megatron-moe-tp-extend-ep.md">Ascend Megatron MoE TP Extended EP</a></td>
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
      <td><a href="../features/balanced_moe.md">Dynamically Balanced Expert Parallelism (Data-Parameter Mutual Search)</a></td>
    </tr>
    <tr>
      <td>Key Scenario</td>
      <td><a href="../features/eod-reset.md">Ascend EOD Reset Training</a></td>
    </tr>
    <tr>
      <td rowspan="5">Multimodal Features</td>
      <td><a href="../features/variable_seq_lengths.md">PP Support for Variable Sequence Lengths</a></td>
    </tr>
    <tr>
      <td><a href="../features/multi_parameter_pipeline.md">Ascend Multi-Parameter Pipeline</a></td>
    </tr>
    <tr>
      <td><a href="../features/multi_parameter_pipeline_and_variable_seq_lengths.md">Ascend Multi-Parameter Pipeline and Variable Sequence Lengths</a></td>
    </tr>
    <tr>
      <td><a href="../features/unaligned_linear.md">Ascend Unaligned Linear Layer</a></td>
    </tr>
    <tr>
      <td><a href="../features/unaligned-ulysses-context-parallel.md">Ascend Unaligned Ulysses Long-Sequence Parallelism</a></td>
    </tr>
    <tr>
      <td>Others</td>
      <td><a href="../features/ops_flops_cal.md">Ascend TFLOPS Calculation</a></td>
    </tr>
  </tbody>
</table>
