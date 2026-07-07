# DualPipeV

## Background and Challenges

MoE models have long faced critical issues such as a high proportion of All2All communication overhead and large memory consumption. To better achieve All2All communication overlap, DeepSeek proposed the DualPipe arrangement and MoE cross-microbatch forward and backward pass communication overlap [1](https://github.com/deepseek-ai/DualPipe).

The DualPipe pipeline not only creates conditions for cross-microbatch computation and communication parallelism, achieving full All2All overlap in the stable stage, but also delivers better overall performance in terms of bubble ratio and activation memory during the warmup stage compared to traditional approaches. In both the warmup and cooldown stages, DualPipe adopts the dw separation concept from Zero Bubble to further reduce bubbles. At the same time, this pipeline arrangement has a natural affinity for the MTP model structure. Since the first and last stages reside on the same card, this pipeline also makes it easier to adjust the load balancing strategy across the PP dimension. The downside is that the required parameter count per card is doubled.

## Solution

Building upon DualPipe, an improved pipeline arrangement called DualPipeV has been proposed [2](https://zhuanlan.zhihu.com/p/26915547331). It takes half of DualPipe along the PP dimension in the pipeline, while further partitioning the model on top of the PP partitioning, resulting in a V-shaped pipeline arrangement. It addresses the issue of redundant parameters in DualPipe, and the algorithm's launch scale is only half that of DualPipe.

The following figure shows the DualPipeV pipeline arrangement using PP4 with 10 microbatches as an example.
![dualpipev](../figures/dualpipev.png)
In the figure, 0~9 and 10~19 represent two stages of the same microbatch on the same card. The green sections represent the parallel forward and backward pass of different microbatches. After enabling cross-microbatch forward and backward pass communication overlap, both All2All communication and P2P communication can be overlapped by computations without dependencies. For a detailed introduction to this feature, see [MoE cross-microbatch forward and backward pass communication overlap](megatron_moe/megatron-moe-fb-overlap.md).

During the warmup stage, MindSpeed achieves as much P2P communication overlap as possible. During the cooldown stage, in the DualPipeV implementation, the last PP card continuously computes PP_size backward stages. Before the corresponding dw computation is completed, the activations are not fully released, which can significantly increase the peak memory during the cooldown stage in certain recomputation or other memory-intensive scenarios. Therefore, in the MindSpeed implementation, dw separation during the cooldown stage is disabled by default, reducing the memory peak in recomputation scenarios at a small performance cost. A parameter option for dw separation is also provided. The pipeline when dw separation is enabled is shown in the following figure.
![dualpipev_dw_detach](../figures/dualpipev_dw_detach.png).

The following figure shows the PP communication group profiling collected using the PP8 TP2 EP32 DualPipeV strategy on the DeepseekV3 671B model.
![dualpipev_profiling](../figures/dualpipev_profiling.png)

The following table shows the bubble comparison across different pipeline arrangements.
<table>
  <thead>
    <tr>
      <th width="150">Pipeline strategy</th>
      <th width="250">Bubble</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1f1b</td>
      <td>(PP-1)*(F+B)</td>
    </tr>
    <tr>
      <td>VPP</td>
      <td>(PP-1)*(F+B)/v</td>
    </tr>
    <tr>
      <td>DualPipeV</td>
      <td>(PP-1)*(F&amp;B+B-W)-F</td>
    </tr>
  </tbody>
</table>

## Application Scenario

In MoE scenarios, when All2All communication is too long and impacts performance, you can leverage the combined characteristics of DualPipeV and MoE cross-microbatch forward and backward pass communication overlap to improve performance.

## Usage

Add `--schedules-method dualpipev` to the launch script to enable the DualPipeV pipeline arrangement.

Additionally add `--moe-fb-overlap` to the launch script to enable MoE cross-microbatch forward and backward pass communication overlap.

Additionally add `--dualpipev-dw-detach` to the launch script to enable dw detachment in the cooldown stage.

When using DualPipeV, the number of model layers should be set to a multiple of `PP*2`. Also, the number of micro batches per PP group should be set to at least `PP*2`. It is incompatible with other pipeline features such as VPP. It is also currently incompatible with features such as long sequence parallelism, asynchronous DDP, swap-attention, and tp_2d.

## Application Effects

After enabling DualPipeV and MoE cross-microbatch forward and backward pass communication overlap, the All2All communication of MoE can be overlapped, achieving performance improvement. However, in certain scenarios (such as severe load imbalance in MoE), enabling DualPipeV may exacerbate PP load imbalance and cause performance degradation. Therefore, performance changes need to be analyzed on a scenario-by-scenario basis.

Enabling DualPipeV may cause an increase in peak memory. The reason is that in DualPipeV, the output head is assigned to PP rank0, which leads to an increase in both static and dynamic memory. However, in certain scenarios (such as when the last N layers of the model are set as empty layers), since the tail stage is on PP rank0, the peak memory may decrease after enabling DualPipeV. Therefore, memory changes after enabling DualPipeV need to be analyzed on a case-by-case basis.

## Usage Constraints

1. Currently, `use_custom_fsdp` and `dualpipev` are incompatible.
