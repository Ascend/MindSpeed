# DualPipeV

## 背景与挑战

MoE类模型一直以来存在All2All通信耗时占比高和显存开销大等关键问题。为了更好地进行All2All通信掩盖，DeepSeek提出了DualPipe流水排布和MoE跨microbatch前反向通信掩盖[1](https://github.com/deepseek-ai/DualPipe)。

DualPipe流水不仅可以创造跨microbatch计算通信并行的条件，实现稳定阶段的All2All全掩盖，在空泡比率和warmup阶段激活显存上相比传统方案也有更好地综合表现。在warmup和cooldown阶段，DualPipe都采取了Zero Bubble思想中的dw分离来进一步压缩气泡。同时，该流水排布天然亲和MTP模型结构，由于首尾stage都在同一张卡上，该流水也可以更方便地调整PP间的负载均衡策略。缺点是每张卡上需要的参数量翻倍。


## 解决方案

在DualPipe的基础之上，一种改进流水排布DualPipeV被提出[2](https://zhuanlan.zhihu.com/p/26915547331)。它在流水上从PP维度截取DualPipe的一半，同时将模型在PP切分的基础上进行进一步地切分，流水呈V字形排布。它解决了DualPipe冗余参数的问题，算法启动规模也只需要DualPipe的一半。

下图以PP4，10个microbatch为例展示DualPipeV流水排布。
![dualpipev](../../sources/images/dualpipev.png)
图中0~9和10~19代表同一个microbatch在同一张卡上的两个stage。绿色部分代表不同microbatch的前反向并行，开启跨micro batch前反向通信掩盖后，All2All通信和P2P通信都可以被没有依赖关系的计算掩盖，该特性的详细介绍参见[MoE跨microbatch前反向通信掩盖](megatron_moe/megatron-moe-fb-overlap.md)。

在warmup阶段，MindSpeed实现了尽可能多的P2P通信掩盖。在cooldown阶段，DualPipeV实现中PP尾卡会连续计算PP_size个反向stage，在对应的dw计算完成之前，激活值并不会完全释放，这会导致在某些重计算或其他内存特性场景下，cooldown阶段的峰值内存被大大拉高。因此MindSpeed实现中默认取消了cooldown阶段的dw分离，以较小的性能代价降低了重计算场景下内存峰值。同时也给出了dw分离的参数选项，开启dw分离时的流水如下图所示。
![dualpipev_dw_detach](../../sources/images/dualpipev_dw_detach.png)。

下图展示了在DeepseekV3 671B模型上采用PP8 TP2 EP32 DualPipeV策略采集的PP通信组profiling。
![dualpipev_profiling](../../sources/images/dualpipev_profiling.png)

下表展示了不同流水排布中bubble对比。
<table><thead>
  <tr>
    <th width='150'>流水策略</th>
    <th width='250'>气泡</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5"> 1f1b </td>
    <td>(PP-1)*(F+B)</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> VPP </td>
    <td>(PP-1)*(F+B)/v</td>
  </tr>
<tbody>
  <tr>
    <td rowspan="5"> DualPipeV </td>
    <td>(PP-1)*(F&B+B-W)-F</td>
  </tr>
</table>

## 使用场景

在Moe场景，All2All通信过长而影响性能时，可以采用DualPipeV和MoE跨microbatch前反向通信掩盖的组合特性来提升性能。

## 使用方法
在启动脚本中添加`--schedules-method dualpipev`即可开启DualPipeV流水排布。

在启动脚本中额外添加`--moe-fb-overlap`来开启MoE跨microbatch前反向通信掩盖。

在启动脚本中额外添加`--dualpipev-dw-detach`来开启cooldown阶段的dw分离。

使用DualPipeV时，模型层数设置应为`PP*2`的倍数。同时每个PP组的micro batch数至少设置为`PP*2`。与VPP等其他流水特性不兼容。与长序列并行、异步DDP、swap-attention、tp_2d等特性暂不兼容。


## 使用效果

在开启DualPipeV和MoE跨microbatch前反向通信掩盖后可以对MoE的All2All通信进行掩盖，获得性能提升。但在某些场景（例如MoE中严重负载不均衡）时，开启DualPipeV可能会加剧PP负载不均，造成性能劣化，因此性能变化需要按场景具体分析。

开启DualPipeV可能会造成内存峰值上升，原因是DualPipeV中output head会被分配到PP rank0，这会同时带来静态内存和动态内存的上升。但在某些场景（例如设置模型后N层为空层时）中，由于尾stage在PP rank0,DualPipeV开启后峰值内存会降低。因此开启DualPipeV之后的内存变化需要具体分析。
