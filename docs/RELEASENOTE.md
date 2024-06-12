# MindSpeed 版本说明书
-   [MindSpeed 1.0](#FrameworkPTAdapter-5-0-RC1md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)


<h2 id="MindSpeed 1.0md">MindSpeed 1.0</h2>

<h3 id="用户须知md">用户须知</h3>

本框架基于NVIDIA主导的开源Megatron进行修改，采用插件化适配方式，延续原生的Megatron特性，使用NPU进行大模型加速训练；代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法。

<h3 id="新增特性md">新增特性</h3>

**表 1** MindSpeed支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>
<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="25.590000000000003%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="15.52%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="58.89%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2620183971614"><td class="cellrowborder" rowspan="8" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p0819102247"><a name="p0819102247"></a><a name="p0819102247"></a>Megatron原生特性</p>
<p id="p15488161812213"><a name="p15488161812213"></a><a name="p15488161812213"></a></p>
<p id="p17381229135615"><a name="p17381229135615"></a><a name="p17381229135615"></a></p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p76365489137"><a name="p76365489137"></a><a name="p76365489137"></a>数据并行</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p363616485131"><a name="p363616485131"></a><a name="p363616485131"></a>支持数据并行训练策略</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>张量并行</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持张量并行训练策略</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>流水并行</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持流水并行训练策略</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>虚拟流水并行</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持虚拟流水并行训练策略</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>序列并行</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持序列并行训练策略</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>重计算</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持选择性重计算和完全重计算策略</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>分布式优化器</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持分布式优化器策略，将优化器状态拆分到所有DP组间</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>异步DDP</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持异步DDP，在进行梯度更新时，将数据并行组中的通信和计算并行执行</p>
</td>
</tr>
<tr id="row3722227133312"><td class="cellrowborder" rowspan="4" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p107221327153315"><a name="p107221327153315"></a><a name="p107221327153315"></a>昇腾专有算法</p>
<p id="p7778931115613"><a name="p7778931115613"></a><a name="p7778931115613"></a></p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p153563917719"><a name="p153563917719"></a><a name="p153563917719"></a>TP 重计算通信优化</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p193246215619"><a name="p193246215619"></a><a name="p193246215619"></a>重计算通信算子消除，优化重计算层划分，实现大模型训练通信性能提升</p>
</td>
</tr>
<tr id="row6631141014305"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1416445591310"><a name="p1416445591310"></a><a name="p1416445591310"></a>内存碎片优化</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1281850105718"><a name="p1281850105718"></a><a name="p1281850105718"></a>通过对不同生命周期的tensor进行分别管理，以减少内存碎片</p>
</td>
</tr>
<tr id="row577853110564"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p546144505813"><a name="p546144505813"></a><a name="p546144505813"></a>自适应选择重计算</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p47781431165614"><a name="p47781431165614"></a><a name="p47781431165614"></a>支持通过自动调整训练内存大小来自动选择重新计算策略</p>
</td>
</tr>
<tr id="row6631141014305"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1416445591310"><a name="p1416445591310"></a><a name="p1416445591310"></a>计算通信并行优化</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1281850105718"><a name="p1281850105718"></a><a name="p1281850105718"></a>通过将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖</p>
</td>
</tr>
<tr id="row3722227133312"><td class="cellrowborder" rowspan="1" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p107221327153315"><a name="p107221327153315"></a><a name="p107221327153315"></a>昇腾自定义算子</p>
<p id="p7778931115613"><a name="p7778931115613"></a><a name="p7778931115613"></a></p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p153563917719"><a name="p153563917719"></a><a name="p153563917719"></a>npu_dropout_add_layer_norm</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p193246215619"><a name="p193246215619"></a><a name="p193246215619"></a>支持自定义算子npu_dropout_add_layer_norm调用</p>
</td>
</tr>
</tbody>
</table>
<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>
<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>RotaryMul融合算子精度问题</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>RotaryMul融合算子反向在cann的0419商发版本上，部分场景存在精度问题。</p>
</td>
</tr>
</tbody>
</table>
<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e