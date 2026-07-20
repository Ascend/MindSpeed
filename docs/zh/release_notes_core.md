# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MindSpeed</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>26.1.0_core_r0.12.1</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>正式版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>2026年7月</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
> 有关MindSpeed的版本维护，具体请参见[分支维护策略](https://gitcode.com/Ascend/MindSpeed/tree/26.1.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

**表 1**  MindSpeed Core软件版本配套表

|MindSpeed Core代码分支名称|CANN版本|TorchNPU版本|Python版本|PyTorch版本|
|--|--|--|--|--|
|master（在研版本）|在研版本|在研版本|Python3.10|2.7.1|
|26.1.0_core_r0.12.1|9.1.0|26.1.0|Python3.10|2.7.1|
|26.0.0_core_r0.12.1|9.0.0|26.0.0|Python3.10|2.7.1|

>[!NOTE]
>
>用户可根据需要选择MindSpeed代码分支下载源码并进行安装。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2**  MindSpeed Core与TorchNPU版本兼容

<table style="table-layout: fixed; width: 750px ; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed Core</th>
      <th colspan="4">TorchNPU版本</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**表 3**  MindSpeed Core与CANN版本兼容

<table style="table-layout: fixed; width: 750px ; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed Core</th>
      <th colspan="4">CANN版本</th>
    </tr>
    <tr>
      <th>8.3.RC1</th>
      <th>8.5.0</th>
      <th>9.0.0</th>
      <th>9.1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## 版本使用注意事项

无

## 更新说明

### 新增特性

- 新增FP8/MXFP8/HiFloat8低精格式训练支持，以及w8a16量化支持。
- 新增MXFP8-32x32量化及FSDP支持，量化后释放bf16权重优化内存。
- 新增SwapMuon及mcore muon特性，支持checkpoint保存与加载。
- 新增DeepSeek V4模型适配及自定义PP布局支持。
- 新增TE算子层Hamilton attention实现。

### 删除特性

- 删除SFA/SFAG/SLI临时版本算子适配，相关功能通过正式算子承载。
- 删除mindspeed/lite模块，Triton算子迁移至mindspeed/ops/triton目录。

### 接口变更说明

无

### 已解决问题

- 修复TE分支LayerNormLinear初始化权重顺序与NVTE不一致问题，以及LayerNorm偏置未初始化为零的问题。
- 修复GMM算子、NPU sparse_attn_sharedkv算子异常，以及l2norm批量一致性和recompute_w_u_fwd的NaN错误。
- 修复fboverlap场景下异常内存占用问题，以及Triton算子chunk_bwd_dqkwg时间退化问题。
- 修复VeRL场景下HCCL缓冲区错误。

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[MindSpeed快速入门](../zh/user-guide/quickstart.md)》|介绍基于MindSpeed如何实现Megatron-LM在昇腾设备上的高效运行。|-|
|《[MindSpeed安装指导](../zh/user-guide/install_guide.md)》|指导如何在NPU上基于PyTorch框架完成MindSpeed的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装的完整安装流程，帮助用户快速搭建大模型分布式训练环境。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-07-06|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00|2026-07-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100588|2026-07-06|无病毒，无恶意|

### 漏洞修补列表

无
