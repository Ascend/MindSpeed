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
<td class="cellrowborder" valign="top" width="73.75%" headers=><p>26.0.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>正式版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>2026年4月</p>
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
> 有关MindSpeed的版本维护，具体请参见[分支维护策略](../../README.md#分支维护策略)。

### 相关产品版本配套说明

详细版本配套信息（包含所有历史版本兼容矩阵）请参考：[版本配套表](./user-guide/installation.md#版本配套表)。

## 版本兼容性说明

|MindSpeed版本|CANN版本|Ascend Extension for PyTorch版本|MindSpeed-Core-MS版本|
|--|--|--|--|
|26.0.0_core_r0.12.1 |CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>| 26.0.0 | - |
|2.3.0_core_r0.12.1|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|r0.5.0|
|2.2.0_core_r0.12.1|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|r0.4.0|

## 版本使用注意事项

无

## 更新说明

### 新增特性

无

### 删除特性

无

### 接口变更说明

无

### 已解决问题

无

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
|《[分布式训练加速库迁移指南](./user-guide/model-migration.md)》|指导具有一定Megatron-LM模型训练基础的用户将原本在其他硬件平台（例如GPU）上训练的模型迁移到昇腾平台（NPU），主要关注点是有效地将Megatron-LM训练模型迁移至昇腾平台上， 并在合理精度误差范围内高性能运行。|-|
|《[MindSpeed LLM基于PyTorch迁移指南](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E8%BF%81%E7%A7%BB%E6%8C%87%E5%8D%97%2FMindSpeed-LLM%E8%BF%81%E7%A7%BB%E6%8C%87%E5%8D%97-PyTorch%E6%A1%86%E6%9E%B6.md)》|以Qwen2.5-7B为例，通过比对HuggingFace的模型结构，抓取其模型特性，并将其特性结构适配到MindSpeed LLM中，从而使MindSpeed LLM与开源模型结构对齐，使权重和输入数据做相同运算，确保输出对齐，同时保持或提升该模型的性能。|-|
|《[MindSpeed MM迁移调优指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/2.3.0/docs/user-guide/model-migration.md)》|以Qwen2-VL-7B为例，基于华为昇腾芯片产品（NPU）开发，将原本运行在GPU或其他硬件平台的深度学习模型迁移至NPU， 并保障模型在合理精度误差范围内高性能运行，并对昇腾芯片进行极致优化以发挥芯片最大性能。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

无

### 漏洞修补列表

无
