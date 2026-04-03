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
> 有关MindSpeed的版本维护，具体请参见[分支维护策略](https://gitcode.com/Ascend/MindSpeed/tree/26.0.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

**表 1**  MindSpeed Core软件版本配套表

|MindSpeed Core代码分支名称|CANN版本|Ascend Extension for PyTorch版本|Python版本|PyTorch版本|MindSpeed-Core-MS版本|
|--|--|--|--|--|--|
|26.0.0_core_r0.12.1|9.0.0|26.0.0|Python3.10|2.7.1|r0.5.0|
|2.3.0_core_r0.12.1|8.5.0|7.3.0|Python3.10|2.7.1|r0.5.0|
|2.2.0_core_r0.12.1|8.3.RC1|7.2.0|Python3.10|2.7.1|r0.4.0|

>[!NOTE]  
>用户可根据需要选择MindSpeed代码分支下载源码并进行安装。

## 版本兼容性说明

|MindSpeed版本|CANN版本|Ascend Extension for PyTorch版本|MindSpeed-Core-MS版本|
|--|--|--|--|
|26.0.0_core_r0.12.1|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>|26.0.0|r0.5.0|
|2.3.0_core_r0.12.1|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|r0.5.0|
|2.2.0_core_r0.12.1|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|r0.4.0|

## 版本使用注意事项

无

## 更新说明

### 新增特性

 - Atlas A3 训练产品系列支持代际AI QoS参数配置。
 - 支持KVAllgather CP并行方案。
 - FSDP后端Device解耦。

### 删除特性

无

### 接口变更说明

无

### 已解决问题

 - 修复ATB算子多流场景的偶现精度异常。
 - 修复TE分支Norm重计算部分场景下功能异常。

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
|《[MindSpeed快速入门](../zh/user-guide/getting_started.md)》|介绍基于MindSpeed如何实现Megatron-LM在昇腾设备上的高效运行。|-|
|《[MindSpeed安装指导](../zh/install_guide.md)》|指导如何在NPU上基于PyTorch和MindSpore框架完成MindSpeed的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装，以及两种框架下的完整安装流程，帮助用户快速搭建大模型分布式训练环境。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-04-02|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00|2026-04-02|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100588|2026-04-02|无病毒，无恶意|

### 漏洞修补列表

无
