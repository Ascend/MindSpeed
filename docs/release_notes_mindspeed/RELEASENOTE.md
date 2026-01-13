# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph176851846203314">MindSpeed</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342">2.3.0</p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">正式版本</p>
</td>
</tr>
<tr id="row474524121520"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p6741024121519">发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p574142413157">2026年1月</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">1年</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]  
> 有关MindSpeed的版本维护，具体请参见[分支维护策略](https://gitcode.com/Ascend/MindSpeed/tree/2.3.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。


### 相关产品版本配套说明

**表 1**  MindSpeed软件版本配套关系

|CANN版本|PyTorch版本|Ascend Extension for PyTorch版本|MindSpeed版本|Python版本|
|--|--|--|--|--|
|<br>商用版：8.5.0</br>社区版：8.5.0|<br>2.7.1</br><br>2.6.0</br><br>2.5.1</br>|7.3.0|2.3.0|Python3.10.*x*|
|<br>商用版：8.3.RC1</br>社区版：8.3.RC1|<br>2.7.1</br><br>2.6.0</br><br>2.1.0</br>|7.2.0|2.2.0|Python3.10.*x*|
|<br>商用版：8.2.RC1</br>社区版：8.2.RC1|<br>2.6.0</br><br>2.1.0</br>|7.1.0|2.1.0|Python3.9.x、Python3.10.x|
|<br>商用版：8.1.RC1</br>社区版：8.1.RC1|2.1.0|7.0.0|2.0.0|Python3.8.*x*、Python3.9.*x*、Python3.10.*x*、Python3.11.*x*|


**表 2**  组件间配套表

|PyTorch版本|MindSpeed Core代码分支名称|MindSpeed LLM代码分支名称|MindSpeed MM代码分支名称|MindSpeed RL代码分支名称|
|--|--|--|--|--|
|2.7.1|2.3.0_core_r0.12.1|2.3.0|2.3.0|-|
|2.6.0|2.3.0_core_r0.12.1|-|2.3.0|-|
|2.5.1|-|-|-|-|2.3.0|
|2.1.0/2.6.0/2.7.1|2.2.0_core_r0.12.1|2.2.0|2.2.0|2.2.0|
|2.1.0/2.6.0|2.1.0_core_r0.12.1|2.1.0|2.1.0|2.1.0|
|2.1.0|2.1.0_core_r0.8.0|-|-|-|
|2.1.0|2.0.0_core_r0.8.0|2.0.0|2.0.0|2.0.0|


>[!NOTE]  
>用户可根据需要选择MindSpeed代码分支下载源码并进行安装。



## 版本兼容性说明


|MindSpeed版本|CANN版本|Ascend Extension for PyTorch版本|
|--|--|--|
|2.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|
|2.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|
|2.1.0|CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.1.0|
|2.0.0|CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.0.0|


## 版本使用注意事项

无


## 更新说明

### 新增特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|新增支持模型|支持Ling-mini v2、PHI 3.5、Qwen3-Next模型训练|
|MindSpeed LLM|支持32K/256K训练|Qwen3-30B和Qwen3-32B支持32K/256K训练|
|MindSpeed MM|支持视频生成和编辑模型|支持Wan2.2、VACE2.2等视频生成和编辑模型|
|MindSpeed MM|支持理解模型|支持Qwen3VL-8B/30B/235B、InternVL3.5等理解模型|
|MindSpeed MM|支持强化模型|支持Qwen3VL GRPO-8B/30B等强化学习模型|
|MindSpeed RL|仓库版本升级|支持core 0.12.0 后端|
|MindSpeed Core|支持No Pipeline并行策略|1F1B Overlap支持无流水线（No Pipeline）并行策略|
|MindSpeed Core|支持与 mbridge 的集成与兼容|Verl 支持与 mbridge 的集成与兼容|
|MindSpeed Core|支持MLA结构|Verl 支持 MLA结构|
|MindSpeed Core|支持算子融合|Verl mcore 后端已完成算子融合支持|
|MindSpeed Core|支持负载均衡优化策略|MoE支持参数数量互寻的负载均衡优化策略|

### 删除特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|模型下架|下架模型清单：</br><br>OpenSora1.0-5.5B</br><br>OpenSora1.2-5.2B</br><br>OpenSoraPlan1.2-8.7B</br><br>LLaVA1.5-7B</br><br>InternVL2.0-2B/8B/26B/76B</br><br>Qwen2.5VL GRPO-3B/7B/32B</br><br>Qwen2.5VL DAPO-7B</br>

### 接口变更说明

无

### 已解决问题

无


### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

-   对业务的影响

    软件版本升级过程中会导致业务中断。

-   对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无


## 配套文档


|文档名称|内容简介|更新说明|
|--|--|--|
|《[分布式训练加速库迁移指南](https://gitcode.com/Ascend/MindSpeed/blob/2.3.0_core_r0.12.1/docs/user-guide/model-migration.md)》|指导具有一定Megatron-LM模型训练基础的用户将原本在其他硬件平台（例如GPU）上训练的模型迁移到昇腾平台（NPU），主要关注点是有效地将Megatron-LM训练模型迁移至昇腾平台上， 并在合理精度误差范围内高性能运行。|-|
|《[MindSpeed LLM基于PyTorch迁移指南](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E8%BF%81%E7%A7%BB%E6%8C%87%E5%8D%97%2FMindSpeed-LLM%E8%BF%81%E7%A7%BB%E6%8C%87%E5%8D%97-Pytorch%E6%A1%86%E6%9E%B6.md)》|以Qwen2.5-7B为例，通过比对HuggingFace的模型结构，抓取其模型特性，并将其特性结构适配到MindSpeed LLM中，从而使MindSpeed LLM与开源模型结构对齐，使权重和输入数据做相同运算，确保输出对齐，同时保持或提升该模型的性能。|-|
|《[MindSpeed MM迁移调优指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/2.3.0/docs/user-guide/model-migration.md)》|以Qwen2-VL-7B为例，基于华为昇腾芯片产品（NPU）开发，将原本运行在GPU或其他硬件平台的深度学习模型迁移至NPU， 并保障模型在合理精度误差范围内高性能运行，并对昇腾芯片进行极致优化以发挥芯片最大性能。|-|
|《[MindSpeed RL使用指南](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.3.0/docs/solutions/r1_zero_qwen25_32b.md)》|以Qwen2.5-32B为例，基于GPRO+规则奖励打分进行训练，复现DeepSeek-R1-Zero在Math领域的工作。|-|



## 病毒扫描及漏洞修补列表

### 病毒扫描结果


|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-01-03 08:00:00.0|2025-12-06|无病毒，无恶意|
|Kasperaky|12.0.0.6672|2026-01-05 10:03:00.0|2025-12-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100087|2025-12-06|无病毒，无恶意|



### 漏洞修补列表

无