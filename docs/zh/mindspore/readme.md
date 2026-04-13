# MindSpeed MindSpore介绍

MindSpeed已支持接入华为自研AI框架MindSpore，旨在提供华为全栈易用的端到端的大模型训练解决方案，以此获得更极致的性能体验。MindSpore后端提供了一套对标PyTorch的API，用户无需进行额外代码适配即可无缝切换。

---

## 安装

### 安装依赖

<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>软件安装指南</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">《 <a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Debian&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>ops（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
    <td>MindSpore</td>
    <td rowspan="1">《 <a href="https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85">MindSpore AI框架安装指南</a> 》</td>
  </tr>
</table>

### 获取 [MindSpeed-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/) 代码仓

执行以下命令拉取MindSpeed-Core-MS代码仓，并安装Python三方依赖库，如下所示：

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
cd MindSpeed-Core-MS
pip install -r requirements.txt
```

可以参考MindSpeed-Core-MS目录下提供的[一键适配命令脚本](https://gitcode.com/Ascend/MindSpeed-Core-MS/#%E4%B8%80%E9%94%AE%E9%80%82%E9%85%8D)， 拉取并适配相应版本的MindSpeed、Megatron-LM和MSAdapter。

若使用MindSpeed-Core-MS目录下的一键适配命令脚本（如[auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh)）可忽略后面步骤。

### 获取并适配相应版本的 MindSpeed、Megatron-LM 和 MSAdapter

1. 进入MindSpeed-Core-MS目录后，获取指定版本仓库的源码：

   ```shell
   # 获取指定版本的MindSpeed源码：
   git clone https://gitcode.com/Ascend/MindSpeed.git -b master
   
   # 获取指定版本的Megatron-LM源码：
   git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1
   
   # 获取指定版本的MSAdapter源码：
   git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
   ```

   具体版本对应关系参考MindSpeed-Core-MS下的[一键适配命令脚本](https://gitcode.com/Ascend/MindSpeed-Core-MS/#%E4%B8%80%E9%94%AE%E9%80%82%E9%85%8D)，如[auto_convert.sh](https://gitcode.com/Ascend/MindSpeed-Core-MS/blob/master/auto_convert.sh)。

2. 设置环境变量：

   ```shell
   # 在MindSpeed-Core-MS目录下执行
   # 若在环境中PYTHONPATH等环境变量失效（例如退出容器后再进入等），需要重新设置环境变量
   MindSpeed_Core_MS_PATH=$(pwd)
   export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/MindSpeed:$PYTHONPATH
   echo $PYTHONPATH
   ```

3. 如需使用Ascend Transformer Boost（ATB）加速库算子，请先安装 CANN-NNAL 并初始化添加环境，例如：

   ```shell
   # CANN-NNAL默认安装路径为：/usr/local/Ascend/nnal
   # 运行CANN-NNAL默认安装路径下atb文件夹中的环境配置脚本set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

## 快速上手

1. 仅仅一行代码就可以轻松使能 MindSpeed 的各项功能。以 GPT 模型为例：在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行：`import mindspeed.megatron_adaptor`，即如下修改：

    ```diff
     import os
     import torch
    +import mindspeed.megatron_adaptor
     from functools import partial
     from typing import Union
    ```

2. （可选）若未准备好相应训练数据，则需进行数据集的下载及处理供后续使用。数据集准备流程可参考
<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0003.html">数据集处理</a>。

3. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。以下示例可供参考。

    ```shell
    bash ./train_distributed_ms.sh
    ```

---

## 自定义优化级别

MindSpeed 提供了多层次的优化解决方案，并划分为三个层级，用户可根据实际需求灵活启用任意层级。高层级兼容低层级的能力，确保了整个系统的稳定性和扩展性。
用户可以通过设置启动脚本中的 `--optimization-level {层级}` 参数来自定义开启的优化层级。该参数支持以下配置：

<table><thead>
  <tr>
    <th width='50'>层级</th>
    <th width='300'>层级名称</th>
    <th width='600'>介绍</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 0 </td>
    <td>基础兼容层</td>
    <td>提供Megatron-LM框架对NPU的支持，确保无缝集成。该层包含基础功能集patch，保证可靠性和稳定性，为高级优化奠定基础。</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 1 </td>
    <td>亲和性增强层🔥</td>
    <td>兼容L0能力，集成高性能融合算子库，结合昇腾亲和的计算优化，充分释放昇腾算力，显著提升计算效率。</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="5" style="text-align: center; vertical-align: middle"> 2 </td>
    <td>自研加速算法层🔥🔥</td>
    <td>默认值。该模式兼容了L1, L0能力，并集成了昇腾多项自主研发核心技术成果，可提供全面的性能优化。</td>
  </tr>
</tbody>
</table>

## MindSpeed 中采集Profile数据

📝 MindSpeed 支持命令式开启Profile采集数据，命令配置介绍如下：

| 配置命令                    | 命令含义                                                                              | 
|-------------------------|-----------------------------------------------------------------------------------|
| --profile               | 打开profile开关                                                                       |
| --profile-step-start    | 配置开始采集步，未配置时默认为10, 配置举例: --profile-step-start 30                                 |
| --profile-step-end      | 配置结束采集步，未配置时默认为12, 配置举例: --profile-step-end 35                                   |
| --profile-level         | 配置采集等级，未配置时默认为level0, 可选配置: level0, level1, level2, 配置举例: --profile-level level1 |
| --profile-with-cpu      | 打开cpu信息采集开关                                                                       |
| --profile-with-stack    | 打开stack信息采集开关                                                                     |
| --profile-with-memory   | 打开memory信息采集开关，配置本开关时需打开--profile-with-cpu                                       |
| --profile-record-shapes | 打开shapes信息采集开关                                                                    |
| --profile-save-path     | 配置采集信息保存路径, 未配置时默认为./profile_dir, 配置举例: --profile-save-path ./result_dir          |
| --profile-ranks         | 配置待采集的ranks，未配置时默认为-1，表示采集所有rank的profiling数据，配置举例: --profile-ranks 0 1 2 3, 需注意: 该配置值为每个rank在单机/集群中的全局值   |

---
