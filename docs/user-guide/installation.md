# 软件安装

## 版本配套表

MindSpeed支持Atlas 800T A2等昇腾训练硬件形态。软件版本配套表如下：

| MindSpeed版本            | Megatron版本  | PyTorch版本    | torch_npu版本 | CANN版本  | Python版本                               |
|------------------------|-------------|--------------|-------------|---------|----------------------------------------|
| master（主线）             | Core 0.12.1 | 2.1.0, 2.6.0 | 在研版本        | 在研版本    | Python3.9.x, Python3.10.x              |
| core_r0.10.0（主线）       | Core 0.10.0 | 2.1.0        | 在研版本        | 在研版本    | Python3.9.x, Python3.10.x              |
| core_r0.9.0（主线）        | Core 0.9.0  | 2.1.0        | 在研版本        | 在研版本    | Python3.9.x, Python3.10.x              |
| 2.1.0_core_r0.12.1（商用） | Core 0.12.1 | 2.1.0, 2.6.0 | 7.1.0     | 8.2.RC1 | Python3.9.x, Python3.10.x              |
| 2.1.0_core_r0.8.0（商用）  | Core 0.8.0  | 2.1.0        | 7.1.0     | 8.2.RC1 | Python3.8.x, Python3.9.x, Python3.10.x |
| 2.0.0_core_r0.8.0（商用）  | Core 0.8.0  | 2.1.0        | 7.0.0       | 8.1.RC1 | Python3.8.x, Python3.9.x, Python3.10.x |
| 1.0.0_core_r0.7.0（商用）  | Core 0.7.0  | 2.1.0        | 6.0.0       | 8.0.0   | Python3.8.x, Python3.9.x, Python3.10.x |
| 1.0.0_core_r0.6.0（商用）  | Core 0.6.0  | 2.1.0        | 6.0.0       | 8.0.0   | Python3.8.x, Python3.9.x, Python3.10.x |


[昇腾辅助软件](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中有更多关于PyTorch和CANN的版本信息。

## 安装操作

### 1. 安装依赖的软件

在安装MindSpeed之前，请参考[版本配套表](#版本配套表)，安装配套的昇腾软件栈，软件列表如下：

<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>软件安装指南</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>

### 2. 下载MindSpeed源码并切换至2.1.0_core_r0.8.0分支



 ```shell
 git clone https://gitee.com/ascend/MindSpeed.git
 cd MindSpeed
 git checkout 2.1.0_core_r0.8.0
 ```

### 3. 安装MindSpeed

```shell
pip install -e MindSpeed
```

> 注：如有旧版本MindSpeed，请先[卸载](#卸载)，再进行安装


### 4. 获取Megatron-LM源码切换 core_r0.8.0 版本

获取并切换 Megatron-LM 版本至 core_r0.8.0 版本，如下所示：
 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_r0.8.0
 ```


 ## 卸载
 
```shell
pip uninstall -y mindspeed
```
