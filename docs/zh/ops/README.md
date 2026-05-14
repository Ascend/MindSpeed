# 如何运行算子

## 前置安装

+ CANN
+ CANN-NNAL(Ascend-Transformer-Boost)
+ torch_npu

## 编译和安装

### 1. 设置环境变量

```shell
# 默认路径，请按需修改。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 如果使用 Ascend-Transformer-Boost

```shell
# 默认路径，请按需修改。
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 2. 包含头文件

+ 最新版本的 torch_npu
+ 最新版本的 cann

### 3. 安装脚本

```shell
python3 setup.py build
python3 setup.py bdist_wheel
pip3 install dist/*.whl --force-reinstall
```
