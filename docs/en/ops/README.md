# How to Run an Operator

## Prerequisites

+ CANN

+ CANN-NNAL (Ascend-Transformer-Boost)

+ torch_npu

## Compilation and Installation

### 1. Environment Variable Settings

```shell
# The default path is used. Change it if necessary.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### If the Ascend Transformer Boost is used

```shell
# The default path is used. Change it if necessary.
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 2. Included Header Files

+ Latest torch_npu version

+ Latest CANN version

### 3. Installation Scripts

```shell
python3 setup.py build
python3 setup.py bdist_wheel
pip3 install dist/*.whl --force-reinstall
```
