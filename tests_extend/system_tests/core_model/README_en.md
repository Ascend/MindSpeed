# How to Run Operators

## Prerequisites

+ CANN
+ CANN-NNAL (Ascend-Transformer-Boost)
+ torch_npu

## Compilation and Installation

### 1. Set Environment Variables

```shell
# Default path, modify as needed
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### When Ascend-Transformer-Boost Is Used

```shell
# Default path, modify as needed
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 2. Include Header Files

+ Latest version of torch_npu
+ Latest version of CANN

### 3. Install the Script

```shell
python3 setup.py build
python3 setup.py bdist_wheel
pip3 install dist/*.whl --force-reinstall
```
