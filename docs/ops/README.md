# How to run the ops?

## previous installation
+ CANN
+ CANN-NNAL(Ascend-Transformer-Boost)
+ torch_npu

## compile and install
### 1. set the environment variables
 ```shell
# Default path, change it if needed.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
 ```
#### if use Ascend-Transformer-Boost
 ```shell
# Default path, change it if needed.
source /usr/local/Ascend/nnal/atb/set_env.sh 
 ```

### 2. include head files

+ newest torch_npu
+ newest cann

### 3. install scripts
```shell
python3 setup.py build
python3 setup.py bdist_wheel
pip3 install dist/*.whl --force-reinstall
```
