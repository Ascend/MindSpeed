# yaml-cfg使用指导

1. 在`Megatron-LM`目录下修改`megatron/core/transformer/transformer_config.py`文件，删除`max_position_embeddings: int = 0`。
此参数在`Megatron-LM`为废弃参数且会导致报错。

2. 复制 `tests_extend`文件夹到`Megatron-LM`根目录下。
   
    ```shell
    cp -r tests_extend {PATH_TO_MEGATRON_LM}
    ```

3. 在`Megatron-LM`目录下，执行示例脚本。
   ```shell
   cd {PATH_TO_MEGATRON_LM}
   bash tests_extend/system_tests/yaml_args_example/pretrain_yaml_args.sh
   ```

4. `example.yaml`为示例yaml文件，在实际使用中，可能会出现因示例yaml中未包含特定参数导致脚本报错，需在示例yaml中添加相应参数后，重新执行脚本。