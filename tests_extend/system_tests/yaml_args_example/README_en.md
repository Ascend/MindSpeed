# yaml-cfg usage guide

1. In the `Megatron-LM` directory, modify the `megatron/core/transformer/transformer_config.py` file and delete `max_position_embeddings: int = 0`.
This parameter is deprecated in `Megatron-LM` and will cause an error.

2. Copy the `tests_extend` folder to the `Megatron-LM` root directory.

    ```shell
    cp -r tests_extend {PATH_TO_MEGATRON_LM}
    ```

3. In the `Megatron-LM` directory, execute the example script.

   ```shell
   cd {PATH_TO_MEGATRON_LM}
   bash tests_extend/system_tests/yaml_args_example/pretrain_yaml_args.sh
   ```

4. `example.yaml` is a sample YAML file. In actual use, the script may report an error because the sample yaml does not include a specific parameter. You need to add the corresponding parameter to the sample YAML and then re-execute the script.
