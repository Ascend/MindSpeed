# Tests Usage Instructions

1. Install `mindspeed`.

    ```shell
    pip install -e .
    ```

2. Copy the entire `tests_extend` directory to the root directory of `Megatron-LM`.

    ```shell
    cp -r tests_extend {PATH_TO_MEGATRON_LM}
    ```

3. Run a single test through the `pytest` command line in the root directory of `Megatron-LM`.

   ```shell
   cd {PATH_TO_MEGATRON_LM}
   pytest tests_extend/unit_tests/megatron/test_distrib_optimizer.py
   ```

4. Run all tests.

    ```shell
   cd {PATH_TO_MEGATRON_LM}
   pytest tests_extend
   ```
