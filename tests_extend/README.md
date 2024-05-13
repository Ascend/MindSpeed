# Tests Usage

1. Install `ascendspeed`
    ```shell
    pip install -e .
    ```

2. Copy the entire `tests_extend` to the root path of `Megatron-LM`
   
    ```shell
    cp -r tests_extend {PATH_TO_MEGATRON_LM}
    ```

3. Run a single test by pytest command line under `Megatron-LM` root path
   ```shell
   cd {PATH_TO_MEGATRON_LM}
   pytest tests_extend/unit_tests/optimizer/test_distrib_optimizer.py
   ```

4. Run the whole tests
    ```shell
   cd {PATH_TO_MEGATRON_LM}
   pytest tests_extend
   ```