# Tests Usage

1. Install `mindspeed`

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
   pytest tests_extend/unit_tests/megatron/test_distrib_optimizer.py
   ```

4. Run the default CI unit tests

    ```shell
   cd {PATH_TO_MEGATRON_LM}
   pytest tests_extend/unit_tests
   ```

5. Run the complete unit test suite, including slow tests

    ```shell
   cd {PATH_TO_MEGATRON_LM}
   pytest tests_extend/unit_tests --run-all
   ```

## Marking Slow Unit Tests

The default CI gate skips tests marked with `pytest.mark.slow`. The complete
suite runs them when `--run-all` is specified.

Mark a test as slow when it is intended for full regression rather than the
default CI gate. Typical examples include large tensor shapes, long sequences,
repeated parameter combinations, and expensive distributed or integration
scenarios. Keep at least one representative smoke case in the default CI gate
when possible.

Mark a whole test file as slow:

```python
import pytest

pytestmark = pytest.mark.slow
```

Mark a single test as slow:

```python
@pytest.mark.slow
def test_large_shape():
    ...
```

Mark only selected parameter combinations as slow:

```python
@pytest.mark.parametrize(
    "seq_len",
    [
        1024,
        pytest.param(32768, marks=pytest.mark.slow),
    ],
)
def test_attention(seq_len):
    ...
```

All tests under `tests_extend/unit_tests/ops/triton` are marked as slow
automatically. New Triton tests do not need to add the marker explicitly.
