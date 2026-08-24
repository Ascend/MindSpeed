# API Reference

## MindSpeedFeature Base Class

- **1. `__init__`**

```python
def __init__(self, feature_name: str, optimization_level: int = 2)
```

**Parameter Description:**

| Name | Type | Description |
| ------ | ------ | ------ |
| `feature_name` | str | Feature name, all lowercase, separated by `-`, such as `async-log-allreduce` |
| `optimization_level` | int | Optimization level: `0=basic optimization`, `1=affinity optimization`, `2=advanced optimization` |

**Default Behavior:**

- When `optimization_level == 0`, `default_patches` is automatically set to `True` (enabled by default)
- `feature_name` is automatically converted to underscore format for storage.

- **2. `register_args`**

```python
def register_args(self, parser: ArgumentParser)
```

**Purpose**: Register feature-related command-line arguments and is called regardless of whether the feature is enabled.

**Parameter Description:**

- **parser**: `argparse.ArgumentParser` instance, the argument parser

**Usage Example:**
Create a feature argument group via `parser.add_argument_group`, then create specific arguments via `group.add_argument`.

```python
def register_args(self, parser):
    group = parser.add_argument_group(title=self.feature_name)
    group.add_argument('--my-feature', action='store_true', help='...')
```

- **3. `pre_validate_args`**

```python
def pre_validate_args(self, args: Namespace)
```

**Purpose**: Temporarily modify certain arguments before Megatron argument validation to bypass the native validation logic.

**Typical Scenarios**:

```python
def pre_validate_args(self, args):
    self._saved_cp_size = args.context_parallel_size
    args.context_parallel_size = 1  # Temporarily modify to bypass validation.
```

- **4. `validate_args`**

```python
def validate_args(self, args: Namespace)
```

**Purpose**: The core method for parameter validation, used to validate parsed arguments against business rules.

**Usage Example:**

```python
def validate_args(self, args):
    if args.context_parallel_size > 1 and args.seq_length % args.context_parallel_size != 0:
        raise AssertionError("seq_length must be divisible by context_parallel_size")
```

- **5. `post_validate_args`**

```python
def post_validate_args(self, args: Namespace)
```

**Purpose**: This method is called after `validate_args` and is used to restore the original parameter values after bypassing the native validation.

**Typical Scenario**:

```python
def post_validate_args(self, args):
    args.context_parallel_size = self._saved_cp_size  # Restore the original value
```

- **6. `pre_register_patches`**

```python
def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace)
```

**Purpose**: Register patches before importing Megatron.

- **7. `register_patches`**

```python
def register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace)
```

**Purpose**: Register feature-related functional patches.

**Trigger Condition**: Called only when `is_need_apply(args)` returns `True`.

**Usage Example:**

```python
def register_patches(self, patch_manager, args):
    from mindspeed.core.my_feature import my_new_function
    patch_manager.register_patch('module.path.to.function', my_new_function)
```

- **8. `is_need_apply`**

```python
def is_need_apply(self, args)
```

**Purpose**: Determine whether the feature needs to be applied.

**Determination Logic**:

```python
return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, None)) \
    or self.default_patches
```

- **9. `incompatible_check`**

```python
def incompatible_check(self, global_args, check_args)
```

**Purpose**: Detect conflicts between parameters.

**Validation Logic**: If both the current feature in `global_args` and `check_args` are `True`, an exception is raised.

**Usage Example:**

```python
def validate_args(self, args):
    self.incompatible_check(args, 'other_feature')
```

- **10. `dependency_check`**

```python
def dependency_check(self, global_args, check_args)
```

**Purpose**: Check whether the dependency conditions required by the feature are satisfied.

**Validation Logic**: If the current feature is `True` but `check_args` is `False`, an exception is raised.

**Usage Example:**

```python
def validate_args(self, args):
    self.dependency_check(args, 'required_feature')
```

- **11. `add_parser_argument_choices_value`**

```python
@staticmethod
def add_parser_argument_choices_value(parser, argument_name, new_choice)
```

**Purpose**: Add a new choice option to an existing argument.

**Parameter Description:**

| Name | Type | Description |
| ------ | ------ | ------ |
| `parser` | ArgumentParser | Argument parser |
| `argument_name` | str | Target argument name (with or without `--`) |
| `new_choice` | str | New option value to add |

### MindSpeedPatchesManager Class

- **1. `register_patch`**

```python
def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False)
```

**Purpose**: Register a function/method that needs to be replaced or enhanced.

**Parameter Description:**

| Name | Type | Description |
| ------ | ------ | ------ |
| `orig_func_name` | str | Full path of the target function, such as `module.class.method` |
| `new_func`| callable | Replacement function, which can be None |
| `force_patch` | bool | Whether to forcibly overwrite an existing patch |
| `create_dummy` | bool | Whether to create a dummy function when the target function does not exist |

**Core mechanisms:**

1. **Delayed effect**: A patch does not take effect immediately upon registration; it is applied only after `apply_patches` is called.
2. **Dummy function mechanism**: When `orig_func_name` does not exist and `create_dummy=True`, a dummy function is automatically created to ensure that imports succeed normally but raise an error when called.
3. **Replacement mode**: When `orig_func_name` is not `None`, it is replaced with `new_func`.
4. **Decorator mode**: If the `new_func` function name ends with `wrapper` or `decorator`, it is applied as a decorator layered onto the original function.
5. **Override policy**: When `force_patch=False`, repeatedly replacing the same function is prohibited (but repeated decoration is allowed); when `force_patch=True`, forced override is applied.

- **2. `apply_patches`**

```python
def apply_patches()
```

**Purpose**: Enable all registered patches in batch.

**Invocation timing**: Typically invoked uniformly after all features have been initialized.
