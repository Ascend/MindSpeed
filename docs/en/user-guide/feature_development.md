# MindSpeed Core Feature Development

MindSpeed Core adopts a plug-in feature management architecture, implementing a flexible feature registration and patch mechanism through the `MindSpeedFeature` class and the `MindSpeedPatchesManager` class. Developers only need to follow the unified interface specification to quickly add new optimization features without modifying the core framework code.

For MindSpeed development specifications, refer to [MindSpeed Development Specifications](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeed%20%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83.md).

For details about the MindSpeed Core design, refer to [MindSpeed Core Design Document](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeedCore%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md).

## MindSpeedFeature Class

`MindSpeedFeature` is the core base class for MindSpeed feature development. All new features must inherit from this class and override the relevant methods. This class provides complete feature lifecycle management, including key stages such as parameter registration, parameter validation, and patch enabling.

### Feature Creation Process

New features need to create a `<Feature Name>_feature.py` file in an appropriate directory under the `mindspeed/features_manager/` folder, then create a `<Feature Name>Feature` class that inherits from the `mindspeed.features_manager.feature.MindSpeedFeature` class.

### Core Method Details

#### __init__ Initialization

This method is used to set the basic attributes of a feature and serves as the starting point for feature definition.

```python
def __init__(self, feature_name: str, default_patches: bool = False, optimization_level: int = None)
```

**Parameter Description**

- `feature_name` (str): Feature name, all lowercase, separated by `-`, for example `unaligned-linear`
- `default_patches` (bool): Whether to enable patches under this feature by default. **Note: Except for native adaptation features, other features are prohibited from being enabled by default.**
- `optimization_level` (int): The optimization level to which the feature belongs. 0 is basic optimization, 1 is affinity optimization, and 2 is advanced optimization

#### register_args Parameter Registration

This method is used to register command-line parameters related to the feature, and it is called regardless of whether the feature is enabled.

```python
def register_args(self, parser)
```

**Parameter Description**

- `parser` (argparse.ArgumentParser): Parameter parser

**Usage Example**
Create a feature parameter group using `parser.add_argument_group`, and then create specific parameters using `group.add_argument`.

#### pre_validate_args Pre-Parameter Validation

This method is called before `validate_args` and is primarily used to temporarily modify certain parameters to bypass native validation logic.

```python
def pre_validate_args(self, args)
```

**Parameter Description**

- `args` (argparse.Namespace): The parameter object after parsing is complete

#### validate_args Parameter Validation

This method is the core of parameter validation, used to validate parsed parameters against business rules.

```python
def validate_args(self, args)
```

**Parameter Description**

- `args` (argparse.Namespace): The parameter object after parsing is complete

#### post_validate_args Post-Parameter Validation

This method is called after `validate_args` and is used to restore original parameter values after bypassing native validation.

```python
def post_validate_args(self, args)
```

**Parameter Description**

- `args` (argparse.Namespace): The parameter object after parsing is complete

#### register_patches Patch Registration

This method is the core of feature implementation, used to register related functional patches through the patch manager.

```python
def register_patches(self, patch_manager, args)
```

**Parameter Description**

- `patch_manager` (mindspeed.patch_utils.MindSpeedPatchesManager): patch manager instance
- `args` (argparse.Namespace): parameter object after parsing is complete

Call condition: It is called only when the `self.feature_name` parameter exists in `args` or when `default_patches=True`.

#### incompatible_check Incompatible Parameter Validation

Used to detect conflicting relationships between parameters, ensuring that incompatible features are not enabled simultaneously.

```python
def incompatible_check(self, global_args, check_args)
```

**Parameter Description**

- `global_args` (argparse.Namespace): global parameter object
- `check_args` (str): parameter name to be validated, separated by underscores, such as `unaligned_linear`

Validation logic: If `check_args` exists in `global_args` and its implicit boolean value is True, an exception is thrown.

#### dependency_check Dependency Parameter Validation

Used to check whether the dependency conditions required by the feature are met.

```python
def dependency_check(self, global_args, check_args)
```

**Parameter Description**

- `global_args` (argparse.Namespace): global parameter object
- `check_args` (str): name of the dependency parameter to be validated

Validation logic: If `check_args` exists in `global_args` and its implicit boolean value is False, an exception is thrown.

#### add_parser_argument_choices_value Adding Parameter Options

A method for adding parameter choices, used to add new `choices` options to existing parameters. It is recommended to call this method in `register_args` if needed.

```python
def add_parser_argument_choices_value(parser, argument_name, new_choice)
```

**Parameter Description**

- `parser` (argparse.ArgumentParser): argument parser
- `argument_name` (str): name of the target argument
- `new_choice` (str): new option value

---

## MindSpeedPatchesManager Class

`MindSpeedPatchesManager` is the core patch management component of MindSpeed, responsible for unified management of functional replacements and enhancements for all features. This class adopts a delayed activation mechanism, where all registered patches only take effect after `apply_patches` is called.

### Core Method Details

#### register_patch Patch Registration

This method is used to register functions/methods that need to be replaced or enhanced.

```python
def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False)
```

**Parameter Description**

- `orig_func_name` (str): full path of the target function/method, such as `module.class.method`
- `new_func` (Any): function/method to replace with, which can be `None`
- `force_patch` (bool): whether to forcibly overwrite an existing patch
- `create_dummy` (bool): whether to create a dummy function when the target function does not exist to avoid import errors

**Core Mechanism**

1. Delayed activation: Patches do not take effect immediately upon registration; they are only applied after `apply_patches` is called
2. Dummy function mechanism: When `orig_func_name` does not exist and `create_dummy=True`, a dummy function is automatically created to ensure normal import but will raise an error when called
3. Replacement mode: When `orig_func_name` is not `None`, it is replaced with `new_func`
4. Decorator mode: If the `new_func` function name ends with `wrapper` or `decorator`, it is applied as a decorator on top of the original function.
5. Override policy: When `force_patch=False`, duplicate replacement of the same function is prohibited (but duplicate decoration is allowed); when `force_patch=True`, forced override is applied.

#### apply_patches Enabling Patch

This method is used to batch enable all registered patches.

```python
def apply_patches()
```

Call time: Typically called uniformly after all features are initialized, ensuring all patches take effect in the expected order.

### Patch Usage Modes

MindSpeedPatchesManager supports two main patch patterns:

1. Direct Replacement Mode
Suitable for scenarios where the original implementation needs to be completely replaced, directly substituting the old function with a new one.

2. Decorator Enhancement Mode
Suitable for scenarios where functionality needs to be added on top of the original logic, with the new function applied as a decorator overlay while preserving the original logic.

---

## Development Practice Tips

1. Feature naming convention: Use lowercase letters separated by `-` to ensure consistency with the command-line argument style.
2. Default enablement control: Non-natively adapted features must not be enabled by default to avoid affecting the stability of basic functionality.
3. Parameter validation completeness: Fully utilize the three stages of `pre_validate_args`, `validate_args`, and `post_validate_args` to ensure parameter validity.
4. Compatibility check: Use `incompatible_check` and `dependency_check` to ensure the correctness of feature combinations.
5. Patch idempotency: Ensure that patch registrations do not conflict with each other, and use the `force_patch` parameter when necessary.
