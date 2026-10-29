# MindSpeed Core Feature Development

This document describes how to develop a new MindSpeed feature, including API references and development tutorials.

## Overview

MindSpeed Core adopts a **plugin-based** feature management architecture, with the following core components:

| Component | Purpose |
| ------ | ------ |
| `MindSpeedFeature` | Inherits from the MegatronAdaptor feature base class and defines lifecycle hooks |
| `MindSpeedPatchesManager` | Centrally manages patch registration and activation |

Developers only need to inherit from `MindSpeedFeature` and override the relevant methods to add a new feature, without modifying the core framework.

## Development Process

Taking `AsyncLogAllreduceFeature` as an example:

1. Create the feature class.

    ```python
    from argparse import ArgumentParser, Namespace
    from mindspeed.features_manager.feature import MindSpeedFeature
    from mindspeed.patch_utils import MindSpeedPatchesManager

    class AsyncLogAllreduceFeature(MindSpeedFeature):
        """Asynchronous logging AllReduce feature"""

        def __init__(self, feature_name: str = "async-log-allreduce", optimization_level: int = 2):
            super().__init__(feature_name, optimization_level)
    ```

    >[!NOTE]
    >
    >- `feature_name` uses `async-log-allreduce`, corresponding to the command-line argument `--async-log-allreduce`.
    >- `optimization_level=2` indicates that this is a high-level optimization feature.

2. Register command-line arguments.

    ```python
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='overlap_p2p_comm_or_async_log_allreduce_')
        group.add_argument(
            '--async-log-allreduce',
            action='store_true',
            help='Transform the AllReduce operation used for transmitting log information into an asynchronous operation to reduce communication overhead.')
    ```

    >[!NOTE]
    >
    >- Use `add_argument_group` to organize related arguments.
    >- `action='store_true'` indicates that this is a switch-type argument.

3. Register patches.

    ```python
    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # // Defer the import: import the module inside the function to avoid circular dependencies.
        from mindspeed.core.data_parallel.async_log_allreduce import train_step
        patch_manager.register_patch('megatron.training.training.train_step', train_step)
    ```

    >[!NOTE]
    >
    >- Perform the import inside the `register_patches` function rather than at the top of the file. This avoids circular dependencies: if `mindspeed.core.data_parallel.async_log_allreduce` were imported at the top of the file, and that module in turn indirectly imports `features_manager`, initialization would fail. This code is executed only when `is_need_apply(args)` returns `True`.
    >- Registering the patch replaces `megatron.training.training.train_step` with the custom implementation.

4. Add to the unified feature list

    Add the feature instance to the corresponding `add_*_features` function in `mindspeed/features_manager/__init__.py`, which is aggregated by `create_features_list()`. `set_default_features_list()` appends it to MA's unified list via `FeaturesManager.add_feature(...)`.

## Development Practice

### Suggestions

- Feature naming convention: Use lowercase letters separated by `-` to stay consistent with the command-line argument style.
- Default enablement control: Non-native adapted features must not be enabled by default, to avoid affecting the stability of basic functionality.
- Complete argument validation: Make full use of `pre_validate_args`, `validate_args`, and `post_validate_args` to ensure argument validity.
- Compatibility check: Use `incompatible_check` and `dependency_check` to ensure the correctness of feature combinations.
- Patch idempotency: Ensure that patch registration does not conflict with each other, and use the `force_patch` parameter when necessary. `MindSpeedFeaturesManager.remove_patches` only removes MindSpeed's own patch layer.
- Conventions for completely replacing `train_step`: The direct replacement pattern overwrites the entire wrapper chain already installed by MegatronAdaptor. If the replacement target is `megatron.training.training.train_step`, you must explicitly call MegatronAdaptor's stable interface `profile_step()` once before each training step returns normally (`from megatron_adaptor.patches.megatron.profile import profile_step`); otherwise, the NPU Profiling collection window will no longer advance. This call returns with zero overhead when profiling is not enabled and can be invoked unconditionally. Reference implementations: `mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks.train_step`, `mindspeed.core.data_parallel.async_log_allreduce.train_step`.
- Argument retrieval: Use `mindspeed.args_utils.get_full_args` to retrieve the full arguments.

### Checklist for Creating a New Feature

```text
Feature creation checklist
├── Basic Setup
│   ├── [ ] Create directory under mindspeed/features_manager/
│   ├── [ ] Create <feature_name>_feature.py file
│   └── [ ] Inherit from MindSpeedFeature base class
├── Parameter Registration
│   ├── [ ] Use add_argument_group to organize parameters
│   ├── [ ] Use hyphens (`-`) for parameter names
│   └── [ ] Provide clear help documentation
├── Parameter Validation
│   ├── [ ] Implement validate_args method as needed
│   └── [ ] Use incompatible_check/dependency_check for compatibility checks
├── Patch Registration
│   ├── [ ] Use lazy imports to avoid circular dependencies
│   └── [ ] Choose appropriate patch mode (replace/decorator)
└── Testing and Validation
    ├── [ ] Test parameter parsing
    └── [ ] Test functional correctness
```

### FAQs

- Scenarios for using `pre_validate_args/post_validate_args`

    `pre_validate_args` is used to normalize arguments and make aliases compatible before native validation; `post_validate_args` is used for checks that depend on the final argument values. For example, CP converts `context_parallel_algo` to `cp_comm_type` in the pre-stage, and MoE overlap checks the noop layer position in the post-stage.

- How to choose between the decorator pattern and the replacement pattern

    | Scenario | Recommended Pattern |
    | ------ | ------ |
    | Need to retain the original function logic while adding extra functionality | Decorator pattern (function name ends with `wrapper`) |
    | Need to completely rewrite the implementation | Direct replacement pattern |

- Check whether the patch takes effect

    1. Check whether `is_need_apply(args)` returns `True`.
    2. Confirm that `register_patches` is called.
    3. Confirm that `apply_patches()` is called at the correct time.
    4. Check whether the patch target path is correct.

### API Reference

For details, see the [API reference](./API.md).
