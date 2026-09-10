# API参考

## MindSpeedFeature基类

- **1. `__init__`**

```python
def __init__(self, feature_name: str, optimization_level: int = 2)
```

**参数说明：**

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| feature_name | str | 特性名称，全小写，以`-`分隔，如 `async-log-allreduce` |
| optimization_level | int | 优化等级：0=基础优化，1=亲和优化，2=高阶优化 |

**默认行为：**

- `optimization_level == 0` 时，`default_patches` 自动设为 `True`（默认使能）
- feature_name 会自动转换为下划线格式存储

- **2. `register_args`**

```python
def register_args(self, parser: ArgumentParser)
```

**作用**：注册特性相关的命令行参数，无论特性是否被使能都会被调用。

**参数说明：**

- **parser**：argparse.ArgumentParser实例，参数解析器

**使用示例：**
通过 `parser.add_argument_group` 创建特性参数组，再通过 `group.add_argument` 创建具体参数。

```python
def register_args(self, parser):
    group = parser.add_argument_group(title=self.feature_name)
    group.add_argument('--my-feature', action='store_true', help='...')
```

- **3. `pre_validate_args`**

```python
def pre_validate_args(self, args: Namespace)
```

**作用**：在Megatron参数校验前临时修改某些参数，绕过原生校验逻辑。

**典型场景**：

```python
def pre_validate_args(self, args):
    self._saved_cp_size = args.context_parallel_size
    args.context_parallel_size = 1  # 临时修改绕过校验
```

- **4. `validate_args`**

```python
def validate_args(self, args: Namespace)
```

**作用**：参数校验核心方法，用于对解析完成后的参数进行业务规则校验。

**使用示例：**

```python
def validate_args(self, args):
    if args.context_parallel_size > 1 and args.seq_length % args.context_parallel_size != 0:
        raise AssertionError("seq_length must be divisible by context_parallel_size")
```

- **5. `post_validate_args`**

```python
def post_validate_args(self, args: Namespace)
```

**作用**：该方法在 `validate_args` 之后被调用，用于在绕过原生校验后恢复原有参数值。

**典型场景**：

```python
def post_validate_args(self, args):
    args.context_parallel_size = self._saved_cp_size  # 恢复原始值
```

- **6. `pre_register_patches`**

```python
def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace)
```

**作用**：在导入Megatron之前注册patch。

- **7. `register_patches`**

```python
def register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace)
```

**作用**：注册特性相关的功能patch。

**触发条件**：只有当 `is_need_apply(args)` 返回True时才会被调用。

**使用示例：**

```python
def register_patches(self, patch_manager, args):
    from mindspeed.core.my_feature import my_new_function
    patch_manager.register_patch('module.path.to.function', my_new_function)
```

- **8. `is_need_apply`**

```python
def is_need_apply(self, args)
```

**作用**：判断特性是否需要被应用。

**判断逻辑**：

```python
return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, None)) \
    or self.default_patches
```

- **9. `incompatible_check`**

```python
def incompatible_check(self, global_args, check_args)
```

**作用**：检测参数之间的冲突关系。

**校验逻辑**：如果 `global_args` 中当前特性和 `check_args` 都为True，则抛出异常。

**使用示例：**

```python
def validate_args(self, args):
    self.incompatible_check(args, 'other_feature')
```

- **10. `dependency_check`**

```python
def dependency_check(self, global_args, check_args)
```

**作用**：检测特性所需的依赖条件是否满足。

**校验逻辑**：如果当前特性为True但 `check_args` 为False，则抛出异常。

**使用示例：**

```python
def validate_args(self, args):
    self.dependency_check(args, 'required_feature')
```

- **11. `add_parser_argument_choices_value`**

```python
@staticmethod
def add_parser_argument_choices_value(parser, argument_name, new_choice)
```

**作用**：为已有参数增加新的choices选项。

**参数说明：**

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| parser | ArgumentParser | 参数解析器 |
| argument_name | str | 目标参数名称（带`--`或不带） |
| new_choice | str | 新增的选项值 |

### MindSpeedPatchesManager类

- **1. `register_patch`**

```python
def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False)
```

**作用**：注册需要替换或增强的函数/方法。

**参数说明：**

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| orig_func_name | str | 目标函数完整路径，如 `module.class.method` |
| new_func | callable | 替换函数，可为None |
| force_patch | bool | 是否强制覆盖已存在的patch |
| create_dummy | bool | 是否在目标函数不存在时创建假函数 |

**核心机制：**

1. **延迟生效**：注册时patch不会立即生效，需调用`apply_patches`后才会应用
2. **Dummy函数机制**：当`orig_func_name`不存在且`create_dummy=True`时，会自动创建一个dummy函数，保证导入正常但调用时会报错
3. **替换模式**：当`orig_func_name`不为None时，将其替换为`new_func`
4. **装饰器模式**：如果`new_func`函数名以`wrapper`或`decorator`结尾，则作为装饰器叠加到原函数上
5. **覆盖策略**：`force_patch=False`时禁止重复替换同一函数（但允许重复装饰），`force_patch=True`时强制覆盖

- **2. `apply_patches`**

```python
def apply_patches()
```

**作用**：批量使能所有已注册的patch。

**调用时机**：通常在所有特性初始化完成后统一调用。
