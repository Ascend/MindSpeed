# MindSpeed Core特性开发指南

本文档介绍如何开发一个新的MindSpeed特性，包含API参考和开发教程。

---

## 概述

MindSpeed Core采用**插件化**的特性管理架构，核心组件：

| 组件 | 作用 |
| ------ | ------ |
| `MindSpeedFeature` | 特性基类，定义生命周期钩子 |
| `MindSpeedPatchesManager` | 统一管理patch注册与生效 |

开发者只需继承 `MindSpeedFeature`，覆写相关方法，即可添加新特性，无需修改核心框架。

---

## 开发流程：以AsyncLogAllreduceFeature为例

### 步骤1：创建特性类

```python
from argparse import ArgumentParser, Namespace
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

class AsyncLogAllreduceFeature(MindSpeedFeature):
    """异步日志AllReduce特性"""

    def __init__(self, feature_name: str = "async-log-allreduce", optimization_level: int = 2):
        super().__init__(feature_name, optimization_level)
```

**关键点**：

- `feature_name` 使用 `async-log-allreduce`，与命令行参数 `--async-log-allreduce` 对应
- `optimization_level=2` 表示这是高阶优化特性

### 步骤2：注册命令行参数

```python
def register_args(self, parser: ArgumentParser):
    group = parser.add_argument_group(title='overlap_p2p_comm_or_async_log_allreduce_')
    group.add_argument(
        '--async-log-allreduce',
        action='store_true',
        help='Transform the AllReduce operation used for transmitting log information into an asynchronous operation to reduce communication overhead.')
```

**关键点**：

- 使用 `add_argument_group` 组织相关参数
- `action='store_true'` 表示这是一个开关型参数
- 提供清晰的帮助文档说明用途

### 步骤3：注册patch

```python
def register_patches(
    self,
    patch_manager: MindSpeedPatchesManager,
    args: Namespace,
):
    # 延迟导入：在函数内部导入模块，避免循环依赖
    from mindspeed.core.data_parallel.async_log_allreduce import train_step
    patch_manager.register_patch('megatron.training.training.train_step', train_step)
```

**关键点**：

- 在 `register_patches` 函数内部进行 import，而非文件顶部导入。这样可以避免循环依赖：如果在文件顶部导入 `mindspeed.core.data_parallel.async_log_allreduce`，而该模块又间接导入 `features_manager`，会导致初始化失败。 只有当 `is_need_apply(args)` 返回True时，才会执行到这段代码
- 注册 patch 将 `megatron.training.training.train_step` 替换为自定义实现

---

## API参考

### MindSpeedFeature基类

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

---

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

---

## 开发实践指南

### 开发实践建议

1. **特性命名规范**：使用小写字母，以`-`分隔，确保与命令行参数风格一致
2. **默认使能控制**：非原生适配特性禁止默认使能，避免影响基础功能稳定性
3. **参数校验完整性**：充分利用`pre_validate_args`、`validate_args`、`post_validate_args`三个阶段确保参数合法性
4. **兼容性检查**：使用`incompatible_check`和`dependency_check`确保特性组合的正确性
5. **patch幂等性**：确保patch注册不会相互冲突，必要时使用`force_patch`参数

### 创建新特性的Checklist

```text
特性创建 Checklist
├── 基础设置
│   ├── [ ] 在 mindspeed/features_manager/ 下创建目录
│   ├── [ ] 创建 <特性名>_feature.py 文件
│   └── [ ] 继承 MindSpeedFeature 基类
├── 参数注册
│   ├── [ ] 使用 add_argument_group 组织参数
│   ├── [ ] 参数名使用 `-` 分隔
│   └── [ ] 提供清晰的帮助文档
├── 参数校验
│   ├── [ ] 根据需要实现 validate_args 方法
│   └── [ ] 使用 incompatible_check/dependency_check 检查兼容性
├── Patch注册
│   ├── [ ] 使用延迟导入避免循环依赖
│   └── [ ] 选择合适的patch模式（替换/装饰器）
└── 测试验证
    ├── [ ] 测试参数解析
    └── [ ] 测试功能正确性
```

### 常见问题

**Q: 何时使用 pre_validate_args/post_validate_args？**

当需要绕过第三方库的参数校验时使用。例如Megatron的校验太严格，但你需要在特定场景下放宽限制。

**Q: 装饰器模式和替换模式如何选择？**

| 场景 | 推荐模式 |
| ------ | ------ |
| 需要保留原函数逻辑，增加额外功能 | 装饰器模式（函数名以`wrapper`结尾） |
| 需要完全重写实现 | 直接替换模式 |

**Q: patch不生效怎么排查？**

1. 检查 `is_need_apply(args)` 是否返回True
2. 确认 `register_patches` 被调用
3. 确认 `apply_patches()` 在正确时机被调用
4. 检查patch目标路径是否正确

---

## 相关文档

- [MindSpeed 开发规范](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeed%20%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83.md)
- [MindSpeed Core设计文档](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeedCore%E8%AE%BE%E8%AE%A4%E6%96%87%E6%A1%A3.md)
