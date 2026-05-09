# MindSpeed Core特性开发

MindSpeed Core采用插件化的特性管理架构，通过`MindSpeedFeature`类和`MindSpeedPatchesManager`类实现了灵活的特性注册与patch机制。开发者只需遵循统一的接口规范，即可快速添加新的优化特性，无需修改核心框架代码。

MindSpeed 开发规范请参考 [MindSpeed 开发规范](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeed%20%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83.md)。

MindSpeed Core设计文档请参考 [MindSpeed Core设计文档](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeedCore%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md)。

## MindSpeedFeature类

`MindSpeedFeature`是MindSpeed特性开发的核心基类，所有新特性都需要继承此类并覆写相关方法。该类提供了完整的特性生命周期管理，包括参数注册、参数校验、patch使能等关键环节。

### 特性创建流程

新特性需要在 `mindspeed/features_manager/` 文件夹下选择合适的目录创建 `<特性名称>_feature.py` 文件，然后创建 `<特性名称>Feature` 类并继承 `mindspeed.features_manager.feature.MindSpeedFeature` 类。

### 核心方法详解

#### __init__ 初始化方法

该方法用于设置特性的基本属性，是特性定义的起点。

```python
def __init__(self, feature_name: str, default_patches: bool = False, optimization_level: int = None)
```

**参数说明：**

- **feature_name** (str): 特性名称，全部小写，以`-`进行间隔，例如 `unaligned-linear`
- **default_patches** (bool): 是否默认使能该特性下的patch。**注意：除原生适配特性外，其他特性禁止默认使能**
- **optimization_level** (int): 特性所属的优化等级，0为基础优化，1为亲和优化，2为高阶优化

#### register_args 参数注册

该方法用于注册特性相关的命令行参数，无论特性是否被使能都会被调用。

```python
def register_args(self, parser)
```

**参数说明：**

- **parser** (argparse.ArgumentParser): 参数解析器

**使用示例：**
通过 `parser.add_argument_group` 创建特性参数组，再通过 `group.add_argument` 创建具体参数。

#### pre_validate_args 参数校验前置

该方法在 `validate_args` 之前被调用，主要用于临时修改某些参数以绕过原生校验逻辑。

```python
def pre_validate_args(self, args)
```

**参数说明：**

- **args** (argparse.Namespace): 解析完成后的参数对象

#### validate_args 参数校验

该方法是参数校验的核心，用于对解析完成后的参数进行业务规则校验。

```python
def validate_args(self, args)
```

**参数说明：**

- **args** (argparse.Namespace): 解析完成后的参数对象

#### post_validate_args 参数校验后置

该方法在 `validate_args` 之后被调用，用于在绕过原生校验后恢复原有参数值。

```python
def post_validate_args(self, args)
```

**参数说明：**

- **args** (argparse.Namespace): 解析完成后的参数对象

#### register_patches patch注册

该方法是特性实现的核心，用于通过patch管理器注册相关的功能patch。

```python
def register_patches(self, patch_manager, args)
```

**参数说明：**

- **patch_manager** (mindspeed.patch_utils.MindSpeedPatchesManager): patch管理器实例
- **args** (argparse.Namespace): 解析完成后的参数对象

**调用条件：** 只有当 `args` 中存在 `self.feature_name` 参数或 `default_patches=True` 时才会被调用。

#### incompatible_check 不兼容参数校验

用于检测参数之间的冲突关系，确保不兼容的特性不会同时被使能。

```python
def incompatible_check(self, global_args, check_args)
```

**参数说明：**

- **global_args** (argparse.Namespace): 全局参数对象
- **check_args** (str): 需要校验的参数名，以下划线间隔，如 `unaligned_linear`

**校验逻辑：** 如果 `global_args` 中存在 `check_args` 且其隐式布尔值为True，则抛出异常。

#### dependency_check 依赖参数校验

用于检测特性所需的依赖条件是否满足。

```python
def dependency_check(self, global_args, check_args)
```

**参数说明：**

- **global_args** (argparse.Namespace): 全局参数对象
- **check_args** (str): 需要校验的依赖参数名

**校验逻辑：** 如果 `global_args` 中存在 `check_args` 且其隐式布尔值为False，则抛出异常。

#### add_parser_argument_choices_value 扩展参数选项

增加参数选择方法，用于在已有参数基础上增加新的`choices` 选项。如有需要建议在 `register_args` 中调用此方法。

```python
def add_parser_argument_choices_value(parser, argument_name, new_choice)
```

**参数说明：**

- **parser** (argparse.ArgumentParser): 参数解析器
- **argument_name** (str): 目标参数名称
- **new_choice** (str): 新增的选项值

---

## MindSpeedPatchesManager类

`MindSpeedPatchesManager`是MindSpeed的patch管理核心，负责统一管理所有特性的功能替换和增强。该类采用延迟生效机制，所有注册的patch只有在调用`apply_patches`后才会真正生效。

### 核心方法详解

#### register_patch patch注册

该方法用于注册需要替换或增强的函数/方法。

```python
def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False)
```

**参数说明：**

- **orig_func_name** (str): 目标函数/方法的完整路径，如 `module.class.method`
- **new_func** (Any): 替换的新函数/方法，可为None
- **force_patch** (bool): 是否强制覆盖已存在的patch
- **create_dummy** (bool): 是否在目标函数不存在时创建假函数以避免导入错误

**核心机制：**

1. **延迟生效**：注册时patch不会立即生效，需调用`apply_patches`后才会应用
2. **Dummy函数机制**：当`orig_func_name`不存在且`create_dummy=True`时，会自动创建一个dummy函数，保证导入正常但调用时会报错
3. **替换模式**：当`orig_func_name`不为None时，将其替换为`new_func`
4. **装饰器模式**：如果`new_func`函数名以`wrapper`或`decorator`结尾，则作为装饰器叠加到原函数上
5. **覆盖策略**：`force_patch=False`时禁止重复替换同一函数（但允许重复装饰），`force_patch=True`时强制覆盖

#### apply_patches 使能patch

该方法用于批量使能所有已注册的patch。

```python
def apply_patches()
```

**调用时机：** 通常在所有特性初始化完成后统一调用，确保所有patch按预期顺序生效。

### patch使用模式

MindSpeedPatchesManager支持两种主要的patch模式：

**1. 直接替换模式**
适用于需要完全替换原有实现的场景，直接用新函数替代旧函数。

**2. 装饰器增强模式**
适用于需要在原有逻辑基础上增加功能的场景，新函数作为装饰器叠加，保留原有逻辑。

---

## 开发实践建议

1. **特性命名规范**：使用小写字母，以`-`分隔，确保与命令行参数风格一致
2. **默认使能控制**：非原生适配特性禁止默认使能，避免影响基础功能稳定性
3. **参数校验完整性**：充分利用`pre_validate_args`、`validate_args`、`post_validate_args`三个阶段确保参数合法性
4. **兼容性检查**：使用`incompatible_check`和`dependency_check`确保特性组合的正确性
5. **patch幂等性**：确保patch注册不会相互冲突，必要时使用`force_patch`参数
