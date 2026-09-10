# MindSpeed Core特性开发指南

本文档介绍如何开发一个新的MindSpeed特性，包含API参考和开发教程。

## 概述

MindSpeed Core采用**插件化**的特性管理架构，核心组件：

| 组件 | 作用 |
| ------ | ------ |
| `MindSpeedFeature` | 特性基类，定义生命周期钩子 |
| `MindSpeedPatchesManager` | 统一管理patch注册与生效 |

开发者只需继承`MindSpeedFeature`，覆写相关方法，即可添加新特性，无需修改核心框架。

## 开发流程

以AsyncLogAllreduceFeature为例

1. 创建特性类

    ```python
    from argparse import ArgumentParser, Namespace
    from mindspeed.features_manager.feature import MindSpeedFeature
    from mindspeed.patch_utils import MindSpeedPatchesManager

    class AsyncLogAllreduceFeature(MindSpeedFeature):
        """异步日志AllReduce特性"""

        def __init__(self, feature_name: str = "async-log-allreduce", optimization_level: int = 2):
            super().__init__(feature_name, optimization_level)
    ```

    >[!NOTE]
    >
    >- `feature_name`使用`async-log-allreduce`，与命令行参数`--async-log-allreduce`对应。
    >- `optimization_level=2`表示这是高阶优化特性。

2. 注册命令行参数

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
    >- 使用`add_argument_group`组织相关参数。
    >- `action='store_true'`表示这是一个开关型参数。

3. 注册patch

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

    >[!NOTE]
    >
    >- 在`register_patches`函数内部进行 import，而非文件顶部导入。这样可以避免循环依赖：如果在文件顶部导入`mindspeed.core.data_parallel.async_log_allreduce`，而该模块又间接导入 `features_manager`，会导致初始化失败。只有当`is_need_apply(args)`返回True时，才会执行到这段代码。
    >- 注册patch将`megatron.training.training.train_step`替换为自定义实现。

## 开发实践指南

### 开发实践建议

- 特性命名规范：使用小写字母，以`-`分隔，确保与命令行参数风格一致。
- 默认启用控制：非原生适配特性禁止默认启用，避免影响基础功能稳定性。
- 参数校验完整性：充分利用`pre_validate_args`、`validate_args`、`post_validate_args`三个阶段确保参数合法性。
- 兼容性检查：使用`incompatible_check`和`dependency_check`确保特性组合的正确性。
- patch幂等性：确保patch注册不会相互冲突，必要时使用`force_patch`参数。

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

- 使用pre_validate_args/post_validate_args的场景。

    当需要绕过第三方库的参数校验时使用。例如Megatron的校验太严格，但需要在特定场景下放宽限制。

- 装饰器模式和替换模式如何选择？

    | 场景 | 推荐模式 |
    | ------ | ------ |
    | 需要保留原函数逻辑，增加额外功能 | 装饰器模式（函数名以`wrapper`结尾） |
    | 需要完全重写实现 | 直接替换模式 |

- 排查patch是否生效

    1. 检查`is_need_apply(args)`是否返回True。
    2. 确认`register_patches`被调用。
    3. 确认`apply_patches()`在正确时机被调用。
    4. 检查patch目标路径是否正确。

### API参考

详见[API参考](./API.md)。
