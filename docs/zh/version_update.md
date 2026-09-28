# MindSpeed core_r0.18.0版本升级兼容性变更

## get_batch 重复构造配置导致 Host 开销增加

### 问题现象

升级到 Megatron-LM core_r0.18.0 后，如果训练性能分析显示 `get_batch` 在 Host 侧耗时较高，且 `core_transformer_config_from_args` 被反复调用，可以考虑开启 `cache-get-batch-config`。这类开销随 microbatch 的重复执行累积，可能影响训练吞吐。

### 问题根因

训练入口的 `get_batch` 会调用 `core_transformer_config_from_args(args)`，构造 `TransformerConfig`，用于判断流水线布局、MTP 层所在阶段等信息。即使模型配置没有变化，每次调用仍会重复构造和校验配置，增加 Host 侧开销；部分流水线中间阶段即使无需读取数据，也会先执行这一步。

### 解决方案

在已接入 MindSpeed 的训练启动参数中添加：

```bash
--cache-get-batch-config
```

`--cache-get-batch-config` 默认关闭，属于 optimization level 2 特性。当前 `--optimization-level` 默认值为 `2`；若脚本将其设置为 `0` 或 `1`，需要同时调整为 `2`。关闭配置缓存时，移除 `--cache-get-batch-config` 即可。

该特性的核心作用是复用 `get_batch` 所需的配置，减少重复构造开销。生效机制如下：

1. MindSpeed 在 `pretrain` 入口根据传入的 `forward_step_func` 定位训练入口，并包装其中的 `core_transformer_config_from_args`。该入口需同时提供可调用的 `get_batch` 和 `core_transformer_config_from_args`。
2. 首次由该 `get_batch` 直接发起配置构造时，正常创建并缓存配置；后续相同缓存键的调用直接复用。缓存键包含 args 对象标识、配置类、MLA 开关和异构层配置路径。
3. 其他函数调用配置构造接口时仍执行原逻辑。缓存对象是 `TransformerConfig`，数据 batch 仍按原流程读取。

自定义训练入口若不满足上述条件，或通过其他辅助函数间接构造配置，则不会命中该缓存。使用时应保持相关配置稳定：原地修改同一个 args 对象中未纳入缓存键的字段，不会自动刷新已缓存的配置。

验证时保持其余训练参数一致，对比开启前后的 `get_batch` Host 耗时、稳态单步耗时和吞吐，并检查 loss 是否正常。若主要耗时来自数据读取或数据传输，应继续定位对应瓶颈。更多说明参见 [get_batch 配置缓存](features/cache-get-batch-config.md)。

## RoPE 融合默认开启，旧脚本需要显式使用 no-rope-fusion 关闭

### 问题现象

升级到 core_r0.18.0 适配版本后，即使训练脚本未添加旧参数 `--use-fused-rotary-pos-emb`，RoPE 仍可能走融合计算路径。仅删除旧的开启参数，已不能保证恢复未融合实现，迁移前后的性能或数值对比也可能因此使用不同的计算路径。

对于当前 MindSpeed 的 MLA YaRN 适配路径，未关闭融合时还可能遇到以下报错：

```text
AssertionError: MLA Yarn RoPE does not support RoPE fusion
```

### 问题根因

RoPE 融合的控制统一到 Megatron 的 `apply_rope_fusion` 字段，命令行采用“默认开启、显式关闭”的方式。`--no-rope-fusion` 将该字段设置为 `False`；旧参数 `--use-fused-rotary-pos-emb` 保留为兼容别名，将同一字段设置为 `True`，不再使用独立的旧字段控制融合。

因此，脚本没有传入旧的开启参数，不代表融合已关闭。融合是否适用还取决于模型路径：例如，当前 MindSpeed 的 MLA YaRN 适配实现要求 `apply_rope_fusion=False`。

### 解决方案

**保持默认融合行为：** 在已通过 `import mindspeed.megatron_adaptor` 接入 MindSpeed 的训练脚本中，使用 RoPE 位置编码即可，无需额外添加融合开启参数：

```bash
--position-embedding-type rope
```

**关闭融合：** 若需保持旧脚本的未融合行为、进行融合开关对比，或使用上述要求关闭融合的 MLA YaRN 路径，在已有训练参数中添加：

```bash
--no-rope-fusion
```

该参数关闭的是 RoPE 融合计算，模型仍使用原有的位置编码。恢复默认融合行为时，移除该参数即可。

旧脚本迁移时可按下表调整：

| 原有意图 | 当前推荐配置 |
| --- | --- |
| 开启 RoPE 融合 | 保留 RoPE 位置编码配置，无需添加旧的开启参数 |
| 不使用 RoPE 融合 | 显式添加 `--no-rope-fusion` |
| 保留旧参数 `--use-fused-rotary-pos-emb` | 仍可兼容，但新脚本建议使用默认开启方式 |

不要同时传入 `--use-fused-rotary-pos-emb` 和 `--no-rope-fusion`：两者写入同一个字段，解析结果受命令行先后顺序影响。

该变更的核心作用是统一融合开关，并提供明确的关闭方式。生效机制如下：

1. MindSpeed 的 RoPE 适配属于 optimization level 0 基础补丁，在支持的 level 0/1/2 下均会注册。无需通过 `--optimization-level 2` 开启，也不能通过降低优化等级关闭融合。
2. 参数解析将开关写入 `args.apply_rope_fusion`，随后传入模型配置。Megatron 的 RoPE 分发逻辑据此选择融合或未融合路径；MindSpeed 的普通 RoPE 实现也读取该字段。关闭后，该实现使用旋转、乘法和加法组合计算。
3. 默认开启仍受适用条件约束：非 RoPE 位置编码会在 Megatron 参数校验时关闭融合；开启 `--reset-attention-mask` 的 EOD Reset 场景也会由 MindSpeed 校验关闭融合。

验证时查看最终参数中的 `apply_rope_fusion`，并确认模型配置与之保持一致。固定模型、精度和输入，对比默认配置与增加 `--no-rope-fusion` 后的 loss、梯度及稳态单步耗时，确认实际计算路径和性能变化。更多说明参见 [Rotary Position Embedding 融合优化](features/rotary-embedding.md)。
