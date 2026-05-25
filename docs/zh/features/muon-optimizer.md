# Muon 优化器

## 背景与挑战

Muon 是一种面向大模型训练的矩阵参数优化器。与 Adam 等逐元素优化器不同，Muon 主要作用于二维权重矩阵，会先维护类似 SGD 的动量更新，再对动量矩阵执行 Newton-Schulz 正交化，使更新方向具备更好的矩阵结构约束。对于 Transformer 中的大量线性层参数，Muon 可以作为矩阵参数的主优化器使用。

大模型训练中，参数并不都适合使用 Muon。二维矩阵权重适合进入 Muon，embedding、输出层、bias、norm 等参数通常仍使用 Adam 等标量优化器。因此，使用 Muon 时需要同时管理两类优化器，并保证它们在学习率、权重衰减、梯度裁剪、状态保存和参数同步等流程中协同工作。

开启 TP、MoE、ZeRO 等并行能力后，Muon 还会遇到额外挑战：

1. **TP 参数重复过滤**：张量并行下，部分参数在不同 TP rank 上是重复视图。梯度范数、梯度裁剪和 zero count 统计时必须过滤重复参数，否则统计结果会被重复计算。
2. **TP 分片矩阵正交化**：Muon 的 Newton-Schulz 正交化作用于矩阵更新。矩阵被 TP 切分后，需要知道参数的切分维度和通信 group，才能按正确的全局矩阵形状计算缩放因子，并在需要时进行跨 TP 通信。
3. **QKV 融合权重处理**：部分模型会将 Q、K、V 权重融合成一个线性层。Muon 直接对融合后的大矩阵做正交化可能不符合预期，需要按 Q/K/V 结构拆分后分别处理。
4. **MoE/expert parallel 通信组选择**：expert 参数可能使用 expert tensor parallel group，而 dense 参数使用普通 TP group。Muon 必须区分 dense/expert 参数，避免梯度统计和 TP duplicate filter 使用错误 group。
5. **ZeRO 参数分片与同步**：启用 `--use-distributed-optimizer` 后，Muon 走 layer-wise distributed optimizer 路径。每个 DP rank 只负责更新部分完整参数，更新后需要将参数同步回所有 rank。
6. **bf16 master param 属性继承**：bf16 训练中 optimizer 会创建 fp32 master param。Muon 依赖的 `expert_tp`、`is_qkv` 等自定义属性需要从模型参数同步到 master param，否则 TP group 选择和 QKV split 会失效。

## 解决方案

MindSpeed 通过 optimizer feature patch 将 Muon 集成到 Megatron optimizer 构建流程中。开启 `--optimizer muon` 后，MindSpeed 会接管 optimizer 构建入口，完成参数分类、参数打标、Muon 构建、标量优化器构建和 wrapper 封装。

整体方案如下：

1. 对模型参数进行扫描，根据参数形状和属性决定其使用 Muon 还是标量优化器。
2. 对 QKV 权重标记 `is_qkv`，对 expert tensor parallel 参数标记 `expert_tp`。
3. 为二维矩阵参数构建 `TensorParallelMuon`。
4. 为非矩阵参数构建标量优化器，默认使用 Adam，也可通过 `--muon-scalar-optimizer` 配置。
5. 按 dense/expert 参数拆分 optimizer wrapper，使每个 wrapper 携带正确的 `grad_stats_parallel_group` 和 `tp_group`。
6. 在 bf16 场景下扩展 tensor-parallel 属性复制逻辑，将 Muon 需要的自定义属性同步到 fp32 master param。
7. 在 layer-wise distributed optimizer 路径中，按完整参数分配 owner rank，step 后同步 all-gather 参数。

## 实现原理

Muon 只处理满足条件的二维矩阵参数。对于每个进入 Muon 的参数，优化器会维护 `momentum_buffer`，每次 step 时先用当前梯度更新动量，再根据是否开启 Nesterov 得到本轮更新矩阵。

随后，Muon 对更新矩阵执行 Newton-Schulz 迭代，得到近似正交化后的更新方向。MindSpeed 本地实现支持不同的 Newton-Schulz coefficient、迭代步数和 scale mode，并支持在 TP 场景下根据参数切分维度计算全局矩阵形状。

对于非矩阵参数，MindSpeed 不使用 Muon，而是将其放入标量优化器参数组。这样可以避免对 embedding、bias、norm、输出层等参数使用不合适的矩阵正交化逻辑。

## TP 兼容适配

开启 TP 后，MindSpeed 主要从以下几个方面适配 Muon：

1. **TP group 感知**

   `TensorParallelMuon` 会根据参数属性选择普通 TP group 或 expert TP group。dense 参数使用普通 TP group，expert 参数使用 expert TP group。

2. **TP duplicate filter**

   MindSpeed patch 了 `param_is_not_tensor_parallel_duplicate`，使其支持显式传入 `tp_group`。梯度范数、梯度裁剪和 zero count 统计时，会基于对应 wrapper 的 `tp_group` 过滤重复参数。

3. **TP 分片矩阵处理**

   Muon 会读取参数的 `partition_dim`。当参数按某个维度被 TP 切分时，计算缩放因子前会将该维度乘以 TP group size，恢复全局矩阵形状的语义。

4. **QKV split**

   对被标记为 `is_qkv` 的参数，`TensorParallelMuon` 会根据模型配置中的 Q/K/V 维度拆分融合权重，分别对 Q、K、V 子矩阵执行正交化，再拼回原始形状。

5. **bf16 master param 属性复制**

   bf16 训练会创建 fp32 master param。MindSpeed 会扩展 tensor-parallel 属性复制流程，将 `expert_tp` 和 `is_qkv` 从原始模型参数复制到 master param，确保 optimizer step 时仍能正确识别 expert 参数和 QKV 参数。

## ZeRO 兼容适配

当 Muon 与 `--use-distributed-optimizer` 同时使用时，MindSpeed 使用 layer-wise distributed optimizer 路径适配 ZeRO 类参数分片场景。

该路径不会按 Megatron 原生 distributed optimizer 的连续参数 buffer 切分方式工作，而是按完整参数分配 owner rank：

1. 将所有参数按参数规模排序。
2. 在 DP/EP DP rank 间以 ping-pong 方式分配 owner rank，尽量平衡各 rank 的参数量。
3. 每个 rank 的本地 optimizer 只保留自己负责更新的参数。
4. 梯度仍由 DDP 完成同步，每个 rank 可以获得完整参数对应的梯度。
5. optimizer step 后，将各 rank 更新后的参数 all-gather 回所有 rank。

不同 rank 拥有的参数数量可能不同，flat tensor 长度也不同。MindSpeed 直接使用 `torch.distributed.all_gather` 收集不等长 flat tensor，随后按各 rank 的参数列表 unflatten 并复制回模型参数。

在 torch checkpoint 场景下，layer-wise distributed optimizer 会按 DP rank 额外保存 optimizer state 文件，并在恢复时由各 rank 加载自己对应的 optimizer state，避免只使用主 checkpoint 中单个 DP rank 的 optimizer 状态。

## 使用场景

Muon 适用于希望对 Transformer 线性层等二维矩阵参数使用矩阵正交化更新，同时保留 Adam 等标量优化器处理非矩阵参数的大模型训练场景。

推荐使用场景包括：

1. bf16 或 fp32 大模型训练。
2. 使用 TP 训练，并希望 Muon 正确处理 TP 分片矩阵。
3. 使用 MoE/expert parallel 训练，并需要区分 dense/expert 的通信 group。
4. 使用 `--use-distributed-optimizer`，希望通过 layer-wise distributed optimizer 降低每个 DP rank 的 optimizer 参数更新压力。

## 使用方法

开启 Muon：

```bash
--optimizer muon \
--bf16
```

开启 Muon 的 layer-wise distributed optimizer 路径：

```bash
--optimizer muon \
--bf16 \
--use-distributed-optimizer
```

在 layer-wise distributed optimizer 路径中开启参数 gather 与 forward 重叠：

```bash
--optimizer muon \
--bf16 \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

常用可选参数如下：

```bash
--muon-momentum 0.95
--muon-nesterov
--muon-scale-mode spectral
--muon-fp32-matmul-prec medium
--muon-coefficient-type quintic
--muon-num-ns-steps 5
--muon-tp-mode blockwise
--muon-extra-scale-factor 1.0
--muon-scalar-optimizer adam
--apply-wd-to-qk-layernorm
```

参数说明如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--optimizer` | `adam` | 指定优化器类型。设置为 `muon` 时启用 Muon 优化器。 |
| `--use-distributed-optimizer` | 关闭 | 与 Muon 同时使用时，启用 layer-wise distributed optimizer 路径。 |
| `--overlap-grad-reduce` | 关闭 | 开启梯度 reduce 与反向计算重叠；使用 `--overlap-param-gather` 时需要同时开启。 |
| `--overlap-param-gather` | 关闭 | 在 layer-wise distributed optimizer 路径中，将参数同步延迟到 DDP forward pre-hook，通过 bucket 触发异步 gather。 |
| `--overlap-param-gather-with-optimizer-step` | 关闭 | 当前 Muon 路径暂不支持该开关。 |
| `--muon-momentum` | `0.95` | Muon 内部动量系数，用于更新 `momentum_buffer`。 |
| `--muon-nesterov` | 关闭 | 是否在 Muon 内部动量更新中使用 Nesterov 形式。 |
| `--muon-scale-mode` | `spectral` | Muon 更新矩阵的缩放方式，支持 `spectral`、`unit_rms_norm`、`shape_scaling`。 |
| `--muon-fp32-matmul-prec` | `medium` | Newton-Schulz 迭代中 fp32 matmul 的精度配置。 |
| `--muon-coefficient-type` | `quintic` | Newton-Schulz 迭代使用的 coefficient 类型。 |
| `--muon-num-ns-steps` | `5` | Newton-Schulz 迭代步数。 |
| `--muon-tp-mode` | `blockwise` | TP 场景下 Newton-Schulz 的计算方式，支持 `blockwise`、`duplicated`、`distributed`。 |
| `--muon-extra-scale-factor` | `1.0` | Muon 更新额外乘上的缩放系数。 |
| `--muon-scalar-optimizer` | `adam` | 非矩阵参数使用的标量优化器，支持 `adam`、`lion`。 |
| `--muon-no-split-qkv` | 默认开启 QKV split | 关闭 QKV 融合权重拆分处理。 |
| `--apply-wd-to-qk-layernorm` | 关闭 | 对 Q/K layernorm 保留 weight decay；开启后其他一维参数和 bias 仍默认跳过 weight decay。 |

如需关闭 QKV split，可配置：

```bash
--muon-no-split-qkv
```

## 使用效果

启用后，MindSpeed 会为不同类型参数选择合适的优化器路径：

1. 二维矩阵参数使用 Muon 的 Newton-Schulz 正交化更新。
2. embedding、输出层、bias、norm 等非矩阵参数使用标量优化器，默认使用 Adam。
3. TP 场景下，Muon 会根据参数切分信息和 TP group 处理矩阵正交化、梯度统计和重复参数过滤。
4. MoE/expert parallel 场景下，dense/expert 参数会分别携带对应的通信 group。
5. ZeRO/layer-wise distributed optimizer 场景下，每个 DP rank 只更新自己负责的参数，并在 step 后同步参数；torch checkpoint 会按 DP rank 保存和恢复对应的 optimizer state。
6. 开启 `--overlap-param-gather` 后，参数同步会挂到 DDP bucket 上，由 forward pre-hook 触发异步 gather，以减少参数同步对训练 step 的阻塞。

## 注意事项

1. 当前 Muon 路径不支持 fp16，请使用 bf16 或 fp32。
2. 当前 Muon 路径不支持与 MindSpeed 自定义 FSDP 或 torch FSDP2 同时使用。
3. Muon 与 `--use-distributed-optimizer` 同时使用时，会走 MindSpeed 的 layer-wise distributed optimizer 路径。
4. Muon 路径支持 `--overlap-param-gather`，但需要同时开启 `--use-distributed-optimizer` 和 `--overlap-grad-reduce`。
5. Muon 路径暂不支持 `--overlap-param-gather-with-optimizer-step`。
6. layer-wise all-gather 依赖 `torch.distributed.all_gather` 支持不等长 tensor，仍建议在目标集群上验证 DP>1、EP>1、MoE 和 checkpoint 恢复场景。
7. 默认标量优化器为 Adam；如果配置 `--muon-scalar-optimizer lion`，需要额外提供 Lion 实现。
