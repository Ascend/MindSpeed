# HiF8 DTS (Delayed Tensor Scaling) 训练

## 背景与挑战

HiF8（Hybrid Float8）是一种动态的低精度浮点数据格式，具备 1 个符号位以及动态的 Dot 位、指数位和尾数位，最大可表示 2E15。相较于固定格式的 E4M3 / E5M2，HiF8 通过动态调整位宽分配，在更大的数值表示范围和更高的精度之间取得平衡，适合更广泛的模型结构与训练场景。

在 HiF8 低精度训练中，scaling factor 的计算策略直接决定了训练的数值稳定性。其中 Delayed Tensor Scaling（DTS）是一种基于历史 amax 的延迟 scaling 策略：recipe 在训练初期收集 amax，稳态阶段按固定间隔更新 scale factor。由于 DTS 依赖历史统计量，当训练过程中出现 NaN/Inf（如异常批次数据、梯度爆炸等）时，已被污染的 amax 历史会导致后续 step 的 scale factor 失准，进而引发连续的数值溢出，最终导致训练崩溃。

## 解决方法

MindSpeed 为 HiF8 DTS 训练提供了 **Step Recovery** 特性，在框架层对 `train_step` 进行事务化封装：

1. **NaN/Inf 检测**：在 optimizer commit 前检查 loss、grad norm 及 optimizer overflow 标志，判定当前 step 是否有效。
2. **全 rank 失败同步**：通过 `all_reduce(MAX)` 在 world group 上同步失败决策和失败原因，确保所有 rank 一致地重试，避免因 pipeline-only rank 的 optimizer all-reduce 未覆盖而导致的拓扑不一致。
3. **瞬态状态清理**：清除梯度、optimizer overflow 标志及挂起的异步通信，但 **不触碰** 参数更新、optimizer 动量/方差、scheduler 状态、global step 和 consumed-sample 计数器（这些仅在成功 attempt 后提交）。
4. **HiF8 amax 历史重置**：调用 TransformerEngineNPU 的 `FP8GlobalStateManager.reset_fp8_amax_history()` 接口，清除被污染的 amax 历史，使后续 step 从干净的 scale factor 重新开始。
5. **RNG 状态恢复**：恢复 Python、NumPy、Torch CPU/Device 及 Megatron TP RNG tracker 的随机数状态，使 dropout / 随机 mask / MoE routing 在重试时尽可能复现。
6. **批次重放**：缓存原始 attempt 消费的 microbatch，在重试时重放同一批次，保证训练数据流的连续性。支持两种模式：
   - **ExternalReplayAdapter**：框架层缓存迭代器产出的 microbatch，适用于 rerun state machine 未激活的场景。支持 `None` data_iterator（pipeline 场景）。
   - **NativeRerunReplayAdapter**：基于 Megatron `RerunDataIterator.saved_microbatches` 实现重放，适用于 rerun state machine 激活的场景。
7. **重试一次**：清理并重置后，使用恢复的 RNG 和重放的批次重新执行 `train_step`。若重试仍然失败，则输出结构化诊断日志并抛出 `RuntimeError`。

框架层 **不维护也不读取** HiF8 的 warmup / CTS / DTS 内部状态，与 TransformerEngineNPU 的唯一交互点就是 `reset_fp8_amax_history()`。

## 正常 step 的额外开销

启用 Step Recovery 后，每个正常 step（无 NaN/Inf）会额外产生 **两次** 小型 `all_reduce(MAX)` 集合通信：

| 同步点 | 时机 | payload 大小 | 说明 |
|--------|------|-------------|------|
| `after_prepare_grads` | optimizer `prepare_grads()` 返回后 | 2 x int32 | 同步 found_inf 标志和失败原因 |
| `before_step_with_ready_grads` | optimizer `step_with_ready_grads()` 调用前 | 2 x int32 | 同步 loss/grad_norm 的 NaN/Inf 检测结果 |

每次 `all_reduce` 的 payload 仅为 2 个 int32（8 字节），开销可忽略。这两个同步点确保任意 rank 发现异常时，所有 rank 都能在 optimizer commit 前一致地停止并重试。

在非 `hif8_delayed` recipe 场景下，wrapper 直接短路到原始 `train_step`，无任何额外开销。

## 使用场景

- 使用 `--fp8-format hif8 --fp8-recipe hif8_delayed` 进行 HiF8 DTS 低精度训练时，Step Recovery 默认启用。
- 训练过程中偶发的 NaN/Inf（如异常数据、梯度爆炸）需要自动恢复而不中断训练的场景。
- 对训练数据流连续性有要求、不希望跳过批次的场景。

## 使用方法

### 基本启用

HiF8 DTS 训练通过以下参数组合启用，Step Recovery 随之自动激活：

```bash
--transformer-impl transformer_engine
--fp8-format hif8
--fp8-recipe hif8_delayed
```

### 显式禁用 Step Recovery

如需关闭 Step Recovery（例如用于调试或对比基线），添加：

```bash
--no-hif8-step-recovery
```

### HiF8 DTS 配置参数

HiF8 DTS recipe 提供以下可调参数，用于控制 amax 收集与 scale factor 更新行为：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--hif8-input-margin` | int | 11 | 输入/激活 tensor 的保护位数，推荐范围 9-11 |
| `--hif8-weight-margin` | int | 12 | 权重 tensor 的保护位数，推荐范围 11-12 |
| `--hif8-grad-margin` | int | 11 | 梯度 tensor 的保护位数，推荐范围 9-11 |
| `--hif8-amax-collect-interval` | int | 5 | warmup 阶段长度（迭代数）。在此期间每步收集 amax，推荐范围 5-20 |
| `--hif8-scale-update-interval` | int | 10 | 稳态阶段 amax 历史收集与 scale factor 更新的间隔（迭代数） |
| `--hif8-amax-history-len` | int | 128 | amax 历史缓冲区长度，推荐值 64 / 128 / 256 |
| `--no-hif8-step-recovery` | flag | False | 禁用 HiF8 NaN/Inf Step Recovery |

## 使用影响

- **训练连续性**：Step Recovery 在检测到 NaN/Inf 后重试同一批次而非跳过，保证训练数据流连续，避免收敛偏移。
- **数值稳定性**：通过重置 HiF8 amax 历史，消除被污染的 scale factor 对后续 step 的影响，避免级联失败。
- **确定性**：通过恢复完整的 RNG 状态，使重试 attempt 尽可能复现原始 attempt 的随机行为（dropout、mask、routing 等）。
- **开销**：仅在检测到 NaN/Inf 时产生额外开销（清理、重置、RNG 恢复、批次重放、一次额外前向反向）。正常 step 的额外开销仅为两次 2 x int32 的 `all_reduce(MAX)`，开销可忽略。wrapper 在非 `hif8_delayed` 场景下直接短路。
- **重试限制**：每个 step 最多重试一次。若重试仍失败，训练将中止并输出结构化诊断日志（包含 iteration、reason、rank、loss、grad_norm）。

## 参数组合限制

`hif8_delayed` recipe 仅允许与 `--fp8-format hif8` 组合使用。其他 fp8-format（如 `e4m3`）与 `hif8_delayed` 组合会在参数校验阶段报错。

<table><thead>
  <tr>
    <th width='120'>功能</th>
    <th>开启方式</th>
    <th>是否支持</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="3">HiF8 DTS 训练</td>
    <td rowspan="3">--transformer-impl transformer_engine
    <br> --fp8-format hif8
    <br> --fp8-recipe hif8_delayed </td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="3">HiF8 DTS + Step Recovery</td>
    <td rowspan="3">--transformer-impl transformer_engine
    <br> --fp8-format hif8
    <br> --fp8-recipe hif8_delayed
    <br>（默认启用，无需额外参数）</td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td rowspan="3">HiF8 DTS + 显式禁用 Step Recovery</td>
    <td rowspan="3">--transformer-impl transformer_engine
    <br> --fp8-format hif8
    <br> --fp8-recipe hif8_delayed
    <br> --no-hif8-step-recovery </td>
    <td style="text-align: center; vertical-align: middle">✅</td>
  </tr>
</tbody>
</table>

## 相关特性参考

- **[Megatron Transformer-engine](transformer_engine.md)**：TE 模块的整体介绍，包含 HiF8 数据格式、低精度训练 scaling 策略及 TE 模块功能说明。
- **[MXFP8 零冗余权重特性](mxfp8/Zero_Redundancy_Weight.md)**：另一种 FP8 scaling 策略下的显存优化方案。
