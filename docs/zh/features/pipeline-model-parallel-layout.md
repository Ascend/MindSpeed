# Megatron 自定义流水线布局

## 背景与挑战

Megatron 的流水线并行（Pipeline Parallelism，PP）和虚拟流水线并行（Virtual Pipeline Parallelism，VPP）默认按 decoder 层数进行均匀切分。对于常规 dense 模型，这种方式通常可以满足训练需求；但在包含 embedding、loss、MTP 或不同计算开销 decoder 层的模型中，均匀切分可能导致不同流水 stage 的计算负载不均衡，进而增加流水线空泡并降低整体吞吐。

例如，在开启 MTP 或将 embedding、loss 计算纳入流水 stage 时，首尾 stage 可能承担额外计算；在 MoE 模型中，不同 stage 的 decoder 层数也可能需要按模型结构或性能 profiling 结果做非均匀划分。此时，仅依赖 `--num-layers-per-virtual-pipeline-stage` 无法表达每个 stage 的具体层类型和层数。

## 解决方案

MindSpeed 支持自定义PP/VPP 中每个 stage 持有的层类型，通过`--pipeline-model-parallel-layout` 参数使能。

该特性可以：

- 显式指定 embedding、decoder、MTP、loss 层所在的流水 stage。
- 支持同一 PP rank 上不同 VPP chunk 持有不同数量的 decoder 层。
- 根据 layout 中 stage 数量自动推导 VPP size。
- 与 MoE 跨 microbatch 前反向通信掩盖（`--moe-fb-overlap`）在受限场景下配合使用。

## Layout 字符串格式

`--pipeline-model-parallel-layout` 使用一个字符串描述所有流水 stage，stage 之间使用 `|` 分隔。字符串从前向计算顺序开始展开：先列出 VPP rank 0 上所有 PP stage，再列出 VPP rank 1 上所有 PP stage，依次类推。

支持的层类型如下：

| 字符 | 含义 |
| --- | --- |
| `E` | embedding 层 |
| `t` | transformer decoder 层 |
| `m` | MTP 层 |
| `L` | loss 层 |

格式规则如下：

- `|` 用于划分 stage。
- `,` 仅用于提升可读性，解析时会被忽略。
- `x*N` 表示重复单个字符，例如 `t*3` 等价于 `ttt`。
- `(pattern)*N` 表示重复一段布局，例如 `(tt|)*2` 等价于 `tt|tt|`。
- 连续的 `||` 可以表示空 stage，但与 `--moe-fb-overlap` 同时使用时暂不支持空 decoder chunk。

## 使用场景

该特性适用于以下场景：

- 需要将 embedding、loss 或 MTP 层显式放置到指定流水 stage。
- 不同流水 stage 的计算负载不均，需要通过非均匀 decoder 层数做负载均衡。
- 需要在 VPP 场景下让同一 rank 的不同 chunk 拥有不同层数。
- MoE 模型中希望结合 `--moe-fb-overlap` 使用自定义 PP/VPP 切分。

## 使用方法

在启动脚本中添加 `--pipeline-model-parallel-layout` 参数即可启用该特性。

以 `PP=2`、`VPP=2`、decoder 层数为 8、MTP 层数为 1 的模型为例：

```shell
--pipeline-model-parallel-size 2
--pipeline-model-parallel-layout Ett|tt|ttt|tmL
--num-layers 8
--mtp-num-layers 1
```

该 layout 包含 4 个 stage，`4 / PP = 2`，因此会自动推导出 `VPP=2`。对应的 stage 分布如下：

| PP rank | VPP rank 0 | VPP rank 1 |
| --- | --- | --- |
| 0 | `Ett` | `ttt` |
| 1 | `tt` | `tmL` |

其中 decoder 层总数为 `2 + 2 + 3 + 1 = 8`，MTP 层总数为 1，embedding 和 loss 各出现一次。

如果只使用 PP 而不使用 VPP，layout 的 stage 数量应等于 `--pipeline-model-parallel-size`。例如：

```shell
--pipeline-model-parallel-size 2
--pipeline-model-parallel-layout Etttt|ttttL
--num-layers 8
```

## 与 MoE FB overlap 配合使用

当 `--pipeline-model-parallel-layout` 与 `--moe-fb-overlap` 同时开启时，MindSpeed 会使用 layout-aware 的 VPP 调度逻辑，并支持同一 rank 上不同 chunk 的 decoder 层数不一致。该场景下，前反向 overlap 会按照当前 forward chunk 与 backward chunk 的实际 layer graph 数量执行；当两侧层数不一致时，先对可配对的层执行 overlap，再处理剩余层。

典型 MoE 配置如下：

```shell
--pipeline-model-parallel-size 2
--pipeline-model-parallel-layout Ett|tt|ttt|tmL
--num-layers 8
--mtp-num-layers 1
--expert-model-parallel-size 2
--expert-tensor-parallel-size 1
--num-experts 8
--moe-grouped-gemm
--moe-token-dispatcher-type alltoall
--moe-fb-overlap
```

使用该组合时需要满足 `--moe-fb-overlap` 原有约束，例如使用 `alltoall` dispatcher、开启 `--moe-grouped-gemm`、`--expert-tensor-parallel-size=1` 且 `--expert-model-parallel-size > 1`。

## 使用约束

使用 `--pipeline-model-parallel-layout` 时存在以下约束：

1. `layout` 中 stage 数量必须能被 `--pipeline-model-parallel-size` 整除。
2. `layout` 中必须且只能包含一个 embedding 层和一个 loss 层。
3. `layout` 中 decoder 层数量必须与 `--num-layers` 一致。
4. 如使用 MTP，`layout` 中 `m` 的数量必须与 `--mtp-num-layers` 一致，且 decoder 层必须放在 MTP 层之前。
5. 当前暂不支持 encoder 层。
6. 不能与 `--num-layers-per-virtual-pipeline-stage`、`--num-virtual-stages-per-pipeline-rank` 同时配置。
7. 不能与 `--pipeline-num-transformer-layers`、`--noop-layers`、`--schedules-method dualpipev` 同时使用。
8. 当前不支持与 `--recompute-in-bubble` 或 `--recompute-in-advance` 同时使用。
9. 当 layout 推导出 VPP 时，暂不支持与 `--optimize-send-recv-comm` 同时使用。
10. 与 `--moe-fb-overlap` 同时使用时，暂不支持空 decoder chunk，也暂不支持 `--noop-layers + --pipeline-model-parallel-layout + --moe-fb-overlap` 组合。

## 使用效果

通过自定义流水线布局，用户可以将 embedding、loss、MTP 以及 decoder 层按实际计算负载放置到不同 stage 中，减少 PP/VPP 中的负载不均和流水线等待。对于 MoE 场景，该特性还可以与 `--moe-fb-overlap` 配合，在非均匀 chunk 层数下继续执行跨 microbatch 前反向通信掩盖，从而提升流水线调度的灵活性。
