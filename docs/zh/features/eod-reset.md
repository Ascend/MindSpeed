# 支持EOD Reset训练场景

## EOD Reset训练场景

通常一个批次中输入进模型的文本序列是由多个文档（doc）拼接得到。在默认情况下，多个文档被视为同一序列，互相间的self attention没有掩盖。在特定情况下，多个文档间要求独立，文档间不能互相做self attention，在这种情况下attention mask和position ids需要在每个文档结束的位置（EOD）被重新设置。--reset-position-ids参数关闭时，整个序列计算位置编码；开启时，在每个序列内独立计算位置编码。

## 解决方案

通过调用底层 Flash Attention 算子的可变长模式，支持 EOD Reset 训练；可与 Ring Attention 或 KVAllGather 长序列并行组合。

## 使用方式

**后端限制：** CP>1 必须设置 `--transformer-impl transformer_engine`。

| 并行方式 | 支持的 mask 类型 |
| --- | --- |
| CP=1 | causal、general |
| Ring（p2p） | causal |
| KVAllGather（all_gather / allgather） | causal |
| Ulysses（a2a） | 不支持 |
| 混合 CP（a2a+p2p） | 不支持 |

CP>1 时需保证各层的 CP 通信类型一致。

### 1. 数据准备

（1）首先确保每一个文档的末尾都添加了EOD Token  
（2）使用 `p2p + causal` 且 CP>1 时，每个子序列会在线 pad 到 `CP*lcm(2, TP)` 的倍数，lcm 为最小公倍数。该逐文档 pad 不用于 CP=1 或 KVAllGather。CP>1 的 EOD Reset 均需开启 `--variable-seq-lengths`。

### 2. 参数设置

| 参数 | 支持情况与作用 |
| --- | --- |
| `--use-flash-attn` | 使用 `local` 时需开启。 |
| `--reset-attention-mask` | 开启文档间的注意力隔离；支持组合见上表。 |
| `--reset-position-ids` | 可开可关；开启后各文档的位置编码重新计数，关闭后保持连续。 |
| `--eod-mask-loss` | 可选，屏蔽 EOD token 对应位置的 loss；DSA 不支持该选项。 |
| `--fix-sub-seq-length N` | 默认 -1，按 EOD 边界划分；0<N≤seq-length 时改用固定子序列长度，最后一段保留余长；其他取值仍按 EOD 边界划分。 |
| `--variable-seq-lengths` | CP>1 时必需；CP=1 不因 EOD Reset 强制要求开启。 |

### 3. 注意事项

- 开启 CP 或 SP 时必须设置 `--micro-batch-size 1`；CP=1 且未开启 SP 时支持 `--micro-batch-size` 大于 1。
- EOD Reset 与 MoE 组合时，若开启 `--variable-seq-lengths`，使用 `alltoall` dispatcher；`allgather` 不支持动态序列长度。
- `--reset-attention-mask` 会关闭 RoPE 融合。
- 不支持 SFT packed sequence。
