# Support for the EOD Reset Training Scenario

## EOD Reset Training Scenario

Typically, a text sequence fed into the model within a batch is formed by concatenating multiple documents. By default, these documents are treated as a single sequence, with no masking applied to the self attention between them. In certain cases, the documents are required to be independent, meaning they cannot perform self attention on each other. In this scenario, the attention mask and position ids need to be reset at the end of each document (EOD). When the `--reset-position-ids` parameter is disabled, position encoding is calculated for the entire sequence; when enabled, position encoding is calculated independently within each sequence.

## Solution

The EOD Reset training scenario is supported by invoking the variable-length mode of the underlying Flash Attention operator; it can be combined with Ring Attention or KVAllGather long sequence parallelism.

## Usage

**Backend restriction:** When CP>1, `--transformer-impl transformer_engine` must be set.

| Parallelism Method | Supported Mask Types |
| --- | --- |
| CP=1 | causal, general |
| Ring (p2p) | causal |
| KVAllGather (all_gather / allgather) | causal |
| Ulysses (a2a) | Not supported |
| Hybrid CP (a2a+p2p) | Not supported |

When CP>1, ensure that the CP communication type is consistent across layers.

### 1. Data Preparation

(1) First, ensure that an EOD Token is added to the end of each document.  
(2) When using `p2p + causal` and CP>1, each subsequence is padded online to a multiple of `CP*lcm(2, TP)`, where lcm is the least common multiple. This per-document padding does not apply to CP=1 or KVAllGather. For EOD Reset with CP>1, `--variable-seq-lengths` must be enabled.

### 2. Parameter Settings

| Parameter | Support and Function |
| --- | --- |
| `--use-flash-attn` | Must be enabled when using `local`. |
| `--reset-attention-mask` | Enables attention isolation between documents; for supported combinations, see the table above. |
| `--reset-position-ids` | Can be enabled or disabled; when enabled, the position encoding of each document is recounted, and when disabled, it remains continuous. |
| `--eod-mask-loss` | Optional; masks the loss at the position of the EOD token; DSA does not support this option. |
| `--fix-sub-seq-length N` | Default is -1, which splits by EOD boundaries; when 0<N<=seq-length, a fixed subsequence length is used and the last segment retains the remainder; other values still split by EOD boundaries. |
| `--variable-seq-lengths` | Required when CP>1; for CP=1, it is not mandatory for EOD Reset. |

### 3. Notes

- When CP or SP is enabled, `--micro-batch-size 1` must be set; when CP=1 and SP is not enabled, `--micro-batch-size` greater than 1 is supported.
- When EOD Reset is combined with MoE, if `--variable-seq-lengths` is enabled, use the `alltoall` dispatcher; `allgather` does not support dynamic sequence lengths.
- `--reset-attention-mask` disables RoPE fusion.
- SFT packed sequence is not supported.
