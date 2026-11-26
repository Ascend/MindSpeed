# DeepSeek Sparse Attention (DSA)

MindSpeed's DSA implementation based on Megatron-Core provides NPU sparse attention, Indexer, KL loss fusion operators, matrix absorption, and KVAllGather context parallelism adaptation.

## Usage

**Backend limitation:** `--transformer-impl transformer_engine` must be set, and `local` is not supported even when CP=1.

Enable `--transformer-impl transformer_engine --experimental-attention-variant dsa --multi-latent-attention --qk-layernorm` in the model configuration. For MLA dimension configuration, see [MLA](multi-head-latent-attention.md). The Indexer uses Megatron's native `--dsa-indexer-head-dim`, `--dsa-indexer-n-heads`, `--dsa-indexer-topk`, and `--dsa-indexer-loss-coeff` arguments.

The following MindSpeed switches are disabled by default; when all of them are disabled, the Megatron native DSA path is used.

| Parameter | Effect and Constraint |
| --- | --- |
| `--use-fused-sparse-flash-attention` (P1) | NPU sparse attention; can be enabled on its own, but CP=1 is required in this case. |
| `--use-fused-lightning-indexer` (P2) | NPU Indexer; must be enabled together with P1 and P3. |
| `--use-fused-lightning-indexer-kl-loss` (P3) | NPU Indexer KL loss; must be enabled together with P1 and P2; `--num-attention-heads` only supports 32, 64, and 128. |
| `--apply-rope-in-complex` | The Indexer uses complex-form RoPE; can be enabled independently of P1/P2/P3. |
| `--use-dsa-absorb` | Use matrix absorption during training; can be enabled independently of P1/P2/P3. |
| `--apply-rope-fusion` | Non-packed Indexer uses Ascend fused RoPE; when complex RoPE is also enabled, the Indexer preferentially uses the complex path. |

When CP>1, set `--transformer-impl transformer_engine --cp-comm-type all_gather`, see [KVAllGather](kvallgather-context-parallel.md). The fusion path must enable P1/P2/P3 simultaneously; enabling P1 alone in combination with CP is not supported. `--eod-mask-loss` is not supported.

## Checkpoint and Training

The matrix absorption path splits the KV weights during training and merges them into the Megatron native KV layout in the checkpoint. The Indexer loss is normalized by micro-batch size. PP/VPP does not require additional DSA-specific pipeline switches.
