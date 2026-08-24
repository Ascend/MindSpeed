# fusion_attention_v2

## Application Scenario

This method is the V2 version of FlashAttention, which extends certain functionalities beyond the V1 version. It currently only supports specific scenarios such as [ALiBi position encoding](./alibi.md) and is disabled by default.

For other scenarios, it is essentially identical to the V1 version, and it is not recommended to enable it proactively. For operator details, see [API Description](../ops/fusion_attention.md).

## Usage

Set `--use-fusion-attn-v2` to invoke this algorithm.

## Effects

The basic effect is equivalent to FlashAttention. For specific scenarios such as [ALiBi Positional Encoding](./alibi.md), it must be manually enabled.
