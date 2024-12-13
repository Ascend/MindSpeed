# fusion_attention_v2

## 使用场景

本方法为FlashAttention的V2版本，对V1版本进行了一定功能拓展，当前仅支持特定场景如[Alibi位置编码](./alibi.md)，默认关闭。

其它场景原则上与V1版本无差异，不建议主动开启。算子说明详情见[接口说明](../ops/fusion_attention.md)。

## 使用方法

设置`--use-fusion-attn-v2`即可调用该算法。

## 使用效果

基础效果等同于Flash attention，特定场景如[Alibi位置编码](./alibi.md)需手动开启。