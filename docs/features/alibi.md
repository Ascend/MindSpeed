# Alibi 位置编码

## 问题分析

当大模型在训练和预测时的输入长度不一致时，模型的泛化能力会下降。若外推能力不佳，大模型在处理长文本或多轮对话时的效果就会受到限制。正弦位置编码的外推能力比较弱，RoPE（Rotary Position Embedding）的外推能力有一定提高但仍然有限。

## 解决方案

支持Alibi位置编码，提高模型外推能力。

### 解决思路:

Alibi算法给attention score添加了一个预设的线性偏置矩阵（如下图所示），使模型能够理解输入之间的相对位置关系。由于位置信息直接作用于attention score上，位置差异性被突出，使模型具有较强的外推能力。
<p align="center"> <img src="../../sources/images/alibi.png" height="180px" width="400px"></p>

[原文链接](https://arxiv.org/pdf/2108.12409)


## 使用方法

（1）对于不开启`--use-fusion-attn-v2`特性的情况，设置`--position-embedding-type alibi`即可调用该算法。

（2）对于开启`--use-fusion-attn-v2`特性的情况设置，需要设置`--position-embedding-type alibi`和`--alibi-fusion-attn-type 2`（支持0，2，3）。
0表示生成alibi后传入，1暂不开放， 2和3表示核内生成， 3做pse的时候会做sqrt。
如果要设置alibi为对角线对称取反，则需设置`alibi_diagonal_opposite`，反之（亦是默认情况，且与2和3时核内生成一致）无需进行设置。

（3）目前alibi位置编码已经支持ring-attention长序列并行，当前只支持mask为causal的场景，以及 `--alibi-fusion-attn-type` 为2，3的压缩模式。暂不支持ulysses长序列并行和混合长序列并行。

（4）开启`--use-fusion-attn-v2`特性和长序列并行时，alibi编码不支持开启dropout。

## 使用效果

模型外推能力提高。
