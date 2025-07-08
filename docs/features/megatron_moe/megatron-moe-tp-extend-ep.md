# Megatron MoE TP拓展EP

## 背景与挑战

开启TP+EP后，专家层TP组切分专家参数，MoE细粒度小专家场景TP切分后GMM算子效率下降严重。

## 解决方案

针对小专家场景TP切分后GMM算子效率下降问题，专家层TP组不切分专家参数，切分专家数量。

## 使用场景

细粒度小专家，类DeepSeek-V2模型，每个专家的参数量较小。

## 使用方法

打开`--moe-tp-extend-ep`启用该特性。

同时需要开启：
- `--moe-permutation-async-comm`
- `--moe-grouped-gemm`，目前仅支持Grouped MLP。

同时需要确保`--num-experts`能被`tp * ep`整除。

##### 注意：
当前该特性不支持Moe Token drop and pad模式，即`--moe-expert-capacity-factor`需要为None。
当前仅支持alltoall dispatcher。

## 使用效果
通过避免TP切分专家参数，提高小专家场景GMM算子效率，从而提高模型整体训练性能，在类DeepSeekV2万亿参数级别的MoE模型下，并且为细粒度小专家，性能最高提升10%以上。
