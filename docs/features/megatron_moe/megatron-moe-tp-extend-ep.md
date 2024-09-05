# Megatron MoE TP拓展EP

## 问题分析

开启TP+EP后，专家层TP组切分专家参数，MoE细粒度小专家场景TP切分后GMM算子效率下降严重。

## 解决方案

针对小专家场景TP切分后GMM算子效率下降问题，专家层TP组不切分专家参数，切分专家数量。

## 使用方法

打开`--moe-tp-extend-ep`启用该特性。

同时需要开启：
- `--moe-permutation-async-comm`
- `--moe-grouped-gemm`，目前仅支持Grouped MLP。

同时需要确保`--num-moe-experts`能被`tp * ep`整除。

## 适用场景

细粒度小专家，类DeepSeek-V2模型，每个专家的参数量较小。



