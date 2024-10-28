# Megatron MoE allgather dispatcher分支通信隐藏优化

## 问题分析

MoE中EP通信没有做通信隐藏，端到端时间占比大。

## 解决方案

针对这种场景，将没有数据依赖的计算与通信进行隐藏

## 使用方法

打开`--moe-allgather-overlap-comm`启用该特性。

同时需要开启：
- `--moe-permutation-async-comm`
- `--moe-token-dispatcher-type allgather`
- `--moe-grouped-gemm`，目前仅支持Grouped MLP。

## 适用场景

适用megatron-moe，dropless方案分支时候，ep通信瓶颈时，需要通信隐藏ep通信的场景。



