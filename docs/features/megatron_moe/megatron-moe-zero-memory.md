# Megatron MoE alltoall dispatcher分支内存优化

## 问题分析

MoE中动态内存比较大，内存墙问题严重，开重计算会导致性能劣化严重。

## 解决方案

针对这种场景，使用重通信，细粒度的重计算和针对性swap进行内存节省，采用计算掩盖重通信和swap，将重计算与未掩盖通信进行隐藏。

## 使用方法

打开启用该特性。
`--moe-zero-memory level0` 或者 `--moe-zero-memory level1`

其中level0性能损失较小，level1性能损失相对level0多，节省的内存分别为重计算mlp能节省内存的70%+和90%+，速度相比重计算mlp性能更优。

同时需要开启：
- `--moe-alltoall-overlap-comm`

其中level1下可配置内存优化的层数, 默认为所有层使能:
- `--moe-zero-memory-num-layers x`

其中x为设置层数，x应大于或等于0且小于等于模型层数(num_layers//pp)，其中level0因为性能损失较小，不支持配置层数，功能为所有层使能；

## 适用场景

目前只支持alltoall dispatcher模式，适用megatron-moe需要重计算的场景。



