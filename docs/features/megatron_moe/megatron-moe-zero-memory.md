# Megatron MoE alltoall dispatcher分支内存优化

## 背景与挑战

MoE中动态内存比较大，在使用overlap策略后，动态内存会进一步提高，内存墙问题严重。此时，使用普通重计算会因为粒度较粗，加重重计算导致的性能问题。

## 解决方案

针对这种场景，使用重通信，细粒度的重计算和针对性swap进行内存节省，采用计算掩盖重通信和swap，将重计算与未掩盖通信进行隐藏。
支持`alltoall` 与 `alltoallseq` dispatcher。 

- level0在专家计算部分进行重计算，性能损失较小，level1进行更大程度上的重计算，性能损失相对level0多，节省的内存分别为重计算MLP能节省内存的70%+和90%+，速度相比重计算MLP性能更优。
- 此处MLP亦包含共享专家部分。
- 在`alltoall`分支中，进行了probs重计算的前移，进一步提高内存节约的效果。

## 使用方法

打开启用该特性。
`--moe-zero-memory level0` 或者 `--moe-zero-memory level1`

同时需要开启：
- `--moe-alltoall-overlap-comm` 或 `--moe-fb-overlap` 使用该特性。
- 如在`--moe-fb-overlap`条件下使用，具体注意事项请参照该特性说明文档。

其中level1下可配置内存优化的层数, 默认为所有层使能:
- `--moe-zero-memory-num-layers x`

其中x为设置层数，x应大于或等于0且小于等于模型层数(num_layers//pp)，其中level0因为性能损失较小，不支持配置层数，功能为所有层使能；

## 适用场景

1.目前支持`alltoall` `alltoall_seq` dispatcher模式，适用megatron-moe需要重计算的场景。
2.支持`moe-fb-overlap` 状态下的level0使能。
3.`--moe-fb-overlap`下不支持`moe-zero-memory-num-layers`配置。
