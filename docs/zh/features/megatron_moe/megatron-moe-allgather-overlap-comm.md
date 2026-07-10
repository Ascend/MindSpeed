# Megatron MoE Allgather Dispatcher分支通信隐藏优化

## 背景与挑战

MoE中，存在大量的EP通信没有做通信隐藏，端到端时间占比大。这些耗时可以通过和计算交替进行，从而提高模型的训练性能。

## 解决方案

在前向过程中，使用异步通信来尽可能与计算做互相掩盖。同时，对整个计算流程进行子图切分，从而在反向过程中也进行通算并行，加速模型训练。此特性对allgather dispatcher进行了针对性优化。

## 使用方法

- 开启特性：添加`--moe-allgather-overlap-comm`启用该功能。
- 必选配套参数：必须同时确保开启以下三个参数：
    - `--moe-permutation-async-comm`
    - `--moe-token-dispatcher-type allgather`
    - `--moe-grouped-gemm`（注意：目前仅支持Grouped MLP）

## 使用场景

适用于megatron-moe，dropless方案分支时，在ep通信瓶颈时，需要通信隐藏ep通信的场景。

> [!NOTE]
>
> 启动该特性会导致显存占用增加，属正常现象。
