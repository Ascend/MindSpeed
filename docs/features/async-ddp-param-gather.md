# 权重更新通信隐藏

## 问题分析

大模型训练过程中，通常会使用数据并行。在进行梯度更新时，数据并行组中的通信要等反向计算完成后再进行。这样的串行执行顺序会造成计算和通信流存在一定的空闲等待时间，导致执行效率较低。

## 解决方案

通过计算和通信任务并行的方式来实现相互的流水掩盖。

### a. 仅打开 `--use-distributed-optimizer`
仅打开分布式优化器时（`--use-distributed-optimizer`），运行流程如下图所示，前向和反向计算完成后，会有独立的通信时间，进行梯度的reduce-scatter、计算权重、进行权重的all-gather，获得权重之后再进入下一轮的前向计算。
<p align="center"> <img src="../../sources/images/async_ddp_param_gather_a.png" height="350px" width="880px"></p>

### b. 打开 `--use-distributed-optimizer` 和 `--overlap-grad-reduce`
在打开`--use-distributed-optimizer`的同时打开`--overlap-grad-reduce`，运行流程如下图所示，对梯度的reduce-scatter过程与反向计算过程并行，从而节省了单独的reduce-scatter过程，提高了计算-通信并行效率。
<p align="center"> <img src="../../sources/images/async_ddp_param_gather_b.png" height="350px" width="880px"></p>

### c. 打开 `--use-distributed-optimizer` 和 `--overlap-grad-reduce` 和 `--overlap-param-gather`
在打开`--use-distributed-optimizer`和`--overlap-grad-reduce`的基础上进一步打开`--overlap-param-gather`，运行流程如下图所示，对权重的all-gather过程与下一轮的前向计算并行，从而节省了单独的all-gather过程。
<p align="center"> <img src="../../sources/images/async_ddp_param_gather_c.png" height="350px" width="880px"></p>

以上流程对比发现，打开--overlap-param-gather后，通信与计算完全并行，极大提高了计算-通信并行效率，进而提升了模型训练效率。

## 使用场景

在数据并行场景可以开启该特性。

## 使用方法

设置`--overlap-param-gather`即可调用该算法。
确保同时开启了`--use-distributed-optimizer`和`--overlap-grad-reduce`。

## 使用效果

使用该特性可以提升性能。
legacy下，`--overlap-param-gather`暂不支持和`reuse_fp32_param`一起使用。
