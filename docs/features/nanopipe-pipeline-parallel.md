# nanopipe流水线并行

## 问题分析

流水线并行是AI大模型大规模分布式训练的关键组成部分之一，但其效率收到流水线中bubble的影响，为了提高吞吐量，需要降低其bubble比例。

## 解决方案

在大模型流水线调度中，反向的input梯度和weight梯度通常是一起调度计算的，然而，通过分析他们计算的依赖关系，可以发现其实只有input梯度的计算存在相互层间的依赖关系。因此，通过独立调度反向的input梯度和weight梯度的计算，我们可以减少流水线调度的bubble。

![img](../../sources/images/virtual-pipeline.PNG)

反向Input梯度和Weight一起调度的Interleaved 1F1B

![img](../../sources/images/nanopipe.png)

独立调度imput梯度和weight梯度的nano-pipe

![img](../../sources/images/FBW.png)

独立调度weight计算展示图

### 解决思路:

* 分离weight梯度计算流程，通过修改RowParallelLinear和ColumnParallelLinear的backward实现，将对weight的梯度计算进行剥离，先存储在调度器的dw计算队列中。
* 在需要对dw计算时，从调度器的dw计算队列中pop出一个计算，然后计算对应的梯度。

### 内存优化及通信隐藏：

相比[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241),在做dw计算时matmul要用的张量存在内存中没有释放，导致动态内存占用过大, nanopipe仅保留部分必要的激活值，节省大量内存，但是增多了通信。再采用并行matmul和all-gather以达到通信隐藏的目的。

其中，将input梯度计算的列切割中的all-gather通信转移到行切割的前方并进行通信隐藏，节省了一个all-gather的通信量。

![img](../../sources/images/ag_mm.png)

## 完整的nanopipe

![img](../../sources/images/nanopipe_mm.png)

## 使用场景

在训练模型时，降低bubble的比例，从而提升计算效率，达到更好的流水线并行。

## 使用方法

nanopipe依赖于vpp，设置`--nun-layers-per-virtual-pipeline-stage N`。要求`--pipeline-model-parallel-size` > 2
设置`--use-nanopipe`，默认为False，根据用户需求配置。

## 使用效果

提升计算效率，减少bubble占比。


场景1：单机, TP=1, 关SP, PP=4, SEQ=4096, hidden-size=4096
| device | TP | SP | PP | SEQ | hidden-size | Nano vs vpp收益 |
| :-----: | :----: | :----: | :-----:| :----: | :----: | :-----: |
| 单机 | 1 | 关 | 4 | 4096 | 4096 | 3.24% |

场景2：双机, TP=4, PP=4, SEQ=8192, hidden-size=12288
| device | TP | SP | PP | SEQ | hidden size | Nano vs vpp收益 |
| :-----: | :----: | :----: | :-----:| :----: | :----: | :-----: |
| 双机 | 4 | 开 | 4 | 8192 | 8192 | 1.02% |

# nanoswap

## 问题分析

使用nano时grad从前向到反向需要持续存储在npu上，生命周期过长，多次累加会增大npu内存的峰值

## 解决方案

将过多的张量做offload动作存储到cpu上，在内存峰值过后再将其张量reload回npu上

### 解决思路

在前向时将上一轮过多的张量offload到cpu，再在连续的反向运算中途reload回npu上，通过swap流控制不会让reload和offload出现顺序错误

## 完整nanopipe-swap原理图

![img](../../sources/images/nanopipe_v2.png)

## 使用方法

基于nanopipe上的使用基础上再开启`--use-nanopipe-swap`

## 使用效果

优化npu内存峰值

场景：单机, TP=2, 开SP, PP=4, SEQ=1024, hidden-size=4096，开启mc2
| device | TP | SP | PP | SEQ | hidden-size | Nano内存峰值 |swap内存峰值 | Nano vs swap内存峰值下降 |
| :-----: | :----: | :----: | :-----:| :----: | :----: | :-----: | :-----: | :-----: |
| 单机 | 2 | 开 | 4 | 1024 | 4096 | 5520.62 | 5177.72 | 6.21% |
