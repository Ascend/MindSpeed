# 高维张量并行

## 背景与挑战

大模型训练时，张量并行（TP）将模型参数切分到多个设备上以减少其内存的占用，在训练过程中为了更新参数梯度信息等，需要引入allreduce通信。当集群规模较大时，如果设置TP域很大时，其通信开销会变得很大，使得训练效率降低。

## 解决方案

为了提高大规模TP域通信效率，采用高维张量并行，其将激活值和参数同时切分到多个计算设备上，相对1D-TP降低了通信域、减少通信次数，从而减少通信时间，提升模型训练的性能。

### 解决思路

#### 2D张量并行策略

给定TP域大小，通过建立多通信域，在原Megatron（ColumnParallelLinear、RowParallelLinear）增加了一维的切分维度。将原tp通信域进行分解为两个子通信域tp_x和tp_y，需要满足`tp = tp_x * tp_y`。以MLP层为例，其实现过程如下：

![img](../../sources/images/tensor-parallel-2d.png)

#### 分布式normalization

在transformer网络中，normalization会将每一层神经元的输入都转换成均值方差都一样的，加快其收敛。在MLP和attention层分别进行2D张量并行时，其输入和输出都分别在first-dim和last-dim做了tp_x和tp_y的切分，如果继续使用原LayerNorm或者RMSNorm需要先将input进行沿first-dim进行all-gather(x)和沿last-dim进行all-gather(y)操作，才能保证input数据的完整性。为了提升这部分的性能，采用了分布式normalization。其处理流程如下：

##### **步骤1：计算输入的总和**

首先，计算输入张量$\mathbf{x}$ 在最后一个维度上的总和：

$$
e_x = \sum_{i=1}^{H} x_i
\
$$

##### **步骤2：分布式归约操作（All-Reduce）**

将步骤1中的总和 $e_x$ 在所有tp_y通信域进程中进行归约（求和），确保每个进程都拥有其通信域全局总和：
$$
\
e_x^{\text{global}} = \text{AllReduce}\left( e_x \right) = \sum_{p=1}^{P} \sum_{i=1}^{H} x_i^{(p)}
\
$$

其中：
- $P$ 是分布式进程的数量。
- $x_i^{(p)}$ 表示第 $p$ 个进程中第 $i$ 个元素的值。

##### **步骤3：计算输入元素的平方和**

接下来，计算输入张量每个元素的平方和：

$$
s_x = \sum_{i=1}^{H} x_i^2
$$

##### **步骤4：分布式归约操作（All-Reduce）**

将步骤3中的平方和 $s_x$ 在所有tp_y通信域进程中进行归约（求和），确保每个进程都拥有其通信域全局平方和：

$$
s_x^{\text{global}} = \text{AllReduce}\left( s_x \right) = \sum_{p=1}^{P} \sum_{i=1}^{H} \left( x_i^{(p)} \right)^2
$$

##### **步骤5：中心化输入数据**

将输入数据 $\mathbf{x}$ 中心化，即减去平均值。平均值 $\mu$ 计算如下：

$$
\mu = \frac{e_x^{\text{global}}}{H}
$$

然后，中心化输入：

$$
x'_i = x_i - \mu \quad \forall i \in \{1, 2, \dots, H\}
$$

##### **步骤6：计算总和的平方**

计算全局总和的平方：

$$
e_x'^2 = \left( e_x^{\text{global}} \right)^2
$$

##### **步骤7：计算归一化因子**

计算归一化因子 $\gamma$，用于标准化输入数据。公式如下：

$$
\gamma = \frac{1}{\sqrt{ \left( \frac{s_x^{\text{global}}}{H} \right) - e_x'^2 + \epsilon }}
$$

这里：
- $\frac{s_x^{\text{global}}}{H}$ 是全局平方和的平均值。
- $e_x'^2$ 是全局总和的平方。
- $\epsilon$ 是一个小常数，防止分母为零，增加数值稳定性。

##### **步骤8：标准化输入数据**

将中心化后的输入数据 $\mathbf{x}'$ 与归一化因子 $\gamma$ 相乘，得到标准化后的数据 $\mathbf{\hat{x}}$：

$$
\hat{x}_i = x'_i \cdot \gamma \quad \forall i \in \{1, 2, \dots, H\}
$$

##### **步骤9：应用权重和偏置**

最后，将标准化后的数据与权重向量 $\mathbf{W}$ 相乘，并根据是否存在偏置向量 $\mathbf{b}$ 来决定最终输出。

- **如果存在偏置**：

$$
\text{output}_i = b_i + W_i \cdot \hat{x}_i \quad \forall i \in \{1, 2, \dots, H\}
$$

- **如果不存在偏置**：

$$
\text{output}_i = W_i \cdot \hat{x}_i \quad \forall i \in \{1, 2, \dots, H\}
$$


## 使用场景

当TP通信域需要设置较大时，通信效率较低，需要通过分解通信域来提升其通信效率。

## 使用方法

在训练脚本的参数列表中加入 `--tp-2d`，开启2D张量并行，`--tp-x N1`和`--tp-y N2`分别设置其x轴、y轴的切分大小，其中需满足`tp = N1 * N2`(N1 > 1, N2 > 1)。

其他优化参数，用于辅助高维张量并行特性进行通信隐藏，需要开启tp-2d时生效：
- `--enable-overlap-ag-with-matmul`: 在linear层forward计算时，开启all-gather通信和matmul进行隐藏，以便加速
- `--enable-overlap-matmul-with-rs`: 在linear层forward计算时，开启matmul计算和reduce-scatter通信进行隐藏，以便加速
- `--coc-fused-kernel`: 在linear层forward计算时，开启计算通信融合算子，将matmul计算与all-gather、reduce-scatter都进行算子级融合，实现进一步加速（该特性不与前两个特性兼容，依赖ATB加速库）
- `--enable-backward-overlap-ag-with-matmul`: 在linear层backward计算梯度时，开启all-gather通信和matmul进行隐藏，以便加速（该特性依赖ATB加速库）

上述3个forward计算优化参数`--enable-overlap-ag-with-matmul`、`--enable-overlap-matmul-with-rs`、`--coc-fused-kernel`只能同时开启1个。

注意事项：

Megatron-MOE支持情况如表1所示

**表1** 高维张量并行支持Megatron-MOE

| Dispather | MLP            | 支持情况 | 性能优化 |
|-----------|----------------|------|------|
| AllGather | Sequential MLP | ✅    | ❌    |
| AllGather | Grouped MLP    | ✅    | ❌    |
| All2All   | Sequential MLP | ❌    | ❌    |
| All2All   | Grouped MLP    | ❌    | ❌    |

当前高维张量并行特性不与`--sequence-parallel`、`--use-fused-rmsnorm`等特性相兼容，请根据实际情况调整配置。

## 使用效果

在llama3-405B模型训练时，tp=16情况下，开启2D张量并行，tp_x=8，tp_y=2，相比原Megatron 1D张量并行性能提升5%+。
开启coc-fused-kernel和enable-backward-overlap-ag-with-matmul通信计算融合优化后，进一步提升性能5%+。
其他场景下，由于计算效率和通信组的划分差异，需根据tp_x和tp_y实际调优情况进行配置，部分配置不能保证效率提升。
