# Megatron Sequence Parallelism

## Background and Challenges

In large model training, while tensor parallelism can effectively reduce memory usage and accelerate training, it requires splitting each layer of the model into independent blocks, which has limitations when handling operations such as LayerNorm and Dropout. Although these operations have low computational cost, the redundant memory required to store activations is significant. To address this issue, sequence parallelism was introduced as a complementary strategy to offload the memory and computation burdens that cannot be partitioned in tensor parallelism.

## Solution

The sequence parallelism (SP) strategy further improves computational efficiency on top of tensor parallelism. It is a parallel computing approach that partitions the sequence dimension of input data.

The inputs of operations such as LayerNorm and Dropout are partitioned along the sequence dimension, so that each device only needs to perform a portion of the Dropout and LayerNorm operations.

For ease of understanding, take the following figure as an example: assume the input $X$ has a size of $ s \times b \times h $, and it is partitioned along the sequence dimension as $X=[X_1^s,X_2^s]$. After the LayerNorm operation, the result is $Y=[Y_1^s,Y_2^s]$, which is then followed by tensor model parallelism.

![image.png](../figures/sequence-parallel.png)

[Original paper](https://arxiv.org/pdf/2205.05198)

## Application Scenario

Sequence parallelism is applicable to the following scenarios:

* High memory usage: Even with tensor parallelism, memory usage is still close to or exceeds the processor memory limit.
* Computing resource optimization: It aims to further reduce memory overhead and improve training speed.

## Usage

Sequence parallelism depends on tensor parallelism. Therefore, to enable sequence parallelism, the following parameter configurations must be added to the training script:

`--tensor-model-parallel-size  N        # Set the tensor model parallel size, where N is the number of NPUs in a single parallel group`
`--sequence-parallel      # Enable sequence parallelism`

## Application Effects

Through the sequence parallelism strategy, leveraging multi-device collaboration, memory usage is further reduced on top of tensor parallelism, enabling devices to accommodate model training with larger parameters.
