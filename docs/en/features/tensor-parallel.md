# Megatron Tensor Parallelism

## Background and Challenges

As the complexity and scale of deep learning models continue to grow, the memory capacity and processing power of a single computing device have gradually become bottlenecks that constrain model training efficiency. The enormous model size not only exceeds the memory limits of a single processor but also significantly prolongs the training cycle. To address this challenge, it is imperative to effectively partition the model, enabling parallel storage and computation across multiple computing devices to accelerate training and reduce memory footprint.

## Solution

Tensor parallelism (TP), as a specific implementation of model parallelism, partitions the model's parameter matrices across multiple computing devices, effectively distributing model weights and optimizer states, thereby overcoming the limitation of a single device's memory capacity. Tensor parallelism not only significantly reduces the memory requirement per device but also greatly accelerates training speed, as each device only needs to handle a portion of the model's computation.
The tensor parallelism strategy primarily includes the following two partitioning approaches:

* Row-wise parameter matrix splitting
The model is partitioned along the row dimension of the parameter matrix. This strategy requires the input matrix to be correspondingly partitioned along the column dimension.
  * During the forward pass of the row-wise splitting strategy, the input matrix is first partitioned. Each partitioned input matrix then enters the corresponding model part for computation. Subsequently, the All-Reduce operation aggregates the computation results from each part to produce the final forward computation output.
  * During the backward pass of the row-wise splitting strategy, the gradient of the final output is distributed proportionally to the output gradients of each model part. The gradients of each partitioned input matrix are then merged using the All-Gather operation to reconstruct the gradient of the original input matrix.
* Column-wise parameter matrix splitting
The model is partitioned along the column dimension of the parameter matrix, in which case the input matrix does not need to be partitioned.

  * During the forward pass of the column-wise partitioning strategy, the complete input matrix is fed into each model partition. After each partition completes its computation independently, the output results from all partitions are concatenated via an All-Gather operation to form the final forward computation output.
  * During the backward pass of the column-wise partitioning strategy, the gradient of the final output is first split proportionally and passed to the output gradients of each model partition. Subsequently, the gradients of the input matrices from all partitions are aggregated via an All-Reduce operation to obtain the gradient of the initial input matrix.

### Figure 1: Tensor parallelism diagram

<p align="center"> <img src="../figures/tensor-parallel.png" height="550px" width="650px"></p>

## Application Scenario

Tensor parallelism is applicable to the following scenarios:

* High memory usage: When memory usage during training approaches or exceeds the processor memory limit, causing training to become unstable or impossible.
* Lengthy training cycles: When the model scale is large and single-device training time is too long, impacting research and development efficiency and cost.

## Usage

To enable tensor parallelism, add the following parameter configuration to the training script:
 `--tensor-model-parallel-size N       # Set the tensor model parallel size, where N is the number of NPUs in a single parallel group`

## Application Effects

By leveraging the tensor parallelism strategy, multiple computing devices share the model storage and computation load. This not only significantly reduces the memory footprint on a single device but also effectively shortens the model training cycle. Furthermore, because the parameters are partitioned, this strategy can resolve the issue where a single computing device cannot fully accommodate a layer with a large number of parameters.
