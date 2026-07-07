# Megatron Pipeline Parallelism

## Background and Challenges

In the current era of large model training, a single computing device often cannot accommodate the storage and computation requirements of an entire model. Although model parallelism strategies can decompose a model across multiple devices to enable parallel training, traditional model parallelism suffers from significant idle time between devices, resulting in low computational resource utilization.

## Solution

To overcome these challenges, pipeline parallelism (PP) technology has emerged. Its core idea draws on the assembly line concept from industrial production: by partitioning the model into multiple stages and assigning them to different computing devices, it enables relay-style parallel computation across successive stages. This minimizes idle time between devices and shortens the distance between forward and backward computations, thereby effectively reducing memory consumption.

The specific implementation strategies for pipeline parallelism are as follows:

* Stage partitioning: The entire neural network model is divided into multiple logical stages, with each stage executed on a different computing device.
* Relay-style parallelism: A relay-style parallel computation approach is adopted between stages, where the computation result of the current stage serves as the input for the next stage, forming a continuous forward and backward computation flow.
* Warm-up and cool-down: At the beginning of pipeline parallel training, a warm-up phase is required to initialize the computation flow; at the end of training, a cool-down phase ensures that all computation tasks across stages are completed.

### Figure 1 Pipeline parallelism scheduling diagram

![alt text](../figures/pipedream1F1B.png)

For details, refer to the paper [Original Link](https://arxiv.org/pdf/1806.03377)

## Application Scenario

Pipeline parallelism is suitable for the following scenarios:

* Large model scale: The model has a huge number of parameters, making it difficult for a single device to handle the storage and computation requirements.
* Abundant computing resources: Sufficient computing devices are available to support model stage partitioning and parallel computation.
* Optimized storage and computing efficiency: Aims to reduce the storage overhead of individual devices and improve the utilization of computing resources.

## Usage

To enable pipeline parallelism, add the following parameter configuration to the training script:
`--pipeline-model-parallel-size  N      # N represents the number of pipeline stages, i.e., the number of NPUs participating in parallel training`
Users should configure this according to actual needs. The default value is 1.

## Application Effects

Pipeline parallelism boosts computational efficiency and lowers memory usage during training, while also optimizing resource allocation.
