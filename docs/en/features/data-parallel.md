# Megatron Data Parallelism

## Background and Challenges

When training models on large-scale datasets, a single computing device often cannot handle the processing load of the entire dataset, resulting in excessively long training cycles. To address this challenge, the original dataset must be effectively partitioned, ensuring that each computing device only needs to process a portion of the dataset, thereby significantly improving training efficiency.

## Solution

The data parallelism (DP) strategy divides the dataset into multiple batches and distributes them evenly across computing devices, so that each device is only responsible for processing a specific batch of data.

The implementation of this scheme requires the following two key elements:
The model structure and parameters deployed on each computing device must remain completely identical.
The data batches processed by each device must be different from one another, ensuring the parallelism and efficiency of the training process.
The overall approach of this scheme is as follows:

* Model replication: Store a complete model replica on each computing device.
* Data partitioning: The original dataset is subdivided into several batches, which are then evenly distributed across all computing devices to ensure load balancing.
* Gradient synchronization: After completing the forward computation and obtaining local gradients, an All-Reduce operation is performed to aggregate the gradients from all devices, calculate the average, and then broadcast the result back to each device, thereby maintaining global parameter consistency.

## Application Scenario

Data parallelism is applicable in the following scenarios:

* Large-scale datasets: When the training dataset is so large that a single device cannot complete processing within a reasonable time.
* Sufficient computing resources: When a sufficient number of computing devices are available to store multiple complete model replicas and support parallel training, thereby effectively shortening the training cycle and reducing the computational load on individual devices.

## Usage

Enabling and configuring data parallelism primarily depends on the following metrics:

* World size: The total number of NPUs participating in parallel training.
* Tensor model parallel size: The number of parallel partitions for model weights.
* Pipeline model parallel size: The degree of pipeline parallelism for the model architecture.
* Context parallel size: The parallel strategy for processing long sequence data.

data_parallel_size = world_size/(tensor_model_parallel_size × pipeline_model_parallel_size × context_parallel_size)

### Notes

* The total number of model layers must be divisible by the pipeline parallelism size.
* `global_batch_size` must be divisible by `data_parallel_size`.

## Application Effects

Data parallelism is automatically calculated based on the settings of other parallelism strategies. The data parallelism approach can significantly shorten the training cycle, especially when processing high-dimensional features and large-scale datasets, and can improve the utilization of hardware resources. The data parallelism architecture offers excellent horizontal scalability, enabling linear performance improvement as computing resources increase. It is easy to deploy and adjust in cluster environments of different scales to meet the computing requirements at various stages.
