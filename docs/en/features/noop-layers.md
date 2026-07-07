# Ascend Custom No-Op Layer

## Background and Challenges

During neural network training, the embedding operation in the initial layers and the logits computation in the final layers are typically compute-intensive tasks, which can significantly impact the overall efficiency of the network. Specifically:

- **Embedding Layer**: When processing text or categorical data, the embedding layer converts high-dimensional sparse features into low-dimensional dense vector representations. This process involves index lookups and potentially large-scale matrix multiplications. Particularly in natural language processing apps, the vocabulary may contain hundreds of thousands or even millions of entries. High-dimensional lookup and conversion operations consume substantial computational resources.

- **Logits Layer**: The logits layer at the end of the network is typically a fully connected layer that maps the hidden states of the final layer to the output space, providing unnormalized prediction values for subsequent loss function computation. If the classification task has a large number of categories, the weight matrix of this layer will be extremely large, causing the matrix multiplication operation to become a performance bottleneck.

The computational complexity of the above operations increases with the number of input features and categories, which may reduce training speed and create performance bottlenecks in environments with limited computing resources.

## Solution

To address the above challenges, we have introduced the No-Op Layers feature, which allows users to dynamically adjust the computational load of a model within the training pipeline by designating specific layers as No-Op Layers. This mechanism helps distribute workloads more evenly across multiple computing nodes, thereby optimizing the utilization of overall computing resources.

## Application Scenario

This feature is particularly useful when users encounter performance bottlenecks caused by uneven distribution of computing resources. By redistributing computationally intensive tasks, it can effectively reduce idle time (pipeline bubbles) in the pipeline, thereby improving system throughput and efficiency.

## Usage

To enable this feature, users need to set target layers as no-op layers through command-line parameters. For example, suppose the original model has 126 layers, with the model parameter `--num-layers 126`, meaning there are 126 layers that perform actual computation. If one no-op Layer is added at both the beginning and the end of this model, the model parameter should be set to `--num-layers 128 --noop-layers 0,127`, indicating a total of 128 layers, where the first and last layers (layer 0 and layer 127, with layer numbering starting from 0) are no-op layers that do not perform actual computation, and the middle 126 layers are the layers that perform actual computation.

## Application Effects

By implementing the strategy of custom addition of no-op layers, it is expected that pipeline bubbles can be significantly reduced, thereby optimizing the computation flow and improving system performance. This not only helps accelerate the model training process but also maximizes hardware resource utilization.

- Reduced pipeline bubbles: By redistributing computationally intensive tasks to different stages of the pipeline, idle time caused by high computational load in certain stages is reduced, improving the overall system throughput.

- Optimized computation flow: Ensures that computing and communication resources are utilized more fully, avoiding idle waste caused by synchronization barriers and enabling each computing unit to work efficiently.

According to actual test results, in MoE models at the scale of tens of billions of parameters, such as DeepSeekV2, where performance bottlenecks arise from uneven computational load caused by the terminal logits layer, using custom no-op layers can achieve over 10% end-to-end training performance improvement.

## Notes

When using the "Ascend Custom No-Op Layer" feature, the total number of layers changes after adding no-op layers. It is necessary to readjust the pipeline (virtual pipeline) configuration based on the total number of layers, including the no-op layers.
