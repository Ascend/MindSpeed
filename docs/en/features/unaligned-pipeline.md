# Non-aligned PP and VPP Allocation

## Background and Challenges

Megatron-LM-like frameworks have become one of the mainstream solutions for large model training. PP (Pipeline Parallelism) and VPP (Virtual Pipeline Parallelism) are fundamental parallel paradigms for large model training, but they suffer from computational imbalance in certain scenarios. Specifically:

- **Embedding Layer**: When processing text or categorical data, the embedding layer converts high-dimensional sparse features into low-dimensional dense vector representations. This process involves index lookups and potentially large-scale matrix multiplications. Particularly in natural language processing apps, the vocabulary may contain hundreds of thousands or even millions of entries. High-dimensional lookup and conversion operations consume substantial computational resources.

- **Logits Layer**: The logits layer at the end of the network is typically a fully connected layer that maps the final hidden states to the output space, providing unnormalized predictions for subsequent loss function computation. If the classification task involves a large number of categories, the weight matrix of this layer becomes extremely large, making the matrix multiplication operation a performance bottleneck.

- **Multi-Token Prediction**: A module located at the end of the network that extends the prediction scope to multiple future tokens at each position. It expands the estimation capability of the logits layer while placing higher demands on computational resources.

The computational complexity of the above operations increases with the number of input features and categories, which may lead to reduced training speed and create performance bottlenecks in environments with limited computational resources.

## Solution

To address the above challenges, we have introduced the "Unaligned PP and VPP Sharding" feature, which allows users to dynamically adjust the computational load of the model within the training pipeline by specifying the number of transformer layers distributed to each PP stage and VPP stage. This mechanism helps distribute the workload more evenly across multiple computing nodes, thereby optimizing the utilization of overall computational resources.

## Application Scenario

- This feature is particularly useful when users encounter performance bottlenecks caused by uneven distribution of computational resources. By redistributing compute-intensive tasks, it effectively reduces idle time (bubbles) in the pipeline, thereby improving system throughput and efficiency.

## Usage

Add the --pipeline-num-transformer-layers parameter to the model parameters, using a two-dimensional matrix to represent the number of transformer layers in PP layers and VPP layers. The horizontal axis represents pp rank, and the vertical axis represents vpp rank.
Assume: pipeline_num_transformer_layers = `[[0,1],[1,1]*4,[1,0]]`, pp_rank = 0, vpp_rank = 1, then `pipeline_num_transformer_layers[pp_rank][vpp_rank]`contains only 1 layer.
If VPP is not configured, it can be set as pipeline_num_transformer_layers = `[[1],[2]*4,[1]]`. If pp_rank = 1, then `pipeline_num_transformer_layers[pp_rank]` contains 2 layers.

### Precautions

- Ensure that the total number of layers in the array matches the value in the --num-layers configuration item.
- Since the default VPP in the framework is evenly split, the content of a single PP stage must be divisible by the configuration item --num-layers-per-virtual-pipeline-stage. The actual effective configuration is controlled by the number of layers in the array.

### Setting Training Script Parameters

```shell
# Enable non-aligned PP/VPP
--pipeline-num-transformer-layers [[0,1],[1,1]*4,[1,0]] \
```

## Application Effects

By implementing a custom strategy for controlling the number of transformer layers in PP and VPP stages, it is expected to significantly reduce pipeline bubbles, thereby optimizing the computation flow and improving system performance. This not only helps accelerate the model training process but also maximizes hardware resource utilization.
