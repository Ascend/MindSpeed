# KV Cache Optimization in Context Parallelism

## Background and Challenges

The context parallelism (CP) feature splits the sequence along the sequence dimension before attention computation. During attention computation, the full sequence needs to be calculated, so CP communication occurs during attention computation.

1. During attention computation, Ring CP cyclically receives KV from other cards via send/recv, ultimately ensuring that Q can attend to the full KV and maintaining mathematical equivalence with the non-split case. Currently, KV is discarded after forward computation completes, and send-recv is required again during the backward pass to retrieve KV. When computing with short sequences, communication time may exceed computation time, making it impossible for computation time to hide communication time, thereby degrading performance. Therefore, optimization is needed for scenarios where computation time is insufficient to hide communication time, in order to accelerate training in such cases.

2. In the Ulysses CP scheme under GQA models, after enabling TP, each rank typically has only one head. In this case, the communication volume of All2All is the same as that of AllGather. However, when there is only one head, the All2All scheme requires repeating KV. When the data layout is typically sbh or sbnd, repeating along the h dimension results in non-contiguous addresses, leading to operator efficiency issues and requiring additional operations such as transpose. In contrast, AllGather operates directly on the s dimension with contiguous addresses, requiring no extra operations.

3. In Ulysses CP, when repeat operations occur, the Key and Value tensors passed into the attention backward pass are expanded by a factor of CP in memory compared to the Key and Value tensors before the repeat. This leads to increased memory consumption and may cause out-of-memory situations.

## Solution

1. Add a KV cache feature on top of Ring Attention long-sequence parallelism. You can choose to cache all K and V, cache only K, or set up a hierarchical cache. During the forward pass of long-sequence parallelism, the KV tensors received in the forward computation are retained until the backward computation, allowing gradient results to be computed directly and reducing communication time.

2. For GQA models with a single head, add an AllGather KV + All2All Q approach on top of Ulysses Attention long-sequence parallelism. This reduces repeat operations and the overhead of non-contiguous memory operations such as transpose, thereby improving training performance.

3. Add a KV cache feature to the All2All and AllGather approaches in Ulysses. You can choose to cache all K and V, cache only K, or set up a hierarchical cache. During the forward pass, the KV tensors before communication are cached and retained until the backward pass, where communication is re-executed for computation, saving memory. The All2All approach can only enable KV caching when repeat operations are performed.

### Approach

1. In the Ring scheme, the sequence is split into CP parts for parallel computation, with each rank computing its own K and V while simultaneously sending and receiving K and V from other ranks. For example, rank 0 sends K0/V0 and K7/V7 to the "downstream" rank while receiving K3/V3 and K4/V4 from the "upstream" rank. Each card repeats the same action CP-1 times, so that ultimately each split sequence can "attend to" the global KV and compute the complete attention result. The backward computation logic follows the same principle: initially each rank has its own KV, and after computing its own gradient, the subsequent steps send the received K and V chunks along with dK and dV to other ranks, while simultaneously receiving K, V chunks, dK, and dV chunks from other ranks, using the received K and V as input to compute and update gradients, thereby achieving overlap of computation and communication.
A key point in the backward process is that inter-rank communication requires sending four data blocks—K, V, dK, and dV—for a total of CP-1 times. Since K and V are already sent and received sequentially among ranks during the forward pass, caching K and V during the forward pass will halve the backward communication time. When CP is relatively large, caching all K and V increases memory pressure. By supporting caching only a portion of K and V, or caching once every N layers, flexible configuration on demand is enabled.

2. In a GQA model with a single head, the AllGather KV communication method is used to replace the original Repeat-All2All KV method for obtaining the full sequence, while the All2All scheme is still used for Q.

3. In the Ulysses scheme, the KV before the Repeat-All2All or AllGather communication in the forward pass is cached and carried to the backward pass, and the post-communication KV is used for computation to ensure correctness. In the backward pass, when the KV from before the Repeat-All2All or AllGather communication is retrieved, the KV undergoes Repeat-All2All or AllGather re-communication for gradient computation. Since re-communication incurs performance loss, a portion of K and V can be cached, or caching can be performed once every N layers, flexibly combined to achieve optimal performance within memory constraints.

The flexible caching strategy is as follows:

1. Support configuring the layer interval for caching K and V: partial caching of K and V can be achieved by caching across different layers, controlled by adding a parameter `interval` to specify the interval of layers to cache. For example, when `interval=1`, K and V will be cached in layers numbered 0, 2, 4, ..., and so on. The cache interval supports starting from 0 and must not exceed the number of layers on the rank. The default value of the interval is 0.

2. Support caching a portion of K and V: on each layer, caching only K (K and V have the same size) is supported. This method is controlled by a parameter. When the parameter value is `half`, only K is cached; when configured as `full`, both K and V are cached. The default is to cache both K and V. This configuration and the layer interval-based cache configuration can be enabled simultaneously. The caching effects after configuration are cumulative and do not conflict with each other.

## Application Scenario

When long sequence parallelism is enabled during training.

`FlashAttention` is required, and it is enabled by default.

To benefit from the KV cache in Ring Attention, the computation time must be less than the communication time. Theoretically, the sequence length assigned to each computation block must satisfy `c < F/B`, where `F` is the FLOPS of each device and `B` is the bandwidth between devices.

In Ulysses Attention, to benefit from AllGather KV + All2All Q, a GQA model is required, and the communication volume must be the same, meaning KV has only one head.

In Ulysses Attention, to benefit from the KV cache, the Repeat-All2All approach requires the use of repeat to achieve memory savings, while AllGather KV + All2All Q can achieve memory savings as soon as CP is enabled.

## Usage

| Important Parameter                                           | Parameter Description                                                     |
|------------------------------------------------|----------------------------------------------------------|
| --context-parallel-kv-cache-policy [full/half] | Enables KV caching and sets its level during CP forward computation. Default is full, which caches both K and V; half caches only K.                   |
| --context-parallel-cache-interval [int]        | Sets the layer interval for KV caching during CP forward computation. Default is 0, meaning every layer is cached. Configure based on user requirements. |
| --use-ulysses-allgather-kv                     | Enables the AllGather scheme for Ulysses Attention. Default is False, meaning it is not enabled.           |

## Application Effects

In scenarios where computation time cannot hide communication time in Ring Attention, enabling the KV cache feature will shorten training time and improve training performance, but it will increase memory usage.

Enabling AllGather KV in Ulysses Attention will, in permissible scenarios, shorten training time and improve training performance.

Enabling KV cache in Ulysses Attention will reduce memory usage when Repeat is applied in Repeat-All2All, but it will cause performance degradation. In the AllGather case, memory usage will also be reduced, but it will cause performance degradation.

## Notes

1. When enabling `--context-parallel-kv-cache-policy`, context parallelism must also be enabled; otherwise, the feature is not supported.
2. When enabling `--context-parallel-cache-interval`, `--context-parallel-kv-cache-policy` must also be enabled, and the interval value must be less than the number of layers; otherwise, the feature is not supported.
3. When enabling `--use-ulysses-allgather-kv`, context parallelism must be enabled, `--context-parallel-algo` must be set to `ulysses_cp_algo`, `--group-query-attention` must be enabled, and the number of KV heads per rank must be 1; otherwise, the feature is not supported.
4. When both `--context-parallel-kv-cache-policy` and `--context-parallel-algo ulysses_cp_algo` are enabled, the KV repeat operation must be performed. Otherwise, the feature is not supported.
