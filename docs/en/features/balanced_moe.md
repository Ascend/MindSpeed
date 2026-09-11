# Dynamically Balanced Expert Parallelism (Data-Parameter Mutual Search)

## Background and Challenges

Mixture of Experts (MoE) models scale model capacity by routing input tokens to different expert networks for computation. In practice, however, factors such as input data distribution, initial router preferences, and concurrent request randomness often lead to severe computation and communication load imbalance across expert nodes. Some experts become bottlenecks due to overloading, while others remain underutilized. This imbalance significantly reduces training efficiency and overall system throughput, posing a key challenge for large-scale MoE training.

## Solution

This feature designs a dynamic and adaptive load balancing mechanism for expert parallelism. Its core idea combines "data finding computation" with "computation finding data," achieving mutual optimization of computation and parameters through dynamic decision-making to attain load balancing. The specific implementation includes the following key innovations:

- Real-time monitoring and decision-making: During the forward propagation process, global token distribution information is obtained through the router.
- Dynamic hot expert selection: Based on global load information, the subset of experts with the heaviest load is dynamically identified as "hot experts".
- Parameter broadcast and sharing: Through cross-node broadcast operations, it is ensured that all expert parallelism (EP) nodes hold the latest copies of hot expert parameters, achieving parameter sharing.
- Token distribution optimization: The token distribution strategy is restructured to convert tokens originally destined for remote hot experts into local computation, thereby reducing communication overhead.
- Computation-communication overlap and pipeline masking: In pipeline parallelism, the forward and backward computation phases of different micro-batches are finely scheduled to achieve complete masking of expert parallelism communication.

This solution employs a dynamic decision-making mechanism that combines "data finding computation" and "computation finding data" to achieve intelligent mutual discovery between computation and parameters, ultimately attaining load balancing and improving system throughput and resource utilization.

## Use Scenario

This feature is applicable to the following scenarios. (**It is strongly recommended to enable it only when both of the following conditions are met simultaneously. Otherwise, you may not achieve performance gains or may even experience performance degradation.**)

- Large-scale EP training
    - EP ≥ 32: The benefit of this feature is positively correlated with the scale of expert parallelism. When the EP value is small, the degree of load imbalance is relatively limited, and the additional overhead introduced by enabling this feature, such as parameter broadcasting, may outweigh its benefits. When the EP value is large, the load imbalance problem is typically more pronounced, and the performance improvement achieved through optimization with this feature is also more significant.

- Significant load imbalance
    - During actual training, dynamic changes in input data distribution cause some experts to receive far more tokens than others, creating obvious computation and communication hotspots. In such scenarios, redistributing computation tasks through dynamic load balancing yields the greatest benefits.

Scenarios not recommended or with limited benefits:

- Small EP configurations (≤ 8) with relatively uniform load: The additional communication overhead may offset the benefits of load balancing.
- Configurations that fail to meet any of the usage restrictions described below.
- Scenarios requiring strictly deterministic training: This feature involves dynamic decision-making and asynchronous communication, and does not guarantee binary alignment or complete determinism.
- In the DSKv3 reduced-layer 4-node Atlas A3 scenario, the loss error is less than 1%. For small-scale model scenarios, errors caused by accumulation order may be amplified, so this feature is not recommended.

## Usage

Users can enable the expert-parallel dynamic load balancing algorithm by adding the following parameters to the training script:

```bash
--balanced-moe-experts --balanced-moe-hot-expert-num N --trans-hot-expert-group-num M
```

**Usage Restrictions**

To ensure the stable operation and performance of EP dynamic load balancing, the following usage restrictions and configuration requirements must be strictly observed:

- Dependency feature requirements
    - `--moe-fb-overlap` must be enabled simultaneously.
    - `--moe-grouped-gemm` must be enabled to support GroupedMatmul.

- Dispatcher support
    - Currently only the `--moe-token-dispatcher-type=alltoall` dispatcher type is supported.
    - `allgather` and `alltoall_seq` dispatcher types are not supported.

- Parallel configuration restrictions
    - `--expert-tensor-parallel-size=1` needs to be set. Expert tensor parallelism is not supported.
    - `--overlap-grad-reduce` must be disabled; asynchronous data-parallel communication hiding is not supported.

- Memory and model configuration
    - Only `--moe-zero-memory=level0` is supported for memory optimization.
    - `moe-zero-memory-num-layers` is not supported.
    - Only Mcore architecture models are supported. Ensure that `--use_legacy_models` is disabled.

- MoE mode restrictions
    - Only Dropless mode is supported; Megatron MoE's Token Drop & Pad functionality is not supported.
    - It is not recommended to enable `--swap-attention` simultaneously, as it may cause performance degradation.

- Pipeline parallel restrictions
    - When using Virtual Pipeline Parallel (VPP), the following conditions must be met:
        - The global batch size (GBS) must satisfy: GBS > 1 × Data Parallelism (DP) × Pipeline Parallelism (PP) × Micro Batch Size (MBS).
        - If noop layers are used, they must be placed at the last VPP stage at the tail of the model.

- Conflicts
    he following features conflict with the EP dynamic load balancing algorithm and cannot be enabled simultaneously:
    - `--moe-alltoall-overlap-comm`
    - `--moe-hierarchical-alltoallv`
    - `--recompute-in-advance`
    - `--recompute-in-bubble`

- Parameter configuration restrictions
    - `--balanced-moe-hot-expert-num` must be less than or equal to the number of local experts per EP rank (`num_experts / expert_model_parallel_size`).
    - `--trans-hot-expert-group-num`  must satisfy `1 ≤ M ≤ N`, where `N` is the value of`--balanced-moe-hot-expert-num`.

- Non-deterministic behavior notice
    **Important: This feature is not binary-aligned.**
    - Fundamental change in computation path:
        - Forward propagation path changes: After enabling this feature, tokens that were originally sent to remote hot experts are computed locally, fundamentally altering the execution path of the computation graph.
        - Gradient accumulation order changes: Gradients for hot experts are computed in parallel across multiple EP ranks and asynchronously reduced, altering the gradient accumulation order.
    - Impact of non-associative floating-point arithmetic:
        - Due to the non-associative nature of floating-point addition, changing the distribution order of tokens across different devices leads to slight differences in intermediate results during forward and backward propagation.
        - Even mathematically equivalent operations like `(A+B)+C`  and `A+(B+C)` may yield different results in floating-point representation.
    - Specific manifestations of non-binary alignment:
        - Inconsistent across runs: Even with the deterministic computation flag `--npu-deterministic` enabled, running the same training twice may not produce bit-level identical results.
    - Applicable scenario restrictions:
        - Absolutely not applicable: Scenarios that require strict bit-level reproducibility.

**Parameter Description**

- `--balanced-moe-experts`: Master switch for enabling the dynamic load balancing algorithm.
- `--balanced-moe-hot-expert-num <N>`: Specifies the number of hot experts `N` (a positive integer) dynamically maintained per layer. Based on this parameter and the real-time load status, the system selects the `N` experts with the highest load for parameter broadcast and load balancing scheduling.
  - Default value: `3`
  - Important recommendations and restrictions:
    1. `N` must be less than or equal to the number of local experts on each EP rank. The formula for calculating the number of local experts is: `num_local_experts = num_experts / expert_model_parallel_size`. For example, if the total number of experts is 64 and the EP degree is 8, each rank has 8 local experts, and the maximum value for `--balanced-moe-hot-expert-num` can only be set to `8`.
    2. Setting `N` too large (exceeding the number of local experts) not only provides no additional benefit, but also increases unnecessary parameter broadcast overhead and memory usage, potentially degrading performance. It is recommended to set `N` based on the actual degree of load imbalance (typically between 3 and 8) to strike a balance between overhead and benefit.

- `--trans-hot-expert-group-num <M>`: Specifies the number of groups `M` (a positive integer) when transmitting hot expert parameters. This controls the concurrency of hot expert parameter broadcast, helping to reduce communication time.
  - Default value: `3`
  - Important restriction: `M` must satisfy: `1 ≤ M ≤ N` (where `N` is the value of `--balanced-moe-hot-expert-num`).

## Effects

In an expert-parallel training environment that meets the recommended usage scenarios (large EP scale and load imbalance) and is correctly configured enabling this feature can achieve the following optimization effects:

- Load balancing optimization: Through the "computation-parameter mutual seeking" mechanism, the computational load is dynamically migrated from overloaded "hot experts" to idle experts, balancing the computational load across expert nodes and alleviating system bottlenecks.
- Effectively improved system throughput: By converting remote hot expert tokens that originally required cross-device communication into local computations, the data volume of All-to-All communication is significantly reduced. Meanwhile, fine-grained computation-communication overlap and pipeline scheduling effectively mask the latency of newly introduced communications such as parameter broadcasting, thereby accelerating training iteration speed.
- Improved hardware resource utilization: It reduces the time during which some devices are overloaded while others remain idle due to load imbalance, enabling all computing units to be utilized more fully and improving overall cluster efficiency.

Typical performance gains: Under conditions of the recommended scenarios (EP ≥ 32, significant load imbalance) and with proper configuration, this feature can deliver substantial end-to-end training throughput improvement. Conversely, if the recommended scenarios are not met or the configuration is improper, it may fail to yield gains or even cause performance degradation.
