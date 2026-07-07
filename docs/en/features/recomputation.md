# Megatron Recomputation

## Background and Challenges

In the training pipeline of large models, traditional practice requires storing the activations generated during the forward propagation phase for subsequent gradient computation during backpropagation. This requirement causes the number of saved activations to grow linearly with model depth, significantly increasing the pressure on hardware memory resources.

## Solution

To address the above challenges, a recomputation strategy is proposed. Specifically, during the forward propagation and loss function computation phases, the memory space of activations that are no longer needed is released immediately, and activations are only recomputed as needed during backpropagation. By effectively shortening the lifecycle of activations, this method significantly reduces the memory burden and improves overall resource utilization efficiency.

## Application Scenario

The recomputation feature can be enabled when GPU memory is insufficient, and it is divided into the following two modes:

- Selective recomputation (recommended): Focuses on recomputing the core_attention component within the Transformer architecture. This strategy retains activations that occupy little memory but are expensive to recompute, while applying activation recomputation to those that occupy large amounts of memory but are relatively cheap to recompute. This approach achieves efficient memory management while maintaining model performance.

- Full recomputation: Suitable for extreme environments where memory resources are critically limited. In this mode, except for the input data, all activations are recomputed when needed, minimizing memory dependency to the greatest extent.

## Usage

- Selective recomputation:
`--recompute-activations   #Enable selective recomputation`.

- Full recomputation:
`--recompute-granularity full    #Enable full recomputation`
`--recompute-method uniform/block    #Specify the recomputation method`

`--recompute-method` can be set to either `uniform` or `block`:

- `--recompute-method uniform`: Divides Transformer layers evenly into groups (each group size is `--recompute-num-layers`), storing inputs and activations per group.

- `--recompute-method block`: Recomputes the first `--recompute-num-layers` Transformer layers, while the remaining layers are not recomputed.

### Notes

- When `--recompute-activations` and `--recompute-granularity full` are configured together, selective recomputation takes effect.

- When the script is configured with `--recompute-method block`, `--recompute-granularity full`, and `--num-layers-per-virtual-pipeline-stage N`, users can configure how many layers each VPP stage recomputes via the `--recompute-num-layers N` parameter. The parameter `--enable-recompute-layers-per-pp-rank` can be used to modify the semantics of `--recompute-num-layers N` in this scenario, where the new semantics indicate that VPP is ignored and the number of recomputed layers is configured per PP stage.

- Note: In the legacy branch, enabling `--use-flash-attn` will prevent the use of selective recomputation.

## Application Effects

Adopting recomputation brings a clear trade-off:

- Memory benefit: Avoids long-term retention of intermediate activations, significantly reducing memory usage.

- Performance cost: Recomputation adds extra computation, which may hurt training throughput. Thus, recomputation should be carefully configured based on available memory and performance requirements.
