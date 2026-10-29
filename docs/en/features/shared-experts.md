# Shared Experts

## Solution Introduction

Large language models face significant computational cost issues when scaling parameter sizes. With the evolution of Mixture of Experts (MoE) models, the concepts of routed experts and shared experts have emerged. For routed experts, input data passes through a routing module that selects experts with higher probabilities for computation; for shared experts, input data does not need to go through the routing module—all data is processed by the shared experts. The computation results of routed experts and shared experts are summed to produce the final output of the MoE module.

## Solution

By combining shared experts with routed experts, the MoE model can attend to both the commonality and specificity of input data across different input scenarios, thereby improving the model's generalization capability.

Some experts are designated as shared experts to capture and integrate common knowledge across different contexts, thereby reducing parameter redundancy in other routed experts.

For more information about shared experts, see [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066).

### Figure 1 Shared expert isolation

![img](../figures/shared-experts.png)

## Application Scenario

**Backend limitation:** When combined with Grouped GEMM, Alltoall communication overlap, or FB overlap, `--transformer-impl transformer_engine` must be set.

Use the native Megatron MoE.

## Usage

Shared expert-related commands and parameter descriptions:

| Command Parameter                     | Description                   |
|--------------------------|------------------------|
| `--moe-shared-expert-intermediate-size N` | Intermediate hidden layer size of shared experts |

## Effects

By enabling shared experts, parameter redundancy among other routed experts can be reduced, parameter efficiency can be improved, and each routed expert can be ensured to focus on unique aspects.
