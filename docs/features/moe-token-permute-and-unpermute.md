# MoE Token Permute and Unpermute 融合优化

## 背景与挑战
在MoE架构中，MoEAlltoAllTokenDispatcher调度器负责将token令牌分配给各个专家进行处理，并将处理后的结果重新组合回原始的token顺序。这个过程通常涉及到以下步骤：

* token路由：确定每个token应该由哪个专家处理。这可以通过专家门控机制（Gating Mechanism）来完成，门控机制为每个token选择最适合的专家。
+ 数据重排（Permute）：将token按选择的专家进行分组，以便每个专家可以并行处理属于它的token。这通常涉及到对token的重排操作。
* 专家处理：每个专家并行处理属于它的token。
+ 结果重组（Unpermute）：处理完成后，需要将来自不同专家的结果重组回原始的token顺序。

在上述流程中，数据重排和结果重组步骤是性能瓶颈之一。这是因为这两个步骤涉及到大量的数据移动，特别是在使用分布式训练时。
## 解决方法

为了优化这一过程，可以考虑将数据重排和结果重组步骤合并成一个操作。MindSpeed将MoE Token Permute和Unpermute操作分别融合成一个算子，提升模型训练性能。算子接口分别见[link](../ops/npu_fused_moe_token_permute.md),[link](../ops/npu_fused_moe_token_unpermute.md)。

## 使用场景
适用于使用MoE架构且moe-token-dispatcher-type设置为alltoall的场景，特别是在需要高效利用计算资源、加快模型训练速度以及优化内存使用的情况下。此优化能够减少数据移动开销，提高计算资源利用率，显著缩短模型训练时间，并通过减少中间结果存储来降低内存占用。

## 使用方法

设置如下参数即可调用该融合算子。
`--use-fused-moe-token-permute-and-unpermute`

##### 注意：
使能MoE Token Permute and Unpermute融合算子前需要开启专家并行，并且配置以下参数：

`--moe-token-dispatcher-type alltoall`
`--expert-model-parallel-size M`
`--num-experts N `

## 使用效果 
启用融合算子后，不仅能够有效节省内存资源，还能提升模型训练性能，在类DeepSeekV2万亿参数级别的MoE模型下，性能提升10%~15%。
