# MoE Token Permute and Unpermute 融合优化

## 背景与挑战
在MoE架构中，MoEAlltoAllTokenDispatcher调度器负责将token令牌分配给各个专家进行处理，并将处理后的结果重新组合回原始的token顺序。这个过程通常涉及到以下步骤：
Token路由：确定每个token应该由哪个专家处理。这可以通过专家门控机制（gating mechanism）来完成，门控机制为每个token选择最适合的专家。
数据重排(Permute)：将token按选择的专家进行分组，以便每个专家可以并行处理属于它的token。这通常涉及到对token的重排操作。
专家处理：每个专家并行处理属于它的token。
结果重组(Unpermute)：处理完成后，需要将来自不同专家的结果重组回原始的token顺序。
在上述流程中，数据重排和结果重组步骤是性能瓶颈之一。这是因为这两个步骤涉及到大量的数据移动，特别是在使用分布式训练时。
## 解决方法
为了优化这一过程，MindSpeed将MoE Token Permute和Unpermute操作分别融合成一个算子，提升模型训练性能。

## 使用方法
1. 启动脚本添加`--moe-permute-fusion` 或  `--use-fused-moe-token-permute-and-unpermute`。两者等价，但推荐优先使用`--moe-permute-fusion`。
2. 建议如下配置获得最佳性能，否则某些场景开启该融合算子可能性能劣化。
(1)`--moe-token-dispatcher-type alltoall`时, 设置`--expert-tensor-parallel-size 1` 
(2)`--moe-token-dispatcher-type alltoall_seq`时, 开启`--moe-tp-extend-ep`

## 使用限制
1.支持的调度器类型：当前仅支持 `--moe-token-dispatcher-type alltoall` 和 `--moe-token-dispatcher-type alltoall_seq`。暂不支持 `--moe-token-dispatcher-type allgather`。
2.融合算子与专家容量参数的兼容性：若要启用 `--moe-expert-capacity-factor`，必须同时开启 `--moe-pad-expert-input-to-capacity` 方可兼容融合算子。仅开启 `--moe-expert-capacity-factor` 而未开启 `--moe-pad-expert-input-to-capacity` 时，暂不兼容该融合算子。
3.系统环境要求：仅限于版本标识为 `CANN 8.3.RC1` / `PTA 7.2.RC1 `及其后续所有迭代版本的系统环境。

## 使用效果 
启用融合算子后，不仅能够有效节省内存资源，还能提升模型训练性能。