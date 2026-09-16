# Megatron 全分片数据并行（Fully Sharded Data Parallel, FSDP）

## 背景与挑战

随着大模型权重增加，需要进一步提高显存利用效率。ZeRO-1 只在 DP 域内切分优化器状态，模型权重和梯度仍占用较多常驻显存。FSDP 进一步切分权重和梯度，以降低每个 DP rank 的常驻训练状态占用。

本文介绍 `core_r0.18.0` 中 Megatron FSDP 的基础适配，不是 PyTorch FSDP2。

## 解决方案

在 `optim_grads_params` 策略下，每个 DP rank 常驻参数分片。前向和反向计算需要完整参数时，在 DP 域内执行 All Gather；计算结束后按上游缓冲区生命周期释放不再需要的完整参数。反向梯度通过 Reduce Scatter 归约并切分，各 rank 使用本地梯度分片更新对应参数。

0.18 适配沿用新的梯度 bucket 归约流程，保留通信 dtype 转换、双缓冲和异步释放逻辑。HCCL reduce-scatter 使用独立输出存储；SUM 路径在通信前显式缩放一次，不依赖 NCCL premultiplied SUM。MoE router 沿用 0.18 实现，保留 bias 和 router dtype 支持。

## 使用场景

DP 大于 1，模型权重、梯度和优化器状态占用较多显存，希望进一步切分这些训练状态。

## 使用方法

1. 开启 Megatron FSDP，加入以下配置：

    ```bash
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --ckpt-format fsdp_dtensor
    ```

    仍兼容 master 的 `--use-custom-fsdp`，在 Megatron 参数校验前将其映射到 `--use-megatron-fsdp`。0.18 的检查点默认格式是 `torch_dist`，Megatron FSDP 必须显式使用 `fsdp_dtensor`；修改格式参数不会自动转换已有检查点。

2. 在加载环境脚本后取消 `CUDA_DEVICE_MAX_CONNECTIONS` 设置：

    ```bash
    unset CUDA_DEVICE_MAX_CONNECTIONS
    ```

3. 使用已有 MoE 用例中的 `--moe-grouped-gemm` 时，需要同时指定：

    ```bash
    --transformer-impl transformer_engine
    ```

4. 安装包含 DTensor FusedAdam 修复的 TransformerEngineNPU 版本，例如包含提交 [5c85c0d](https://gitcode.com/GuoHaifeng1999/TransformerEngineNPU/commit/5c85c0df3e9c1887a4aa5a000f96b2f01d9fb031) 的版本，并确认训练环境实际导入该安装包。旧版 TENPU 会拒绝 FSDP 提供的 DTensor 参数；仅调整检查点格式不能解决优化器类型错误。

## 支持范围与验证状态

| 项目 | 当前适配情况 |
|------|--------------|
| 基础 FSDP | 已接入 0.18 Megatron FSDP；已有脚本使用 `optim_grads_params` |
| 参数与梯度 | 配套 TENPU 支持非量化 FP32/FP16/BF16 DTensor 参数，以及普通梯度、独立梯度和空分片处理 |
| 优化器状态 | 配套 TENPU 支持浮点 moments，以及既有条件下的 FP32/FP16 master weights |
| 通信生命周期 | 保留上游通信 dtype、双缓冲和异步释放；需在目标 NPU 环境验收 |
| HSDP/HFSDP 与 AVG | 保留上游分支并适配独立 RS 输出，尚未完成多卡组合验证 |
| 量化与 remainder | 当前 TENPU DTensor 路径不支持量化模型参数、FP8/uint8 moments 或 INT16 parameter remainder |
| 检查点 | 接入已有 `fsdp_dtensor` 保存加载路径；真实多卡保存恢复及改变 DP 大小的恢复仍待验证 |
| 特性组合 | 不保证与仓内其他特性任意组合；已有 MoE 脚本的训练精度和性能也需在 Ascend 多卡环境验证 |

上述“已接入”表示源码完成基础适配，不表示所有组合已通过 NPU 端到端测试。TENPU 的 CPU 本地分片验证不能替代 HCCL 训练和真实检查点恢复验收。

## 使用效果

权重、梯度和优化器状态进一步分片后，可降低常驻训练状态的显存占用。按需收集参数会增加通信，独立 reduce-scatter 输出和通信 dtype 转换也会使用临时缓冲区；实际显存峰值与迭代耗时应在目标模型和并行配置下测量。

> [!NOTE]
>
> 复用 master 的已有 [custom_fsdp.sh 脚本](../../../tests_extend/system_tests/feature_tests/custom_fsdp.sh)，未新增测试用例。运行前需修改数据和 tokenizer 路径。
>
> 默认脚本在第 50 步退出，而保存间隔为 2000 步，且包含 `--no-load-optim` 和 `--no-load-rng`，不能用一次默认运行证明完整断点恢复。验证恢复时，应在同一已有用例中配置 save/load 目录和合适的保存间隔，并移除这两个不加载选项，比较恢复后的下一步参数、优化器状态和 loss。
