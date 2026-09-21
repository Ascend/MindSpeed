# VPP 前向 P2P 发送等待延后

## 功能与原理

Megatron Core 0.18 的交错流水线调度在开启输出伪释放时，会在释放前等待当前
前向发送完成。这一步可能提前建立通信到计算的依赖，压缩 P2P 发送与后续计算
的重叠窗口。

MindSpeed 的 `defer-p2p-send-wait` 将当前发送的等待延后到下一次受管理的
前向发送提交之后；最后一笔发送在本次调度退出时收尾。接收等待仍由原调度
负责，计算顺序、通信内容和通信次数保持不变。

该优化面向 NPU 上的原生 VPP 交错调度，主要目标是改善通信与计算重叠，
不承诺降低峰值显存，也不属于 MoE fast ops 开关控制的显存优化。

## 使用方式

`--defer-p2p-send-wait` 默认开启，属于 optimization level 2 特性。
实际生效需要 `optimization_level >= 2`，并满足下文全部适用条件。
显式添加该参数不会绕过兼容性校验；使用 `--no-defer-p2p-send-wait` 关闭。

以下是添加到已有合法训练命令中的 PP/VPP 配置示例，其余模型、并行和训练
参数仍需按实际任务设置：

```bash
--optimization-level 2 \
--num-layers 16 \
--pipeline-model-parallel-size 2 \
--num-layers-per-virtual-pipeline-stage 4
```

此例每个 PP rank 有两个虚拟流水线阶段。不要同时设置另一种 VPP/layout
指定方式，也不要添加 `--no-overlap-p2p-communication`。最终配置还需满足
`batch_p2p_comm=False`、`deallocate_pipeline_outputs=True`。

进行开关对比时，保持原配置，关闭组只增加：

```bash
--no-defer-p2p-send-wait
```

## 适用条件与自动回退

完整参数校验后，只有以下条件全部满足才保留开启状态：

| 条件 | 要求 |
| --- | --- |
| 流水线并行 | `pipeline_model_parallel_size > 1` |
| 虚拟流水线 | 推导出的 `virtual_pipeline_model_parallel_size > 0` |
| P2P 通信重叠 | `overlap_p2p_comm=True` |
| 批量 P2P | `batch_p2p_comm=False` |
| 输出伪释放 | `deallocate_pipeline_outputs=True` |
| 调度选择 | `schedules_method=None`，使用原生交错调度 |
| 不兼容特性 | 下列开关均未启用 |

不兼容开关为：

- `moe_fb_overlap`
- `use_multiparameter_pipeline_model_parallel`
- `recompute_in_bubble`、`recompute_in_advance`
- `tp_2d`
- `variable_seq_lengths`
- `optimize_send_recv_comm`
- `dist_train`

条件不满足时，校验将 `args.defer_p2p_send_wait` 自动设置为 `False`，
继续走原有调度。PP/VPP 数值允许使用可转换的数字字符串，无法转换时也关闭。
例如，PP2 但未开启 VPP 的配置不会应用此优化。

补丁安装早于完整参数校验。注册时只判断用户开关，避免把尚未推导的 VPP
或早期字符串参数误判为不兼容。每次调度入口读取一次最终开关；关闭后直接
调用原调度，不接管发送。

运行中，未处于受管理的调度、未开启 P2P overlap、没有待发送输出、输出不是
单个 NPU Tensor，或通信配置使用 batch P2P / 关闭输出伪释放时，该次调用
直接走原通信接口。

自动回退仅处理这些预先可判断的适用条件，不会吞掉原训练配置的错误，也不会
在发送已经提交后捕获异常并重新执行通信。

## 发送存储的生命周期

Megatron 的输出伪释放会替换输出张量的 `.data`。仅保存同一个 Tensor 的
Python 引用不能保护原发送存储，因此实现按以下顺序管理异步发送：

1. 提交前，通过 `detach()` 创建独立的 Tensor 元数据并持有原 storage；
   不复制发送载荷，也不额外持有其 autograd 图。
2. 调用原 `send_forward_recv_forward`，接管返回的 `send_next` Work。
   只从交还调度器的 handles 中移除这一项，保留全部接收等待。
3. 当前发送提交后，对前一笔受管理发送执行 `wait()`，再释放对应持有者。
   正常稳定运行时保留最新一笔发送；提交新发送到前一笔收尾之间会短暂持有两笔。
4. `Work.wait()` 可能只向当前 NPU stream 加入事件依赖，因此在丢弃持有者前
   对发送张量调用 `record_stream()`，保护存储直到相关流上的工作完成。
5. 调度正常退出、forward-only 返回或发生异常时，通过 `finally` 收尾剩余发送，
   并恢复调度上下文。

正常路径不新增载荷拷贝、通信调用或全设备同步。若提交过程抛出异常，未能返回
Work，则清理路径使用设备同步确认存储安全；若等待或清理失败，错误继续传播，
相关存储保留到进程退出。原接口没有返回预期的 `send_next` Work 时也明确报错。

## 验证与性能观察

对比时固定模型、PP/VPP 布局、microbatch 数量、精度、重计算配置和随机种子，
分别运行默认开启组与显式关闭组：

- 检查最终参数中 `defer_p2p_send_wait` 的值，确认优化实际生效。
- 对比 loss、梯度或参数更新，确认训练结果符合所用精度的容差。
- 预热后比较稳态 step time 和吞吐；在时间线中观察前向发送与后续计算的重叠。
- 同时记录 peak allocated / reserved。延长发送存储的存活时间可能提高局部
  显存占用，不能仅凭等待被延后推断显存下降或吞吐一定提升。

收益取决于流水线布局、通信时长和可重叠的计算量，需要实际多卡 NPU 验证。

## 实现位置

- 参数与兼容性校验：
  [`defer_p2p_send_wait.py`](../../../mindspeed/features_manager/pipeline_parallel/defer_p2p_send_wait.py)
- 调度包装与发送存储管理：
  [`defer_p2p_send_wait.py`](../../../mindspeed/core/pipeline_parallel/defer_p2p_send_wait.py)

实现由 MindSpeed 安装在 Megatron 的调度和 P2P 接口上，不需要在 TENPU 重复
安装补丁。
