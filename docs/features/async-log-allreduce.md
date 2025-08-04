# 异步日志全归约 (Async Log Allreduce)

## 背景与挑战
在分布式训练场景中，传统的同步日志全归约操作会阻塞训练流程，等待所有进程完成损失值计算和通信。这种同步方式会导致以下问题：
1. 计算资源利用率不足，存在等待空闲时间
2. 训练吞吐量受限于最慢的进程
3. 日志记录成为性能瓶颈，影响整体训练效率

特别是在大规模分布式训练中，当使用大量NPU时，同步日志全归约的开销会变得显著，可能影响整体训练速度。

## 解决方案
Async Log Allreduce 特性通过以下方式解决上述挑战：
1. **异步通信**：使用非阻塞式(non-blocking)全归约操作，允许训练流程继续执行而不等待日志通信完成
2. **重叠计算与通信**：将日志相关的全归约操作与后续训练步骤重叠，提高硬件利用率
3. **延迟处理**：在需要实际记录日志时再同步通信结果，而不是立即等待

该特性通过`torch.distributed.all_reduce`的`async_op=True`参数实现异步通信，并配合特殊的损失值处理机制来确保数据一致性。

## 使用场景
该特性特别适用于以下场景：
1. 大规模分布式训练（数百至数千NPU）
2. 需要频繁记录训练指标的场景
3. 计算与通信需要高度重叠的场景
4. 对训练吞吐量敏感的应用

## 使用方法
1. 在启动bash脚本中添加参数`--async-log-allreduce`
2. 替换`pretrain_gpt.pt`中 loss_func函数为
```
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    allreduce_handle = torch.distributed.all_reduce(
        reporting_loss, group=mpu.get_data_parallel_group(), async_op=True
    )

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        ({'lm loss': (reporting_loss[0], reporting_loss[1])}, allreduce_handle),
    )

```


## 使用效果
启用Async Log Allreduce特性后可带来以下改进：
1. **训练吞吐量提升**
2. **资源利用率提高**
