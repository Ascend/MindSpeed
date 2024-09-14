# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import torch


def start_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operation
    for this bucket.

    When overlap_grad_reduce is set to True, dispatches an asynchronous
    communication call. When overlap_grad_reduce is set to False, makes
    synchronous call.
    """
    assert (
            self.communication_handle is None and not self.communication_issued
    ), 'Should not have multiple communication calls in flight at once'

    # Make sure norm of grads in bucket are not NaN
    # prior to data-parallel all-reduce / reduce-scatter.
    if self.check_for_nan_in_grad:
        global_rank = torch.distributed.get_rank()
        norm = self.grad_data.norm(p=2)
        assert not norm.isnan(), (
            f'Rank {global_rank}: found NaN in local grad norm in '
            f'backward pass before data-parallel communication collective. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    self.grad_data *= self.gradient_scaling_factor
    # Use async_op only when overlap_grad_reduce is True.
    self.communication_handle = torch.distributed.all_reduce(
        self.grad_data, group=self.data_parallel_group, async_op=self.overlap_grad_reduce
    )
    self.communication_issued = True


def pipe_register_grad_ready(self, param: torch.nn.Parameter):
    assert (self.overlap_grad_reduce), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
    from mindspeed.moe.pipe_experts import FLAG_GRAD_REDUCE
    if self.is_last_microbatch and FLAG_GRAD_REDUCE:
        bucket = self.param_to_bucket[param]
        bucket.register_grad_ready(param)
