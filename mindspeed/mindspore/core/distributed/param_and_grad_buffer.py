# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch


def register_grad_ready(self, param: torch.nn.Parameter):
    """
    Registers grads for the passed-in param to be "ready" for grad sync.

    When the number of microbatches is greater than 1, we only want to register
    grads as ready when processing the last microbatch and overlap_grad_reduce is True.
    """
    assert param in self.params, 'Param is not in the bucket'
    assert param not in self.params_with_grad, 'Cannot set grad twice'
    if param in self.params_with_grad:
        return
    assert (
        self.ddp_config.overlap_grad_reduce
    ), 'register_grad_ready() should be called only when overlapping grad reduce'
    self.params_with_grad.add(param)
    # If all params in bucket have grads available, issue communication call.
    if len(self.params_with_grad) == len(self.params):
        self.start_grad_sync()