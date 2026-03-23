# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""
Fix for DDP initialization running on a non-default NPU stream.

Root cause (Megatron 0.15 only):
  megatron/training/training.py wraps the entire DDP model construction inside:
      with torch.cuda.stream(torch.cuda.Stream()):
          model = [DP(...) for ...]
  This is intended for CUDA graph support on NVIDIA GPUs. On NPU (via torch_npu's
  torch.cuda -> torch.npu aliasing), a new temporary NPU stream (StreamA) is created.
  DDP.__init__ calls param.expand_as(param) for every parameter while StreamA is
  active, which on NPU associates each AccumulateGrad node with StreamA.

  When StreamA's Python object is garbage-collected after the with-block exits, the
  NPU driver recycles the stream handle. HCCL then allocates its internal communication
  stream and receives the same recycled handle. Now the AccumulateGrad nodes are
  effectively bound to the HCCL stream.

  During backward, when AccumulateGrad fires the DDP hook, NPU's autograd engine
  switches to the associated stream (= HCCL stream). The hook's param.main_grad.add_()
  and grad_weight = torch.empty(...) therefore run on the HCCL stream, not the default
  stream. grad_weight's memory is held by the HCCL allocator until the stream drains,
  delaying deallocation and increasing peak memory.

Fix:
  Replace torch.cuda.stream with contextlib.nullcontext during get_model execution,
  so DDP is always initialized on the current (default) stream. CUDA graph support
  is not affected on NPU because NPU uses a different graph mechanism.
"""
from contextlib import nullcontext
from functools import wraps

import torch


def get_model_wrapper(get_model_func):
    """Patch megatron.training.training.get_model to skip non-default stream for DDP init."""

    @wraps(get_model_func)
    def wrapper(*args, **kwargs):
        original_stream_ctx = torch.cuda.stream
        # Replace torch.cuda.stream with a no-op so DDP is initialized on the
        # default stream.  torch.cuda on NPU is aliased to torch.npu by torch_npu,
        # so this also covers torch.npu.stream calls originating from the same module.
        
        def noop_stream_ctx(s):
            return nullcontext()

        torch.cuda.stream = noop_stream_ctx
        try:
            return get_model_func(*args, **kwargs)
        finally:
            torch.cuda.stream = original_stream_ctx

    return wrapper
