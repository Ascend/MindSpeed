# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

# NOTE patch import begin
# NOTE patch import end

import contextlib
from typing import Iterator, List, Union
from functools import lru_cache
from torch.autograd.variable import Variable
from mindspore.ops import composite as C
from mindspore.common.api import _pynative_executor

from megatron.core.pipeline_parallel.schedules import (
    set_current_microbatch,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    get_tensor_shapes,
    send_forward,
    recv_forward,
    send_forward_recv_backward,
    send_backward,
    send_backward_recv_forward,
    recv_backward,
    get_pp_rank_microbatches,
    get_schedule_table
)

from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.enums import ModelType
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)
import torch


def deallocate_output_tensor_(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.untyped_storage().resize_(0)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )