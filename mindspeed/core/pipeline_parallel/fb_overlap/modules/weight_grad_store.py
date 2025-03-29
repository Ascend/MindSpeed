# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import operator
import queue
from functools import reduce
import torch
import torch_npu

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size
)
from megatron.training import get_args
from mindspeed.ops.gmm import GMMFunction
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


def gather(input_slice, stream):
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input_slice.size())
    dim_size[0] = dim_size[0] * world_size

    all_gather_buffer = torch.empty(
        dim_size, dtype=input_slice.dtype, device=torch.cuda.current_device(), requires_grad=False
    )
    handle = None
    forward_event = torch.npu.Event()
    forward_event.record()
    with torch.no_grad():
        with torch_npu.npu.stream(stream):
            stream.wait_event(forward_event)
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input_slice, group=get_tensor_model_parallel_group(), async_op=True
            )

    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # gather is scheduled before the input gradient computation
    return all_gather_buffer, handle


class WeightGradStore:
    cache = []
    weight_grad_queue = queue.Queue()
    store_grad_cache = []
    grad_store = []
    gather_stream = None
    is_decoupleBlock = False

    @classmethod
    def put(cls, total_input, grad_output, weight, sequence_parallel, in_row=False):
        cls.cache.append((total_input, grad_output, weight, sequence_parallel, in_row))

    @classmethod
    def flush_chunk_grad(cls):
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def start_decouple(cls):
        cls.is_decoupleBlock = True

    @classmethod
    def end_decouple(cls):
        cls.is_decoupleBlock = False

    @classmethod
    def overlap_all_gather(cls):
        # used for grad_output all gather in RowParallel and input all gather in ColumnParallel.
        if len(cls.cache) > 0:
            [input_, grad_output_slice, weight, sequence_parallel, in_row] = cls.cache.pop(0)
            if not sequence_parallel:
                return (input_, grad_output_slice, weight, sequence_parallel, in_row), None
            if not in_row:
                total_input, handle = gather(input_, cls.gather_stream)
                grad_output = grad_output_slice
            else:
                grad_output, handle = gather(grad_output_slice, cls.gather_stream)
                total_input = input_
            return [total_input, grad_output, weight, sequence_parallel, in_row], handle
        else:
            raise Exception("All Gather empty queue.")


    @classmethod
    def overlap_matmul(cls, grad_store_cache):
        total_input, grad_output, weight, sequence_parallel, in_row = grad_store_cache
        args = get_args()
        if hasattr(weight, 'gmm_weight'):
            inputs, group_list, group_list_data_type = total_input
            if get_args().gemm_gradient_accumulation_fusion:
                npu_groupmatmul_add_fp32(inputs, grad_output, group_list, weight.main_grad)
            else:
                grad_weight = GMMFunction.builder.load().npu_gmm([inputs.t()], [grad_output], [], group_list, 2, 0)[0]
                weight.main_grad.data.add_(grad_weight.view(-1, weight.shape[-1]))
            inputs.untyped_storage().resize_(0)
            grad_output.untyped_storage().resize_(0)
        else:
            if len(grad_output.shape) > 2:
                grad_output = grad_output.contiguous()
                sb = grad_output.shape[0] * grad_output.shape[1]
                # Convert the tensor shapes to 2D for execution compatibility
                grad_output = grad_output.view(
                    sb, grad_output.shape[2]
                )
                total_input = total_input.view(
                    sb, total_input.shape[2]
                )
            if get_args().gradient_accumulation_fusion:
                import fused_weight_gradient_mlp_cuda
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            else:
                grad_weight = grad_output.t().matmul(total_input)
                weight.main_grad.data.add_(grad_weight)
            total_input.untyped_storage().resize_(0)
            grad_output.untyped_storage().resize_(0)


    @classmethod
    def pop(cls, overlap_arg=None):
        if len(cls.cache) == 0:
            return
        if cls.gather_stream is None:
            cls.gather_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
            
        (input_, grad_output_slice, weight, sequence_parallel, in_row), handle = cls.overlap_all_gather()
        if not sequence_parallel or get_args().moe_fb_overlap:
            grad_output = grad_output_slice
        else:
            grad_output, handle = gather(grad_output_slice, cls.gather_stream)
        cls.store_grad_cache = (input_, grad_output, weight, sequence_parallel, in_row)
        while len(cls.cache) > 0:
            if handle is not None:
                handle.wait()
            next_grad_cache, handle = cls.overlap_all_gather()
            cls.overlap_matmul(cls.store_grad_cache)
            cls.store_grad_cache = next_grad_cache
        if handle is not None:
            handle.wait()
        cls.overlap_matmul(cls.store_grad_cache)

        cls.store_grad_cache = None

    @classmethod
    def pop_single(cls):
        if cls.weight_grad_queue.empty():
            return

        cache_list = cls.weight_grad_queue.get()
        assert len(cls.cache) == 0
        cls.cache = cache_list
        cls.pop()