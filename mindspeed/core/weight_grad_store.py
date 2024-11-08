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
    if get_args().use_nanopipe_swap:
        swap_event = WeightGradStore.swap_event.pop(0)
    with torch.no_grad():
        with torch_npu.npu.stream(stream):
            stream.wait_event(forward_event)
            if get_args().use_nanopipe_swap:
                stream.wait_event(swap_event)
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input_slice, group=get_tensor_model_parallel_group(), async_op=True
            )

    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # gather is scheduled before the input gradient computation
    return all_gather_buffer, handle


def swap_d2h(ori_tensor, stream):
    storage_size = ori_tensor.storage().size()
    tensor_cpu = torch.empty(ori_tensor.shape, dtype=ori_tensor.dtype, pin_memory=True, device='cpu')
    forward_event = torch.npu.Event()
    forward_event.record()
    with torch.no_grad():
        with torch_npu.npu.stream(stream):
            stream.wait_event(forward_event)
            tensor_cpu.storage().copy_(ori_tensor.storage(), non_blocking=True)
            WeightGradStore.ori_storage.append(ori_tensor)

    return storage_size, tensor_cpu


def swap_h2d(ori_tensor, tensor_cpu, storage_size, stream):
    with torch.no_grad():
        with torch_npu.npu.stream(stream):
            ori_tensor.storage().resize_(storage_size)
            ori_tensor.storage().copy_(tensor_cpu.storage(), non_blocking=True)


class WeightGradStore:
    cache = []
    weight_grad_queue = queue.Queue()
    store_grad_cache = []
    grad_store = []
    swap_event = []
    prefetch_stream = None
    gather_stream = None
    host_tensors_gradoutput = []
    host_pipe_experts_grad = []
    host_tensors_input = []
    ori_storage = []
    is_decoupleBlock = False
    grad_overlap_count = 0
    interval_per_layers_count = 0
    interval_per_layers = []

    @classmethod
    def put(cls, total_input, grad_output, weight, sequence_parallel, in_row=False, pipe_experts=False):
        if get_args().use_nanopipe_swap:
            if cls.prefetch_stream is None:
                cls.prefetch_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
            if grad_output is not None:
                cls.host_tensors_gradoutput.append(swap_d2h(grad_output, cls.prefetch_stream))
            cls.host_tensors_input.append(swap_d2h(total_input, cls.prefetch_stream))
        cls.interval_per_layers_count += 1
        cls.cache.append((total_input, grad_output, weight, sequence_parallel, in_row, pipe_experts))

    @classmethod
    def flush(cls):
        cls.interval_per_layers.append(cls.interval_per_layers_count)
        cls.interval_per_layers_count = 0

    @classmethod
    def save_grad_output(cls, grad):
        if get_args().use_nanopipe_swap:
            if cls.prefetch_stream is None:
                cls.prefetch_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
            cls.host_pipe_experts_grad.append(swap_d2h(grad, cls.prefetch_stream))
        cls.grad_store.append(grad)

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
            [input, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts] = cls.cache.pop(0)
            if not sequence_parallel:
                return (input, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts), None
            if not in_row:
                total_input, handle = gather(input, cls.gather_stream)
                grad_output = grad_output_slice
            else:
                if pipe_experts and not get_args().use_nanopipe_swap:
                    grad_output_slice = cls.grad_store.pop(0)
                grad_output, handle = gather(grad_output_slice, cls.gather_stream)
                total_input = input
            return [total_input, grad_output, weight, sequence_parallel, in_row, pipe_experts], handle
        else:
            raise Exception("All Gather empty queue.")

    @classmethod
    def swap_tensors(cls):
        if get_args().use_nanopipe_swap:
            if cls.prefetch_stream is None:
                cls.prefetch_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
            cls.prefetch_stream.wait_stream(torch.npu.current_stream())
            for cache_id in range(len(cls.cache)):
                cls.cache[cache_id] = list(cls.cache[cache_id])
                if cls.cache[cache_id][-1] and cls.cache[cache_id][1] is None:
                    cls.cache[cache_id][1] = cls.grad_store.pop(0)
                input, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts = cls.cache[cache_id]
                if pipe_experts:
                    storage_size_g, tensor_cpu_g = cls.host_pipe_experts_grad.pop(0)
                else:
                    storage_size_g, tensor_cpu_g = cls.host_tensors_gradoutput.pop(0)
                storage_size_i, tensor_cpu_i = cls.host_tensors_input.pop(0)
                swap_h2d(grad_output_slice, tensor_cpu_g, storage_size_g, cls.prefetch_stream)
                swap_h2d(input, tensor_cpu_i, storage_size_i, cls.prefetch_stream)
                cls.swap_event.append((cls.prefetch_stream.record_event()))

    @classmethod
    def overlap_matmul(cls, grad_store_cache):
        total_input, grad_output, weight, sequence_parallel, in_row, pipe_experts = grad_store_cache
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
        cls.grad_overlap_count += 1

    @classmethod
    def pop(cls, overlap_arg=None):
        if len(cls.cache) == 0:
            return
        if cls.gather_stream is None:
            cls.gather_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        if get_args().overlap_grad_reduce:
            if overlap_arg is None:
                raise RuntimeError("overlap_arg is invalid")
            pipeline_parallel_size, nano_flag, synchronized_model_chunks, grad_sync_func, model = overlap_arg 
            model_chunk_id = len(nano_flag) - 1
        input, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts = cls.cache.pop(0)
        if not sequence_parallel:
            grad_output = grad_output_slice
            handle = None
        else:
            if pipe_experts and not get_args().use_nanopipe_swap:
                grad_output_slice = cls.grad_store.pop(0)
            grad_output, handle = gather(grad_output_slice, cls.gather_stream)
        layers_count = 0
        cls.store_grad_cache = (input, grad_output, weight, sequence_parallel, in_row, pipe_experts)
        while len(cls.cache) > 0:
            if handle is not None:
                handle.wait()
            next_grad_cache, handle = cls.overlap_all_gather()
            cls.overlap_matmul(cls.store_grad_cache)
            if get_args().overlap_grad_reduce:
                if cls.grad_overlap_count == cls.interval_per_layers[0]:
                    cls.interval_per_layers.pop(0)
                    layers_count += 1
                    if layers_count == pipeline_parallel_size:
                        if model_chunk_id not in synchronized_model_chunks and not nano_flag[model_chunk_id]:
                            grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                            synchronized_model_chunks.add(model_chunk_id)
                        model_chunk_id -= 1
                        layers_count = 0 
                    cls.grad_overlap_count = 0
            cls.store_grad_cache = next_grad_cache
        if handle is not None:
            handle.wait()
        cls.overlap_matmul(cls.store_grad_cache)
        if get_args().overlap_grad_reduce:
            if model_chunk_id not in synchronized_model_chunks and not nano_flag[model_chunk_id]:
                grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
                model_chunk_id -= 1
                cls.grad_overlap_count = 0
        cls.stored_grads = None
        cls.store_grad_cache = None
        cls.swap_event = []
        cls.grad_store = []
        cls.host_pipe_experts_grad = []
        cls.interval_per_layers = []
        cls.interval_per_layers_count = 0

    @classmethod
    def resize_ori_storage(cls, use_nano_swap):
        if use_nano_swap and len(cls.ori_storage) > 0:
            torch.npu.current_stream().wait_stream(cls.prefetch_stream)
            for ori_storage_ in cls.ori_storage:
                ori_storage_.storage().resize_(0)
            cls.ori_storage = []