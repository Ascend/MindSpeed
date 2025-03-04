# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16


def unaligned_divide(numerator, world_size, rank):
    res = numerator // world_size
    if rank < numerator % world_size:
        res += 1
    return res


def unaligned_split_along_first_dim(input_, group):
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    rank = torch.distributed.get_rank(group=group)

    # Split along first dimension.
    dim_size = input_.size()[0]

    local_dim_size = unaligned_divide(dim_size, world_size, rank)

    less_dim_size = dim_size // world_size
    dim_offset = rank * less_dim_size
    if rank >= dim_size % world_size:
        dim_offset += dim_size % world_size
    else:
        dim_offset += rank

    output = input_[dim_offset: dim_offset + local_dim_size].contiguous()

    return output


def unaligned_gather_along_first_dim(input_, dim_size, group, async_op=False):
    """Gather tensors and concatinate along the first dimension."""

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output = []
    for rank in range(world_size):
        rank_dim_size = dim_size // world_size
        if rank < dim_size % world_size:
            rank_dim_size += 1
        output.append(torch.empty((int(rank_dim_size), *(input_.size()[1:])), dtype=input_.dtype,
                                  device=torch.cuda.current_device()))

    handle = torch.distributed.all_gather(output, input_.contiguous(), group=group, async_op=async_op)

    def post_process():
        if handle is not None:
            handle.wait()
        return torch.cat(output)

    if async_op:
        return post_process
    return post_process()


class UnalignedScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.dim_size = list(input_.size())[0]
        ctx.parallel_group = group
        return unaligned_split_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return unaligned_gather_along_first_dim(grad_output, ctx.dim_size, ctx.parallel_group), None


def unaligned_scatter_to_sequence_parallel_region(input_, group):
    return UnalignedScatterToSequenceParallelRegion.apply(input_, group)


def unaligned_reduce_scatter_along_first_dim(input_, group, async_op=False):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    rank = torch.distributed.get_rank(group=group)

    # Split along first dimension.
    dim_size = input_.size()[0]

    local_dim_size = unaligned_divide(dim_size, world_size, rank)

    less_dim_size = dim_size // world_size
    dim_offset = rank * less_dim_size
    if rank >= dim_size % world_size:
        dim_offset += dim_size % world_size
    else:
        dim_offset += rank

    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=group, async_op=async_op)

    def post_process():
        if handle is not None:
            handle.wait()
        return input_[dim_offset: dim_offset + local_dim_size].contiguous()

    if async_op:
        return post_process
    return post_process()


class UnalignedReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.dim_size = list(input_.size())[0]
        ctx.parallel_group = group
        return unaligned_reduce_scatter_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return unaligned_gather_along_first_dim(grad_output, ctx.dim_size, ctx.parallel_group), None


def unaligned_reduce_scatter_to_sequence_parallel_region(input_, group):
    return UnalignedReduceScatterToSequenceParallelRegion.apply(input_, group)


class UnalignedGatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from the sequence parallel region."""

    @staticmethod
    def forward(ctx, input_, dim_size, group, tensor_parallel_output_grad):
        ctx.dim_size = dim_size
        ctx.parallel_group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return unaligned_gather_along_first_dim(input_, dim_size, group)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_parallel_output_grad:
            return (
                unaligned_reduce_scatter_to_sequence_parallel_region(grad_output),
                None, 
                None, 
                None
            )
            
        else:
            return (
                unaligned_split_along_first_dim(grad_output),
                None,
                None,
                None
            )
            
            
class UnalignedLinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    def forward(
            ctx,
            input,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,

            # unaligned parallel arguments
            parallel_group,
            seq_length=None
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.grad_output_buffer = grad_output_buffer
        ctx.parallel_group = parallel_group

        if sequence_parallel:
            if seq_length is None:
                seq_len = torch.Tensor([list(input.size())[0]]).cuda()
                torch.distributed.all_reduce(seq_len, group=parallel_group)
                seq_length = seq_len.item()
            total_input = unaligned_gather_along_first_dim(input, seq_length, parallel_group)
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias

        ctx.seq_length = seq_length
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        parallel_group = ctx.parallel_group

        wgrad_compute = True
        post_process = None
        total_input = None
        if grad_output_buffer is not None:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                post_process = unaligned_gather_along_first_dim(input, ctx.seq_length, parallel_group, async_op=True)
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            total_input = post_process()

        if wgrad_compute and grad_output.dim() == 3:
            grad_output = grad_output.contiguous()
            grad_output = grad_output.view(-1, grad_output.shape[2])
            total_input = total_input.view(-1, total_input.shape[2])

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=parallel_group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            post_process = unaligned_reduce_scatter_along_first_dim(grad_input, parallel_group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    npu_matmul_add_fp32(total_input, grad_output, weight.main_grad)
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    npu_matmul_add_fp16(total_input, grad_output, weight.main_grad)
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            sub_grad_input = post_process()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
    
    
def unaligned_gather_from_sequence_parallel_region(input_, dim_size, group, tensor_parallel_output_grad):
    return UnalignedGatherFromSequenceParallelRegion.apply(input_, dim_size, group, tensor_parallel_output_grad)


def unaligned_linear_with_grad_accumulation_and_async_allreduce(
        input, weight, bias, gradient_accumulation_fusion, allreduce_dgrad, sequence_parallel, grad_output_buffer,
        parallel_group, seq_length=None
):
    args = [input, weight, bias, gradient_accumulation_fusion, allreduce_dgrad, sequence_parallel, grad_output_buffer,
        parallel_group, seq_length]
    return UnalignedLinearWithGradAccumulationAndAsyncCommunication.apply(*args)
