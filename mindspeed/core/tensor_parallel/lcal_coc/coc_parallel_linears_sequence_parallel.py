from functools import reduce
import torch
import torch_npu

from .min_comm_cfg import min_comm_config
from .matmul_soc_friendly import get_aligned_mm_inputs
from .coc_utils import CommunicationType, COCParallel, get_output_shape
from .coc_utils import shuffle_as_coc_reduce_scatter, shuffle_as_coc_all_gather
from .coc_utils import set_context, reshape_to_2D, async_gather_along_first_dim, is_grad_needed, get_parallel_num
from .rewrite_parallel_linears_sequence_parallel import RewriteColumnSeqParallelFunction, RewriteRowSeqParallelFunction

ALIGN_SIZE = 512


class COCColumnSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        trans_weight = weight.t()

        parallel_num = get_parallel_num(m=reduce(lambda x, y: x * y, input_.shape[:-1]) * min_comm_config.tp_world_size,
                                        k=trans_weight.shape[0],
                                        n=trans_weight.shape[1])
        if parallel_num == 1:
            return RewriteColumnSeqParallelFunction.forward(ctx, input_, weight, bias)

        output_orig_shape = get_output_shape(input_, trans_weight, min_comm_config.tp_world_size, is_gather=True)
        gathered_input_shape = get_output_shape(input_, None, min_comm_config.tp_world_size, is_gather=True)
        input_ = reshape_to_2D(input_)

        if min_comm_config.matmul_soc_friendly_enabled:
            input_, trans_weight = get_aligned_mm_inputs(input_, trans_weight, sp_coef=min_comm_config.tp_world_size,
                                                         parallel_num=parallel_num)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, trans_weight, out=output_tensor)
            return output_tensor

        coc_parallel = COCParallel(input_, CommunicationType.ALL_GATHER, compute_fcn, compute_first=False,
                                 weight_shape_list=list(trans_weight.shape), parallel_num=parallel_num)
        output = coc_parallel.run()
        output = shuffle_as_coc_reduce_scatter(output, min_comm_config.tp_world_size, parallel_num)
        if not min_comm_config.all_gather_recomputation_enabled:
            total_input = shuffle_as_coc_reduce_scatter(coc_parallel.comm_output, min_comm_config.tp_world_size,
                                                       parallel_num)
            ctx.total_input = total_input.reshape(gathered_input_shape)
        output = output.reshape(output_orig_shape)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        grad_input_orig_shape = get_output_shape(grad_output, weight, 1, is_gather=True)
        grad_output = reshape_to_2D(grad_output)

        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        total_input_work, total_input = None, None

        if is_grad_weight_needed:
            if min_comm_config.all_gather_recomputation_enabled:
                total_input_work, total_input = async_gather_along_first_dim(input_, min_comm_config.tp_group,
                                                                             min_comm_config.tp_world_size)
            else:
                total_input = ctx.total_input

        # if grad_output.shape[-1] is not 512B aligned, transpose its memory alignment but keep its shape
        if grad_output.is_contiguous() and (grad_output.shape[-1] * grad_output.element_size()) % ALIGN_SIZE > 0:
            grad_output = grad_output.t().contiguous().t()
        grad_input = grad_output.matmul(weight)
        grad_input = grad_input.reshape(grad_input_orig_shape)
        sub_grad_input = torch.empty(list(input_.size()), dtype=input_.dtype, device=torch.cuda.current_device())
        sub_grad_input_work = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                                     group=min_comm_config.tp_group, async_op=True)
        grad_weight, grad_bias = None, None
        if is_grad_weight_needed:
            if min_comm_config.all_gather_recomputation_enabled:
                total_input_work.wait()
            total_input = reshape_to_2D(total_input)
            grad_weight = grad_output.t().matmul(total_input)
            sub_grad_input_work.wait()
            if is_grad_bias_needed and ctx.use_bias:
                grad_bias = grad_output.sum(dim=0) if grad_output.is_contiguous() else grad_output.t().sum(dim=1)
        else:
            sub_grad_input_work.wait()
        return sub_grad_input, grad_weight, grad_bias


class COCRowSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        ctx.world_size = min_comm_config.tp_world_size
        trans_weight = weight.t()

        parallel_num = get_parallel_num(m=reduce(lambda x, y: x * y, input_.shape[:-1]),
                                        k=trans_weight.shape[0],
                                        n=trans_weight.shape[1])
        if parallel_num == 1:
            return RewriteRowSeqParallelFunction.forward(ctx, input_, weight, bias)

        output_orig_shape = get_output_shape(input_, trans_weight, min_comm_config.tp_world_size, is_gather=False)
        input_ = reshape_to_2D(input_)

        if min_comm_config.matmul_soc_friendly_enabled:
            input_, trans_weight = get_aligned_mm_inputs(input_, trans_weight, parallel_num=parallel_num)

        def compute_fcn(input_tensor):
            sub_output = torch.matmul(input_tensor, trans_weight)
            return sub_output

        input_ = shuffle_as_coc_all_gather(input_, ctx.world_size, parallel_num)
        coc_reduce_scatter = COCParallel(input_, CommunicationType.REDUCE_SCATTER, compute_fcn, compute_first=True,
                                       weight_shape_list=list(trans_weight.shape), parallel_num=parallel_num)
        output_ = coc_reduce_scatter.run()
        output_ = output_.reshape(output_orig_shape)
        if bias is not None:
            output_ = output_ + bias
        return output_

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors

        parallel_num = get_parallel_num(
            m=reduce(lambda x, y: x * y, grad_output.shape[:-1]) * min_comm_config.tp_world_size,
            k=weight.shape[0],
            n=weight.shape[1]
        )
        if parallel_num == 1:
            return RewriteRowSeqParallelFunction.backward(ctx, grad_output)

        grad_input_orig_shape = get_output_shape(grad_output, weight, min_comm_config.tp_world_size, is_gather=True)
        grad_output = reshape_to_2D(grad_output)

        if min_comm_config.matmul_soc_friendly_enabled:
            grad_output, weight = get_aligned_mm_inputs(grad_output, weight, sp_coef=min_comm_config.tp_world_size,
                                                        parallel_num=parallel_num)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, weight, out=output_tensor)
            return output_tensor

        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)

        coc_all_gather = COCParallel(grad_output, CommunicationType.ALL_GATHER, compute_fcn, compute_first=False,
                                   weight_shape_list=list(weight.shape), parallel_num=parallel_num)
        grad_input = coc_all_gather.run()
        grad_input = shuffle_as_coc_reduce_scatter(grad_input, ctx.world_size, parallel_num)

        grad_input = grad_input.reshape(grad_input_orig_shape)

        grad_weight, grad_bias = None, None

        if is_grad_weight_needed:
            grad_output = coc_all_gather.comm_output
            grad_output = shuffle_as_coc_reduce_scatter(grad_output, ctx.world_size, parallel_num)
            total_input = reshape_to_2D(total_input)
            grad_weight = grad_output.t().matmul(total_input)
            if is_grad_bias_needed and ctx.use_bias:
                grad_bias = grad_output.sum(dim=0) if grad_output.is_contiguous() else grad_output.t().sum(dim=1)

        return grad_input, grad_weight, grad_bias
