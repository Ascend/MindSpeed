from functools import reduce
import torch
import torch_npu

from .min_comm_cfg import min_comm_config
from .coc_utils import get_parallel_num, set_context, is_grad_needed, check_equal
from .coc_utils import async_gather_along_first_dim, reshape_to_2D, allocate_for_output
from .coc_parallel_linears_sequence_parallel import COCColumnSeqParallelFunction, COCRowSeqParallelFunction
from .rewrite_parallel_linears_sequence_parallel import RewriteColumnSeqParallelFunction, RewriteRowSeqParallelFunction

ALIGN_SIZE = 512


class FusedCOCColumnSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        from mindspeed.ops.lcal_functional import coc_ops

        set_context(ctx, input_, weight, bias)

        parallel_num = get_parallel_num(reduce(lambda x, y: x * y, input_.shape[:-1]) * min_comm_config.tp_world_size,
                                        weight.shape[1], weight.shape[0], default_parallel_num=-1)
        if parallel_num == 1:
            return RewriteColumnSeqParallelFunction.forward(ctx, input_, weight, bias)
        elif parallel_num in [2, 4, 8]:
            return COCColumnSeqParallelFunction.forward(ctx, input_, weight, bias)

        output_shape = list(input_.shape)[:-1] + list([weight.shape[0]])
        output_shape[0] = output_shape[0] * min_comm_config.tp_world_size
        input_ = reshape_to_2D(input_)

        output = allocate_for_output(input1=input_, input2=weight.t(),
                                     tp_world_size=min_comm_config.tp_world_size, is_gather=True)

        coc_ops.all_gather_matmul(input_, weight, output, bias)
        output = output.reshape(output_shape)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        from mindspeed.ops.lcal_functional import coc_ops

        input_, weight = ctx.saved_tensors
        check_equal(grad_output.shape[0] % min_comm_config.tp_world_size, 0,
                    error_info="m size must be multiple of world size")
        sub_grad_input_shape = [grad_output.shape[0] // min_comm_config.tp_world_size] + \
                               list(grad_output.shape[1:-1]) + [weight.shape[-1]]
        # manually make sure grad_output is 2D and its memory inner axis is 512B aligned
        grad_output = reshape_to_2D(grad_output)
        if grad_output.is_contiguous() and (grad_output.shape[-1] * grad_output.element_size()) % ALIGN_SIZE > 0:
            grad_output = grad_output.t().contiguous().t()
        sub_grad_input = allocate_for_output(input1=reshape_to_2D(input_))
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        grad_weight, grad_bias = None, None

        if is_grad_weight_needed:
            if min_comm_config.all_gather_recomputation_enabled:
                total_input_work, total_input = async_gather_along_first_dim(input_, min_comm_config.tp_group,
                                                                             min_comm_config.tp_world_size)
            else:
                total_input = ctx.total_input
            total_input = reshape_to_2D(total_input)

            if min_comm_config.enable_coc_in_column_backward:
                coc_ops.matmul_reduce_scatter(grad_output, weight, sub_grad_input, bias=None)
            else:
                grad_input = grad_output.matmul(weight)
                sub_grad_input_work = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                                             group=min_comm_config.tp_group,
                                                                             async_op=True)

            if min_comm_config.all_gather_recomputation_enabled:
                total_input_work.wait()

            grad_weight = grad_output.t().matmul(total_input)
            if is_grad_bias_needed and ctx.use_bias:
                grad_bias = grad_output.sum(dim=0) if grad_output.is_contiguous() else grad_output.t().sum(dim=1)

            if not min_comm_config.enable_coc_in_column_backward:
                sub_grad_input_work.wait()

        else:
            grad_input = grad_output.matmul(weight)
            torch.distributed._reduce_scatter_base(sub_grad_input, grad_input, group=min_comm_config.tp_group)

        sub_grad_input = sub_grad_input.reshape(sub_grad_input_shape)
        return sub_grad_input, grad_weight, grad_bias


class FusedCOCRowSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        from mindspeed.ops.lcal_functional import coc_ops

        set_context(ctx, input_, weight, bias)
        ctx.world_size = min_comm_config.tp_world_size

        parallel_num = get_parallel_num(reduce(lambda x, y: x * y, input_.shape[:-1]), weight.shape[1],
                                        weight.shape[0], default_parallel_num=-1)
        if parallel_num == 1:
            return RewriteRowSeqParallelFunction.forward(ctx, input_, weight, bias)
        elif parallel_num in [2, 4, 8]:
            return COCRowSeqParallelFunction.forward(ctx, input_, weight, bias)

        output_shape = list(input_.shape)[:-1] + list([weight.shape[0]])
        output_shape[0] = output_shape[0] // min_comm_config.tp_world_size
        input_ = reshape_to_2D(input_)

        output = allocate_for_output(input_, weight.t(), min_comm_config.tp_world_size, is_gather=False)
        coc_ops.matmul_reduce_scatter(input_, weight, output, bias)
        output = output.reshape(output_shape)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        from mindspeed.ops.lcal_functional import coc_ops

        total_input, weight = ctx.saved_tensors

        parallel_num = get_parallel_num(
            reduce(lambda x, y: x * y, grad_output.shape[:-1]) * min_comm_config.tp_world_size,
            weight.shape[0], weight.shape[1], default_parallel_num=-1
        )
        if parallel_num == 1:
            return RewriteRowSeqParallelFunction.backward(ctx, grad_output)
        elif parallel_num in [2, 4, 8]:
            return COCRowSeqParallelFunction.backward(ctx, grad_output)

        grad_input_shape = list(grad_output.shape)[:-1] + list([weight.shape[-1]])
        grad_input_shape[0] = grad_input_shape[0] * min_comm_config.tp_world_size
        grad_output = reshape_to_2D(grad_output)
        total_input = reshape_to_2D(total_input)
        grad_input = allocate_for_output(grad_output, weight, min_comm_config.tp_world_size, is_gather=True)

        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        grad_weight, grad_bias = None, None

        if is_grad_weight_needed:
            gathered_grad_output = allocate_for_output(grad_output, tp_world_size=min_comm_config.tp_world_size,
                                                       is_gather=True)
            coc_ops.all_gather_matmul_v2(grad_output, weight, grad_input, gathered_grad_output, bias=None)

            grad_weight = gathered_grad_output.t().matmul(total_input)
            if is_grad_bias_needed and ctx.use_bias:
                grad_bias = gathered_grad_output.sum(dim=0) if gathered_grad_output.is_contiguous() \
                    else gathered_grad_output.t().sum(dim=1)
        else:
            coc_ops.all_gather_matmul(grad_output, weight, grad_input, bias=None)

        grad_input = grad_input.reshape(grad_input_shape)
        return grad_input, grad_weight, grad_bias
