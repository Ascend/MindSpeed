from functools import reduce
import torch
import torch_npu

from .min_comm_cfg import min_comm_config
from .matmul_soc_friendly import get_aligned_mm_inputs
from .rewrite_parallel_linears_all_reduce import RewriteColumnAllReduceFunction, RewriteRowAllReduceFunction
from .coc_utils import set_context, CommunicationType, COCParallel
from .coc_utils import reshape_to_2D, is_grad_needed, get_parallel_num, get_output_shape


class COCColumnAllReduceFunction(RewriteColumnAllReduceFunction):
    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        grad_input = grad_output.matmul(weight)
        handle = torch.distributed.all_reduce(grad_input, group=min_comm_config.tp_group, async_op=True)
        grad_weight, grad_bias = None, None
        if is_grad_weight_needed:
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(input_))
            handle.wait()
            grad_bias = grad_output.sum(dim=0) if ctx.use_bias and is_grad_bias_needed else None
        else:
            handle.wait()

        return grad_input, grad_weight, grad_bias


class COCRowAllReduceFunction(RewriteRowAllReduceFunction):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        trans_weight = weight.t()

        parallel_num = get_parallel_num(m=reduce(lambda x, y: x * y, input_.shape[:-1]),
                                        k=trans_weight.shape[0],
                                        n=trans_weight.shape[1])
        if parallel_num == 1:
            return RewriteRowAllReduceFunction.forward(ctx, input_, weight, bias)

        output_orig_shape = get_output_shape(input_, trans_weight, 1, is_gather=True)
        input_ = reshape_to_2D(input_)

        if min_comm_config.matmul_soc_friendly_enabled:
            input_, trans_weight = get_aligned_mm_inputs(input_, trans_weight, sp_coef=min_comm_config.tp_world_size,
                                                         parallel_num=parallel_num)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, trans_weight, out=output_tensor)
            return output_tensor

        coc_all_gather = COCParallel(input_, CommunicationType.ALL_REDUCE, compute_fcn, compute_first=True,
                                   weight_shape_list=list(trans_weight.shape))
        output_ = coc_all_gather.run()
        output_ = output_.reshape(output_orig_shape)
        if bias is not None:
            output_ = output_ + bias
        return output_

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        grad_input = grad_output.matmul(weight)
        grad_weight, grad_bias = None, None
        if is_grad_weight_needed:
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(total_input))
            grad_bias = grad_output.sum(dim=0) if ctx.use_bias and is_grad_bias_needed else None

        return grad_input, grad_weight, grad_bias
