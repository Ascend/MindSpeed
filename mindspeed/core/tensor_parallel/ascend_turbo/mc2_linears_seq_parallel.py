import torch
import torch_npu
from megatron.training import get_args
from .ascend_turbo_cfg import ascend_turbo_cfg


class ColumnSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_)
        ctx.use_bias = bias is not None
        ctx.weight = weight
        ctx.gradient_accumulation_fusion = get_args().gradient_accumulation_fusion

        rank = torch.distributed.get_rank(group)
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )

        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        world_size = ascend_turbo_cfg.get_world_size()
        # npu_all_gather_base_mm currently do not support bias
        output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            x,
            weight.t(),
            hcomm_info,
            world_size,
            bias=None,
            gather_index=0,
            gather_output=(not ascend_turbo_cfg.all_gather_recomputation),
        )

        if bias is not None:
            output = output + bias

        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )

        ctx.all_gather_output = all_gather_grad_output
        ctx.world_size = world_size
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors[0]
        weight = ctx.weight

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        if ascend_turbo_cfg.all_gather_recomputation:
            dim_size = list(input_.size())
            dim_size[0] = dim_size[0] * ctx.world_size
            all_gather_output = torch.empty(
                dim_size,
                dtype=input_.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            all_gather_work = torch.distributed._all_gather_base(
                all_gather_output, input_.contiguous(), group=ctx.group, async_op=True
            )
        else:
            all_gather_output = ctx.all_gather_output

        grad_input = grad_output_.matmul(weight)
        grad_input = grad_input.reshape(
            grad_output.shape[0], grad_output.shape[1], weight.shape[1]
        )

        sub_grad_input = torch.empty(
            list(input_.size()), dtype=input_.dtype, device=torch.cuda.current_device()
        )
        reduce_scatter_work = torch.distributed._reduce_scatter_base(
            sub_grad_input, grad_input, group=ctx.group, async_op=True
        )

        if ascend_turbo_cfg.all_gather_recomputation:
            all_gather_work.wait()
        all_gather_output = all_gather_output.reshape(
            all_gather_output.shape[0] * all_gather_output.shape[1],
            all_gather_output.shape[2],
        )

        if ctx.gradient_accumulation_fusion and weight.main_grad.dtype == torch.float32:
            from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
            npu_matmul_add_fp32(all_gather_output, grad_output_, weight.main_grad)

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output_.t().matmul(all_gather_output)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = (
                grad_output_.sum(dim=0)
                if grad_output_.is_contiguous()
                else grad_output_.t().sum(dim=1)
            )
        else:
            grad_bias = None

        reduce_scatter_work.wait()
        return sub_grad_input, grad_weight, grad_bias, None


class RowSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_)
        ctx.use_bias = bias is not None
        ctx.weight = weight
        ctx.gradient_accumulation_fusion = get_args().gradient_accumulation_fusion

        rank = torch.distributed.get_rank(group)
        world_size = ascend_turbo_cfg.get_world_size()
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        # npu_mm_reduce_scatter_base currently do not support bias
        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=None
        )

        if bias is not None:
            output = output + bias

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors[0]
        weight = ctx.weight
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        grad_input, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )

        grad_input = grad_input.view_as(input_)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        if ctx.gradient_accumulation_fusion and weight.main_grad.dtype == torch.float32:
            from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
            npu_matmul_add_fp32(x, all_gather_grad_output, weight.main_grad)

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = all_gather_grad_output.t().matmul(x)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = (
                grad_output.sum(dim=0)
                if grad_output.is_contiguous()
                else grad_output.t().sum(dim=1)
            )
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None


class ColumnSeqParallelLinearWithFrozenWeight(ColumnSeqParallelLinear):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.input_shape = input_.shape
        ctx.use_bias = bias is not None
        ctx.weight = weight

        rank = torch.distributed.get_rank(group)
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )

        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        world_size = ascend_turbo_cfg.get_world_size()
        # npu_all_gather_base_mm currently do not support bias
        output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            x,
            weight.t(),
            hcomm_info,
            world_size,
            bias=None,
            gather_index=0,
            gather_output=(not ascend_turbo_cfg.all_gather_recomputation),
        )

        if bias is not None:
            output = output + bias

        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )
        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        weight = ctx.weight

        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size
        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        sub_grad_input = torch_npu.npu_mm_reduce_scatter_base(
            grad_output_, weight, hcomm_info, world_size, bias=None
        )

        sub_grad_input = sub_grad_input.view(input_shape)

        return sub_grad_input, None, None, None


class RowSeqParallelLinearWithFrozenWeight(RowSeqParallelLinear):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.input_shape = input_.shape
        ctx.use_bias = bias is not None
        ctx.weight = weight

        rank = torch.distributed.get_rank(group)
        world_size = ascend_turbo_cfg.get_world_size()
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        # npu_mm_reduce_scatter_base currently do not support bias
        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=None
        )

        if bias is not None:
            output = output + bias

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        weight = ctx.weight
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size
        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        grad_input, _ = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )

        grad_input = grad_input.view(input_shape)

        return grad_input, None, None, None
