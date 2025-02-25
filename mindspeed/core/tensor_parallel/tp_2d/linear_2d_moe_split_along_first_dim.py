# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
 
import torch
import torch_npu
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
 
from mindspeed.core.tensor_parallel.comm_autograd_function import CollectiveCommIntf
from mindspeed.core.tensor_parallel.comm_group_api import OverlapCollectiveIntf, TPYCollectiveComm
from mindspeed.core.tensor_parallel.comm_utils import async_gather_tensors, sync_gather_along_last_dim, \
    _split_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import async_reduce_scatter_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_reduce_scatter_along_first_dim
 
G_FORWARD_PADDING_SIZE = 0
G_BACKWARD_PADDING_SIZE = 0
 
 
class MoELinear2DFC1(torch.autograd.Function):
    """2D Linear out axe communication implementation."""
 
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        activation_input,
        weight,
        bias,
        ag_comm_intf: CollectiveCommIntf,
        ag_overlap_comm_intf: OverlapCollectiveIntf,
        rs_comm_intf: CollectiveCommIntf,
        rs_overlap_comm_intf: OverlapCollectiveIntf,
        enable_overlap_ag_with_matmul=False,
        enable_overlap_matmul_with_rs=False,
        gradient_accumulation_fusion=False,
        enable_backward_overlap_ag_with_matmul=False,
        partition_dim=0,
    ):
        """
        :param ctx: context to save some tensors or vars for backward use.
        :param activation_input: with shape: [s/(x*cp), b, h/y]
        :param weight: with shape: [h/y, E/x], E means the output size.
        :param bias: bias parameter tensor.
        :param ag_comm_intf: AllGather communication process group interface.
        :param ag_overlap_comm_intf: AllGather communication overlap send and recv comm group
        :param rs_comm_intf: ReduceScatter communication process group interface.
        :param rs_overlap_comm_intf: ReduceScatter communication overlap send and recv comm group
        :param enable_overlap_ag_with_matmul:  enable overlap all-gather with matmul in forward
        :param enable_overlap_matmul_with_rs: enable overlap matmul with reduce-scatter in forward
        :param gradient_accumulation_fusion: enable gradient accumulation fusion
        :param enable_backward_overlap_ag_with_matmul: enable overlap all-gather with matmul
        :return: forward result tensor.
        """
        ctx.weight = weight
        ctx.use_bias = bias is not None
        ctx.rs_comm_intf = rs_comm_intf
        ctx.ag_comm_intf = ag_comm_intf
        ctx.ag_overlap_comm_intf = ag_overlap_comm_intf
        ctx.rs_overlap_comm_intf = rs_overlap_comm_intf
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.enable_backward_overlap_ag_with_matmul = enable_backward_overlap_ag_with_matmul
 
        activation_input = activation_input.contiguous()
        # [n, h] -> [n, h/y]
        activation_input = _split_along_last_dim(activation_input, TPYCollectiveComm)
        ctx.save_for_backward(activation_input)
        # [N, h/y] @ [h/y, E/x] -> [N, E/x]
        matmul_res = torch.matmul(activation_input, weight.npu().t())
        matmul_res = matmul_res.contiguous()
        n_tokens, h = matmul_res.shape
        rs_size = rs_comm_intf.get_comm_group_world_size()
        global G_FORWARD_PADDING_SIZE
        remaining = n_tokens - n_tokens // rs_size * rs_size
        G_FORWARD_PADDING_SIZE = rs_size - remaining if remaining else 0
        if G_FORWARD_PADDING_SIZE != 0:
            padding_tensor = torch.zeros(G_FORWARD_PADDING_SIZE, h, dtype=matmul_res.dtype,
                                         device=matmul_res.device)
            matmul_res = torch.cat((matmul_res, padding_tensor), dim=0)
        matmul_res = matmul_res.contiguous()
        # [N1, E/x] -> [N1/y, E/x]
        matmul_res = sync_reduce_scatter_along_first_dim(matmul_res, rs_comm_intf)
        return matmul_res
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # activation_input shape: [n, h]
        # weight shape: [h/y, E/x]
        activation_input, = ctx.saved_tensors
        weight = ctx.weight
        use_bias = ctx.use_bias
        # [N1/y, E/x]---AG(y)---> [N1, E/x]
        grad_output = grad_output.contiguous()
        global G_BACKWARD_PADDING_SIZE
        total_grad_output = sync_gather_along_first_dim(grad_output, ctx.rs_comm_intf)
        if G_BACKWARD_PADDING_SIZE != 0:
            real_input_num = total_grad_output.shape[0] - G_BACKWARD_PADDING_SIZE
            # [N1, E/x] --> [N, E/x]
            total_grad_output = total_grad_output[:real_input_num, :]
 
        # prepare total activation_input for computing grad weight.
        # [N, h/y]
        total_activation_input = activation_input.contiguous()
 
        # [N, E/x] @ [E/x, H/y]--> [N, H/y] (partial x)
        partial_grad_input = total_grad_output.matmul(weight).contiguous()
        grad_input = partial_grad_input
        if ctx.gradient_accumulation_fusion:
            import fused_weight_gradient_mlp_cuda
            total_grad_output = total_grad_output.contiguous()
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_activation_input, total_grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_activation_input, total_grad_output, weight.main_grad
                )
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
                        dtype=activation_input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=activation_input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            # [E/x, N] @ [N, h/y] ---> [E/x, h/y]
            grad_weight = total_grad_output.t().matmul(total_activation_input)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None
        grad_input = sync_gather_along_last_dim(grad_input, ctx.rs_comm_intf)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None
 
 
class MoELinear2DFC2(torch.autograd.Function):
    """2D Linear out axe communication implementation."""
 
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        activation_input,
        weight,
        bias,
        ag_comm_intf: CollectiveCommIntf,
        ag_overlap_comm_intf: OverlapCollectiveIntf,
        rs_comm_intf: CollectiveCommIntf,
        rs_overlap_comm_intf: OverlapCollectiveIntf,
        enable_overlap_ag_with_matmul=False,
        enable_overlap_matmul_with_rs=False,
        gradient_accumulation_fusion=False,
        enable_backward_overlap_ag_with_matmul=False,
        partition_dim=0,
    ):
        """
        :param ctx: context to save some tensors or vars for backward use.
        :param activation_input: with shape: [s/(x*cp), b, h/y]
        :param weight: with shape: [h/y, E/x], E means the output size.
        :param bias: bias parameter tensor.
        :param ag_comm_intf: AllGather communication process group interface.
        :param ag_overlap_comm_intf: AllGather communication overlap send and recv comm group
        :param rs_comm_intf: ReduceScatter communication process group interface.
        :param rs_overlap_comm_intf: ReduceScatter communication overlap send and recv comm group
        :param enable_overlap_ag_with_matmul:  enable overlap all-gather with matmul in forward
        :param enable_overlap_matmul_with_rs: enable overlap matmul with reduce-scatter in forward
        :param gradient_accumulation_fusion: enable gradient accumulation fusion
        :param enable_backward_overlap_ag_with_matmul: enable overlap all-gather with matmul
        :return: forward result tensor.
        """
        ctx.save_for_backward(activation_input)
        ctx.weight = weight
        ctx.use_bias = bias is not None
        ctx.rs_comm_intf = rs_comm_intf
        ctx.ag_comm_intf = ag_comm_intf
        ctx.ag_overlap_comm_intf = ag_overlap_comm_intf
        ctx.rs_overlap_comm_intf = rs_overlap_comm_intf
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.enable_backward_overlap_ag_with_matmul = enable_backward_overlap_ag_with_matmul
        activation_input = activation_input.contiguous()
        # [N1/y, E/x] -> ag(y) -> [N1, E/x]
        total_input = sync_gather_along_first_dim(activation_input, ag_comm_intf)
        if G_FORWARD_PADDING_SIZE != 0:
            real_input_num = total_input.shape[0] - G_FORWARD_PADDING_SIZE
            # [N1, E/x] -> [N, E/x]
            total_input = total_input[:real_input_num, :]
        # [N, E/x] @ [E/x, h/y] -> [N, h/y] (partial x)
        matmul_res = torch.matmul(total_input, weight.npu().t())
        matmul_res = sync_gather_along_last_dim(matmul_res, TPYCollectiveComm)
        return matmul_res
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # activation_input shape: [N1/y, E/x]
        # weight shape: [h/y, E/x]
        activation_input, = ctx.saved_tensors
        weight = ctx.weight
        use_bias = ctx.use_bias
        # [N, h] -> [N, h/y]
        grad_output = grad_output.contiguous()
        grad_output = _split_along_last_dim(grad_output, ctx.ag_comm_intf)

        global G_BACKWARD_PADDING_SIZE
        # [N1/y, E/x]---AG(y)--->[N1, E/x]
        activation_input = activation_input.contiguous()
        gather_input_handle, gathered_tensors = async_gather_tensors(
            local_rank_input=activation_input, ag_comm_intf=ctx.ag_comm_intf
        )
        # [N, h/y] @ [E/x, H/y]--> [N, E/x] (partial y)
        partial_grad_input = grad_output.matmul(weight).contiguous()
        sb, h = partial_grad_input.shape
        rs_size = ctx.ag_comm_intf.get_comm_group_world_size()
 
        remaining = sb - sb // rs_size * rs_size
        G_BACKWARD_PADDING_SIZE = rs_size - remaining if remaining else 0
 
        if G_BACKWARD_PADDING_SIZE != 0:
            padding_tensor = torch.zeros(G_BACKWARD_PADDING_SIZE, h, dtype=partial_grad_input.dtype,
                                         device=partial_grad_input.device)
            # [N, E/x] --> [N1, E/x]
            partial_grad_input = torch.cat((partial_grad_input, padding_tensor), dim=0)
        partial_grad_input = partial_grad_input.contiguous()
        # [N1, E/x] --> [N1/y, E/x]
        rs_grad_input_handle, grad_input = async_reduce_scatter_along_first_dim(
            partial_grad_input, comm_intf=ctx.ag_comm_intf
        )
 
        if gather_input_handle:
            gather_input_handle.wait()
        # [N1, E/x]
        total_activation_input = gathered_tensors.contiguous()
        if G_BACKWARD_PADDING_SIZE != 0:
            real_input_num = total_activation_input.shape[0] - G_BACKWARD_PADDING_SIZE
            # [N1, E/x] -> [N, E/x]
            total_activation_input = total_activation_input[:real_input_num, :]
        total_activation_input = total_activation_input.contiguous()
        if ctx.gradient_accumulation_fusion:
            import fused_weight_gradient_mlp_cuda
            grad_output = grad_output.contiguous()
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_activation_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_activation_input, grad_output, weight.main_grad
                )
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
                        dtype=activation_input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=activation_input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            # [N, E/x] @ [E/x, h/y] --> [N, h/y] (partial x)
            grad_weight = grad_output.t().matmul(total_activation_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
 
        if rs_grad_input_handle:
            rs_grad_input_handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None
