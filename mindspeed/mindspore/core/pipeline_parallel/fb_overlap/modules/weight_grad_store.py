# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu
from mindspore import ops
from megatron.training import get_args
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


@classmethod
def overlap_matmul(cls, grad_store_cache):
    total_input, grad_output, weight, sequence_parallel, in_row = grad_store_cache
    args = get_args()
    if hasattr(weight, 'gmm_weight'):
        inputs, group_list, group_list_data_type = total_input
        if get_args().gemm_gradient_accumulation_fusion:
            npu_groupmatmul_add_fp32(inputs, grad_output, group_list, weight.main_grad)
        else:
            grad_weight = ops.function.math_func.gmm([inputs.t()], [grad_output], [], group_list, 2, 0)[0]
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
