# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


class _GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor,
                m_split=None,
                group_list_type=None,
                *weights) -> torch.Tensor:

        # Due to ascend gmm kernal k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split
        weights_t = [w.T for w in weights]
        ctx.group_list_type = group_list_type
        fwd_output = torch_npu.npu_grouped_matmul([input_tensor], weights_t, bias=None, group_list=ctx.group_list,
                                                  split_item=2, group_type=0, group_list_type=ctx.group_list_type)[0]
        ctx.save_for_backward(input_tensor, *weights)
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        group_list = ctx.group_list
        inp, *weights = ctx.saved_tensors
        group_list_type = ctx.group_list_type
        grad = torch_npu.npu_grouped_matmul([grad_output], weights, bias=None, group_list=group_list,
                                            split_item=2, group_type=0, group_list_type=group_list_type)[0]
        # K spilt gmm.
        grad_weight = torch_npu.npu_grouped_matmul([inp.T], [grad_output], bias=None, group_list=group_list,
                                                   split_item=3, group_type=2, group_list_type=group_list_type)[0]
        grad_weight = [w.T for w in grad_weight]
        return grad, None, None, *grad_weight


def fused_grouped_matmul(inputs, m_split, *weights):
    return _GroupedMatmul.apply(inputs, m_split, 1, *weights)


def eager_grouped_matmul(inputs, m_split, *weights):
    output_shape = inputs.shape[:-1] + (weights[0].shape[0],)
    final_hidden_states = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)

    group_list = [0] + torch.cumsum(m_split, dim=0).tolist()
    for i in range(len(group_list) - 1):
        final_hidden_states[group_list[i]:group_list[i + 1], ...] = torch.matmul(
            inputs[group_list[i]:group_list[i + 1], ...], weights[i].T)

    return final_hidden_states
