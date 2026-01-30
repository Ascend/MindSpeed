# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


class _GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weights, weights_bias, m_split, group_list_type) -> torch.Tensor:
        # Due to ascend gmm kernal k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split

        ctx.group_list_type = group_list_type
        weights_t = [w[0].T for w in weights.chunk(weights.shape[0], dim=0)]
        fwd_output = torch_npu.npu_grouped_matmul([input_tensor], weights_t, bias=weights_bias,
                                                  group_list=ctx.group_list, split_item=2, group_type=0,
                                                  group_list_type=ctx.group_list_type)[0]
        ctx.save_for_backward(input_tensor, weights)
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
        return grad, torch.stack(grad_weight), None, None, None,


def fused_grouped_matmul(inputs, m_split, weights):
    return _GroupedMatmul.apply(inputs, weights, None, m_split, 1)


def eager_grouped_matmul(inputs, m_split, weights):
    """
    Grouped matrix multiplication that handles two weight tensor formats.

    Args:
        inputs: Tensor of shape [batch_size, input_dim]
        m_split: Tensor of group sizes that sum to batch_size
        weights: Weight tensor of either:
                 Format 1: [num_groups, input_dim, output_dim] - ready for matmul
                 Format 2: [num_groups, output_dim, input_dim] - needs transpose

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    batch_size, input_dim = inputs.shape

    # Automatically detect and adjust weight format
    # Check if second dimension matches input dimension (Format 1)
    if weights.shape[1] == input_dim:
        # Format 1: [num_groups, input_dim, output_dim]
        output_dim = weights.shape[2]
        # No transformation needed - weights are already in correct format
    else:
        # Format 2: [num_groups, output_dim, input_dim]
        # Transpose to convert to Format 1: [num_groups, input_dim, output_dim]
        output_dim = weights.shape[1]
        weights = weights.transpose(1, 2)

    # Initialize output tensor
    output_shape = (batch_size, output_dim)
    final_hidden_states = torch.zeros(output_shape, dtype=inputs.dtype, device=inputs.device)

    # Calculate group boundaries from cumulative sum
    group_list = [0] + torch.cumsum(m_split, dim=0).tolist()

    # Process each group separately
    for i in range(len(group_list) - 1):
        start_idx = group_list[i]
        end_idx = group_list[i + 1]

        # Matrix multiplication for current group
        # inputs[start_idx:end_idx, :] has shape [group_size, input_dim]
        # weights[i] has shape [input_dim, output_dim] (after format normalization)
        final_hidden_states[start_idx:end_idx, :] = torch.matmul(
            inputs[start_idx:end_idx, :],
            weights[i]
        )

    return final_hidden_states
