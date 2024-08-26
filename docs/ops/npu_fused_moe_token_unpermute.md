# npu_moe_token_unpermute对外接口

npu_moe_token_unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
)

小算子等价计算逻辑：
```python
import torch

def unpermute_with_padded_tokens(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their corresponding probabilities.
    
    This function takes a tensor of permuted tokens and reorders them according to the provided indices. It also combines the tokens with their associated probabilities.
    
    Parameters:
        permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for each expert.
        probs (torch.Tensor): A tensor with the same shape as indices, containing probabilities corresponding to each token.
        restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.
    
    Returns:
        torch.Tensor: A tensor of unpermuted tokens, merged with their probabilities.

    """
    # Ensure permuted_tokens is 2D
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

    # Reshape and expand probabilities and indices to match permuted_tokens
    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    assert (
        permuted_tokens.shape == indices.shape
    ), "Shape mismatch between permuted_tokens and indices."

    # Combine tokens with their probabilities
    combined_output = probs * permuted_tokens

    # Prepare a tensor of zeros with the desired output shape
    empty_tokens = torch.zeros(
        restore_shape,
        dtype=combined_output.dtype,
        device=combined_output.device,
        requires_grad=True,
    )

    # Scatter the combined tokens back to their original positions
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens

def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
        padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity] to denote selected tokens per expert. Defaults to False.
        restore_shape (torch.Size, optional): The input shape before permutation, only used in padding mode. Defaults to None.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens
```

## 前向接口：

输入：

- permuted_tokens：必选输入，2维Tensor，数据类型bfloat16(当前版本permuted_tokens仅支持bfloat16)
- sorted_indices: 必选输入，1维Tensor，数据类型int32(当前版本sorted_indices仅支持int32)
- probs：可选输入，2维Tensor，数据类型bfloat16(当前版本probs仅支持bfloat16)

输出：

- unpermuted_tokens：必选输出，2维Tensor，数据类型bfloat16(当前版本unpermuted_tokens仅支持bfloat16)

属性：

- padded_mode: 可选属性，数据类型int64_t，如果为 True，则表示索引被填充到 [num_expert，capacity] 以表示每个专家选择的token


## 反向接口：

输入：

- permuted_tokens：必选输入，2维Tensor，数据类型bfloat16(当前版本permuted_tokens仅支持bfloat16)
- grad_unpermuted_tokens：必选输入，2维Tensor，数据类型bfloat16(当前版本grad_permuted_tokens仅支持bfloat16)
- sorted_indices: 必选输入，1维Tensor，数据类型int32(当前版本sorted_indices仅支持int32)
- probs：可选输入，2维Tensor，数据类型bfloat16(当前版本probs仅支持bfloat16)

输出：

- grad_permuted_tokens：必选输出，2维Tensor，数据类型bfloat16(当前版本grad_permuted_tokens仅支持bfloat16)
- grad_probs：必选输出，2维Tensor，数据类型bfloat16(当前版本grad_probs仅支持bfloat16)

属性：

- padded_mode：可选属性，数据类型int64_t，表示有效输出token数


**备注**：
1. 目前仅支持padded_mode为False
2. 目前仅支持bfloat16


## 案例

```python
import torch
import torch_npu

from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute

dtype = torch.bfloat16
permuted_tokens = torch.tensor([[1., 1., 1.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [3., 3., 3.],
                                    [2., 2., 2.],
                                    [1., 1., 1.],
                                    [2., 2., 2.],
                                    [3., 3., 3.]]).npu().to(dtype).requires_grad_(True)
sorted_indices = torch.tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=torch.int32).npu()
indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
probs = torch.ones_like(indices) / 2
probs = probs.npu().to(dtype).requires_grad_(True)

# 正向接口案例
unpermuted_tokens = npu_moe_token_unpermute(
    permuted_tokens, sorted_indices, probs=probs)

# 反向接口案例
unpermuted_tokens.backward(torch.ones(unpermuted_tokens.shape).to(torch.bfloat16).npu())
```