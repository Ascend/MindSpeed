# npu_moe_token_permute对外接口

npu_moe_token_permute(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int = None,
        padded_mode: bool = False
)

小算子等价计算逻辑：
```python
import torch

def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode. 
       The input indices shape is [num_expert, capacity], it indicates which tokens were selected by each expert separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for each expert.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

    return permuted_tokens, indices


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately. 
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should equal the number of tokens not dropped. By default, set to None, meaning no tokens are dropped.
        padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity] to denote selected tokens per expert. Defaults to False.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)

    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices1
```

## 前向接口：

输入：

- tokens：必选输入，2维Tensor，数据类型bfloat16(当前版本tokens仅支持bfloat16)
- indices: 必选输入，2维Tensor，数据类型int64

输出：

- permuted_tokens：必选输出，2维Tensor，数据类型bfloat16(当前版本permuted_tokens仅支持bfloat16)
- sorted_indices：必选输出，1维Tensor，数据类型int32(当前版本sorted_indices仅支持int32)

属性：

- num_out_tokens：可选属性，数据类型int64_t，表示有效输出token数
- padded_mode: 可选属性，数据类型int64_t，如果为 True，则表示索引被填充到 [num_expert，capacity] 以表示每个专家选择的token


## 反向接口：

输入：

- grad_permuted_tokens：必选输入，2维Tensor，数据类型bfloat16(当前版本grad_permuted_tokens仅支持bfloat16)
- sorted_indices：必选输入，2维Tensor，数据类型int32(当前版本sorted_indices1仅支持int32)

输出：

- grad_tokens：必选输出，2维Tensor，数据类型bfloat16(当前版本grad_tokens仅支持bfloat16)

属性：

- num_topK：必选属性，数据类型int64_t，表示每条token输出的专家个数
- padded_mode：可选属性，数据类型int64_t，表示有效输出token数


**备注**：
1. 目前仅支持padded_mode为False
2. 目前仅支持bfloat16



## 案例

```python
import torch
import torch_npu

from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute

dtype = torch.bfloat16
tokens = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).npu().to(dtype).requires_grad_(True)
indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
num_out_tokens = indices.numel()
probs = torch.ones_like(indices) / 2
probs = probs.npu().to(dtype)
# 正向接口案例
permuted_tokens, sorted_indices = npu_moe_token_permute(tokens, indices, num_out_tokens)

# 反向接口案例
permuted_tokens.backward(torch.ones(permuted_tokens.shape).to(torch.bfloat16).npu())

```