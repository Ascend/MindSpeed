# npu_grouped_mat_mul_all_reduce External API

```python
def npu_grouped_mat_mul_all_reduce(x: List[torch.Tensor],
                                      weight: List[torch.Tensor],
                                      hcomm: str,
                                      *,
                                      bias: Optional[List[torch.Tensor]] = None,
                                      group_list: Optional[List[int]] = None,
                                      split_item: Optional[int] = 0,
                                      reduce_op: str = "sum",
                                      comm_turn: int = 0) -> List[torch.Tensor]
```

Calculation Logic:
The GroupedMatMulAllReduce operator is the multi-card communication version of the GroupedMatmul operator. It can implement grouped matrix computation, where the dimension sizes of each group's matrix multiplication can differ, offering a flexible combination approach. Both input data x and output data y support split or non-split modes, and whether to split can be determined by the parameter split_item. When x requires splitting, the group_list parameter is used to describe the m-axis split configuration of x. This operator adds the AllReduce collective communication operation, which can split the matrix multiplication task across multiple cards for parallel computation, and then sum the computation results of all cards together through the AllReduce collective communication operation, ultimately completing the entire task. Based on the number of tensors for input x, weight, and output y, this operator can support the following four scenarios:

- The number of tensors for x, weight, and y are all equal to the number of groups group_num, meaning the data corresponding to each group has independent tensors.
- The number of tensors for x is 1, while the number of tensors for weight and y equals the number of groups. In this case, group_list is needed to describe the grouping of x along the m-axis. For example, group_list[0]=10 indicates that the first 10 rows of the x matrix participate in the first group's matrix multiplication computation.
- The number of tensors for x and weight are both equal to the number of groups group_num, and the number of tensors for y is 1. In this case, the results of each group's matrix multiplication are placed consecutively in the same output tensor.
- The tensor count of x and y is both 1, and the tensor count of weight equals the number of groups. This is a combination of the first two cases.

The calculation formula is:
For each grouped matrix multiplication task i: $$y_i = x_i * weight_i + bias_i$$
After splitting across n cards, the computation form can be represented as:

$$
y_i = [x_{i1}, x_{i2}, ..., x_{in}] *
\begin{bmatrix}
{weight_{i1}} \\
{weight_{i2}} \\
{...} \\
{weight_{in}}
\end{bmatrix}+\sum^{n}{bias_i/n}
$$

## Forward API

Inputs:

- x: Required input, List[Tensor], Data Type float16, bfloat16. Maximum Supported Length is 64.
- weight: Required input, List[Tensor], Data Type float16, bfloat16. Maximum Supported Length is 64.
- bias: Optional input, List[Tensor], Data Type float16, float32. Maximum Supported Length is 64. For scenarios without bias, you can directly omit the bias parameter or set it to None.
- group_list: Optional input, Optional[List[int64]], default None. Represents the matmul size distribution in the M dimension for inputs and outputs. Maximum Supported Length is 64.

Outputs:

- y: List[Tensor], Data Type float16, bfloat16. Maximum Supported Length is 64.

Attributes:

- split_item: Optional Attribute, int64. Represents whether the input and output require tensor splitting. 0 means neither input nor output requires splitting; 1 means input requires splitting but output does not; 2 means input does not require splitting but output does; 3 means both input and output require splitting. Defaults to 0.
- hcomm: Required Attribute, Data Type supported: string. Represents the communication domain name, a string identifying the column group on the host side. Obtained through the interface provided by Hccl.
- reduce_op: Optional Attribute, Data Type supported: string. The type of reduce operation. **Currently only supports input "sum".**
- comm_turn: Optional attribute, int64. An integer on the host side, representing the number of splits for communication data, i.e., total data volume / single communication volume. **Currently, only supports input 0.**

## Backward API

None

## Input Constraints

- The maximum supported length of List is 64;
- The attribute `reduce_op` only supports input "sum";
- The attribute `comm_turn` only supports input 0;
- Communication supports 2, 4, or 8 cards.
- When `split_item` is 0 or 2, the number of tensors in `x` is the same as that in `weight`; when `split_item` is 1 or 3, the number of tensors in `x` is 1.
- When `split_item` is 0 or 2, `group_list` is empty; when `split_item` is 1 or 3, the length of `group_list` is the same as the number of tensors in `weight`.
- If bias is not empty, its tensor count must be the same as that of weight.
- The m/k/n dimension relationship for matrix multiplication must be satisfied.
- Supported input element types:
  1. The element type in x is float16, the element type in weight is float16, the element type in bias is float16, and the element type in output y is float16;

  2. The element type in x is bfloat16, the element type in weight is bfloat16, the element type in bias is float32, and the element type in output y is bfloat16;
- The input must include the communication domain hcomm string, which needs to be obtained from the interface in the torch.distributed package.
- Currently, only PyTorch 2.1 is supported.

## Example Call of the npu_grouped_mat_mul_all_reduce Class

```python
import os
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import math
from mindspeed.ops.npu_grouped_mat_mul_all_reduce import npu_grouped_mat_mul_all_reduce


def get_hcomm_info(world_size, rank):
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method='tcp://127.0.0.1:5001')
    print(f'device_{rank} init_process_group success.')
    if dist.is_available():
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    print(f'device_{rank} get_hccl_comm_name success.')
    return dist, hcomm_info


def run(rank, n_device):
    torch_npu.npu.set_device(rank)
    dist, hcomm_info = get_hcomm_info(n_device, rank)

    x1 = torch.randn(32, 256, dtype=torch.float16)
    x1_npu = x1.npu()
    w1 = torch.randn(256, 32, dtype=torch.float16)
    w1_npu = w1.npu()

    x2 = torch.randn(64, 128, dtype=torch.float16)
    x2_npu = x2.npu()
    w2 = torch.randn(128, 64, dtype=torch.float16)
    w2_npu = w2.npu()

    group_list = None
    split_item = 0
    hccl_group = hcomm_info
    reduce_op = "sum"
    comm_turn = 0

    # Forward call example
    x1_npu.requires_grad = False
    x2_npu.requires_grad = False
    w1_npu.requires_grad = False
    w2_npu.requires_grad = False
    y_npu = npu_grouped_mat_mul_all_reduce([x1_npu, x2_npu], [w1_npu, w2_npu], hccl_group, bias=[],
                                           group_list=group_list, split_item=split_item, reduce_op=reduce_op,
                                           comm_turn=comm_turn)
    if rank == 0:
        for i, y in enumerate(y_npu):
            y.cpu().numpy().tofile(f"./y_{i}.bin")


if __name__ == '__main__':
    n_device = 2
    mp.spawn(run, args=(n_device,), nprocs=n_device)

```
