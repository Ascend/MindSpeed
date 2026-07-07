# gmm External API

npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None)

npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None)

Compared with the [npu_gmm] API, [npu_gmm_v2] has a different meaning for group_list. In the npu_gmm interface, the values in group_list are the cumsum results (cumulative sum) of the axis group sizes, while in the npu_gmm_v2 interface, the values in group_list are the size of each group on the grouping axis. There is no performance difference between the two interfaces. You can decide which one to use based on the group_list situation in the entire network. If the group_list output by the preceding operator represents the size of each group, it is recommended to use the npu_gmm_v2 interface, because using the npu_gmm interface in this case would require calling torch.cumsum first to convert group_list into the cumulative sum form, which introduces additional overhead.

## Forward API

Inputs:

- x: Required Input, a tensor, Data Type float16, bfloat16, float32
- weight: Required Input, a tensor, Data Type float16, bfloat16, float32
- bias: Optional Input, a tensor, Data Type float16, float32, Default Value None. In training scenarios, only bias=None is supported
- group_list: Optional Input, Data Type list[int64], tensor, Default Value None. The value definitions differ across interfaces, as detailed above.
- group_type: Optional input, data type int64, representing the axis to be grouped. For example, if the matrix multiplication is C[m,n]=A[m,k]xB[k,n], then group_type values are -1: no grouping, 0: group along m axis, 1: group along n axis, 2: group along k axis. Default Value: 0.
- gemm_fusion: Optional input, bool, data type True, False. Used to enable the GMM+ADD fusion operator when accumulating gradients in the backward pass. Default Value: False.
- original_weight: Optional input, tensor, data type float16, bfloat16, float32. Used to obtain the main_grad of the weight before view for gradient accumulation in GMM+ADD. Default Value: None.

Output:

- y: Required Output, data type float16, bfloat16, float32

Constraints and Limitations:

- In the npu_gmm interface, group_list must be a non-negative, monotonically non-decreasing sequence, and its length cannot be 1.
- In the npu_gmm_v2 interface, group_list must be a non-negative sequence, its length cannot be 1, and the data type only supports tensor.
- Supported scenarios for different group_type:

    |  group_type   |   Scenario Limitations  |
    | :---: | :---: |
    |  0  |  1. The tensor in weight must be 3-dimensional, and the tensors in x and y must be 2-dimensional.<br>2. group_list must be passed. If the npu_gmm interface is called, the last value must equal the first dimension of the tensor in x. If the npu_gmm_v2 interface is called, the sum of the values must equal the first dimension of the tensor in x.  |
    |  2  |  1. The tensors in x and weight must be 2-dimensional, and the tensor in y must be 2-dimensional.<br>2. group_list must be passed. If the npu_gmm interface is called, the last value must equal the first dimension of the tensor in x. If the npu_gmm_v2 interface is called, the sum of the values must equal the first dimension of the tensor in x.  |

- group_type does not support the scenario where group_type=1. For Ascend 310 series processors, the supported transpose scenario is group_type=0, with x being a single tensor, weight being a single tensor, and y being a single tensor.
- The size of the last dimension of each group of tensors in x and weight should be less than 65536. The last dimension of $x_i$ refers to the K axis of $x_i$ when the attribute transpose_x is False, or the M axis of $x_i$ when transpose_x is True. The last dimension of $weight_i$ refers to the N axis of $weight_i$ when the attribute transpose_weight is False, or the K axis of $weight_i$ when transpose_weight is True.
- The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, should be less than the maximum int32 value of 2147483647.

## Backward API

Input:

- grad: Required Input, a tensor, Data Type float16, bfloat16, float32
- x: Required Input, a tensor, Data Type float16, bfloat16, float32
- weight: Required Input, a tensor, Data Type float16, bfloat16, float32
- group_list: Optional Input, Data Type list[int64], tensor, Default Value None. Data comes from the forward input

Output:

- grad_x: Required Output, Data Type float16, bfloat16, float32
- grad_weight: Required Output, Data Type float16, bfloat16, float32
- grad_bias: Currently not supported, Default Value None

## Example Call of the gmm Class

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import gmm

num_expert, seq_len, hidden_dim = 8, 32, 256
group_list = [1, 3, 6, 10, 15, 21, 28, 32]
group_type = 0

x_shape = (seq_len, hidden_dim)
weight_shape = (num_expert, hidden_dim, seq_len)
dtype = torch.float16
x = (torch.rand(x_shape).to(dtype) - 0.5)
weight = (torch.rand(weight_shape).to(dtype) - 0.5)

# Forward call example
x.requires_grad = True
weight.requires_grad = True
result = gmm.npu_gmm(x.npu(), weight.npu(), bias=None, group_list=group_list, group_type=group_type)

# Backward call example
result.backward(torch.ones(result.shape).npu())

# Weight transpose example
weight_shape_trans = (num_expert, seq_len, hidden_dim)
weight_trans = (torch.rand(weight_shape_trans).to(dtype) - 0.5)
weight_trans.requires_grad = True
result = gmm.npu_gmm(x.npu(), weight_trans.transpose(-1,-2).npu(), bias=None, group_list=group_list, group_type=group_type)
```

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import gmm

num_expert, seq_len, hidden_dim = 8, 32, 256
group_list = torch.tensor([1, 3, 3, 4, 5, 6, 7, 4])
group_type = 0

x_shape = (seq_len, hidden_dim)
weight_shape = (num_expert, hidden_dim, seq_len)
dtype = torch.float16
x = (torch.rand(x_shape).to(dtype) - 0.5)
weight = (torch.rand(weight_shape).to(dtype) - 0.5)

# Forward call example
x.requires_grad = True
weight.requires_grad = True
result = gmm.npu_gmm_v2(x.npu(), weight.npu(), bias=None, group_list=group_list.npu(), group_type=group_type)

# Backward call example
result.backward(torch.ones(result.shape).npu())

# Weight transpose example
weight_shape_trans = (num_expert, seq_len, hidden_dim)
weight_trans = (torch.rand(weight_shape_trans).to(dtype) - 0.5)
weight_trans.requires_grad = True
result = gmm.npu_gmm_v2(x.npu(), weight_trans.transpose(-1,-2).npu(), bias=None, group_list=group_list.npu(), group_type=group_type)
```
