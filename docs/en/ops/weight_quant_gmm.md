# weight_quant_gmm External Interfaces

npu_weight_quant_gmm(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None, group_list=None, act_type=0)

npu_weight_quant_gmm_v2(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None, group_list=None, act_type=0)

Compared with the [npu_weight_quant_gmm] interface, the meaning of group_list in [npu_weight_quant_gmm_v2] is different. In the npu_weight_quant_gmm interface, the values in group_list are the cumsum result (cumulative sum) of the group axis sizes. In the npu_weight_quant_gmm_v2 interface, the values in group_list are the size of each group along the group axis. There is no difference in operator performance between the two interfaces. You can decide which one to use based on the group_list situation in the entire network. If the group_list output by the preceding operator consists of the sizes of each group, it is recommended to use the npu_weight_quant_gmm_v2 interface, because using the npu_weight_quant_gmm interface in this case would require calling torch.cumsum first to convert group_list into the cumulative sum form, which introduces additional overhead.

## Forward API

Inputs:

- x: Required input, parameter is a tensor, data type float16, bfloat16;
- weight: Required input, parameter is a tensor, data type int8;
- antiquant_scale: Required input, parameter type is tensor, data type float16, bfloat16;
- antiquant_offset: Optional parameter, parameter type is tensor, data type float16, bfloat16, default value is None, currently passing None is not supported;
- bias: Optional input, parameter type is tensor, data type float16, float32, default value is None;
- group_list: Optional input, parameter type is tensor, data type int64, default value is None. The numerical definition varies across different interfaces, as detailed in the interface descriptions above;
- act_type: Optional parameter, parameter type is int, used to specify the activation function type, default value is 0, indicating no activation function. Currently, only the default value 0 is supported;

Outputs:

- y: Required output, data type float16, bfloat16.

Constraints and limitations:

- In the npu_weight_quant_gmm interface, group_list must be a non-negative, monotonically non-decreasing sequence, and its length cannot be 1;
- In the npu_weight_quant_gmm_v2 interface, group_list must be a non-negative sequence, its length cannot be 1, and its data type only supports tensor;
- The size of the last dimension of each group of tensors in x and weight must be less than 65536. The last dimension of $x_i$ refers to the K-axis of $x_i$ when the transpose_x attribute is False, or the M-axis of $x_i$ when transpose_x is True. The last dimension of $weight_i$ refers to the N-axis of $weight_i$ when the transpose_weight attribute is False, or the K-axis of $weight_i$ when transpose_weight is True;
- The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, must be less than the maximum int32 value of 2147483647;
- The data types of x, antiquant_scale, antiquant_offset, and y must be consistent.
- When the output y data type is bfloat16, the bias type must be float32.
- When the output y data type is float16, the bias type must be float16.
- FLOPS calculation is not currently supported.

## Example Call of the gmm Class

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import weight_quant_gmm

num_expert, seq_len, hidden_dim, out_channel = 8, 32, 256, 128
group_list = torch.tensor([1, 3, 6, 10, 15, 21, 28, 32], dtype=torch.int64).npu()

x = torch.rand(seq_len, hidden_dim, dtype=torch.float16).npu()
weight = torch.randint(-128, 128, (num_expert, hidden_dim, out_channel), dtype=torch.int8).npu()
antiquant_scale = torch.rand(num_expert, out_channel, dtype=torch.float16).npu()
antiquant_offset = torch.rand(num_expert, out_channel, dtype=torch.float16).npu()

result = weight_quant_gmm.npu_weight_quant_gmm(x, weight, antiquant_scale, antiquant_offset=antiquant_offset,
                                               bias=None, group_list=group_list)

# weight transpose example
weight_trans = torch.randint(-128, 128, (num_expert, out_channel, hidden_dim), dtype=torch.int8).npu()
result = weight_quant_gmm.npu_weight_quant_gmm(x, weight_trans.transpose(-1,-2), antiquant_scale,
                                               antiquant_offset=antiquant_offset, bias=None, group_list=group_list)
```

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import weight_quant_gmm

num_expert, seq_len, hidden_dim, out_channel = 8, 32, 256, 128
group_list = torch.tensor([1, 3, 3, 4, 5, 6, 7, 4], dtype=torch.int64).npu()

x = torch.rand(seq_len, hidden_dim, dtype=torch.float16).npu()
weight = torch.randint(-128, 128, (num_expert, hidden_dim, out_channel), dtype=torch.int8).npu()
antiquant_scale = torch.rand(num_expert, out_channel, dtype=torch.float16).npu()
antiquant_offset = torch.rand(num_expert, out_channel, dtype=torch.float16).npu()

result = weight_quant_gmm.npu_weight_quant_gmm_v2(x, weight, antiquant_scale, antiquant_offset=antiquant_offset,
                                                  bias=None, group_list=group_list)

# weight transpose example
weight_trans = torch.randint(-128, 128, (num_expert, out_channel, hidden_dim), dtype=torch.int8).npu()
result = weight_quant_gmm.npu_weight_quant_gmm_v2(x, weight_trans.transpose(-1,-2), antiquant_scale,
                                                  antiquant_offset=antiquant_offset, bias=None, group_list=group_list)
```
