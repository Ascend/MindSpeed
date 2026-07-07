# quant_gmm External API

npu_quant_gmm(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
output_dtype=None, act_type=0)

npu_quant_gmm_v2(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
output_dtype=None, act_type=0)

Compared with the [npu_quant_gmm] interface, [npu_quant_gmm_v2] differs in the meaning of group_list. In the npu_quant_gmm interface, the values in group_list represent the cumsum result (cumulative sum) of the grouping axis sizes. In the npu_quant_gmm_v2 interface, the values in group_list represent the size of each group on the grouping axis. There is no performance difference between the two interfaces. You can decide which one to use based on the group_list situation in the entire network. If the group_list output by the preceding operator represents the size of each group, it is recommended to use the npu_quant_gmm_v2 interface, because using the npu_quant_gmm interface in this case would require calling torch.cumsum first to convert group_list into the cumulative sum form, which introduces additional overhead.

## Forward API

Inputs:

- x: Required Input, Parameter Type tensor, Data Type int8;
- weight: Required Input, Parameter Type tensor, Data Type int8;
- scale: Required Input, Parameter Type tensor, Data Type int64, bfloat16, float32;
- offset: Reserved parameter, currently not enabled;
- per_token_scale: Optional parameter, Parameter Type is tensor, Data Type is float32, Default Value is None;
- bias: Optional Input, Parameter Type is tensor, Data Type is int32, Default Value is None;
- group_list: Optional Input, Parameter Type is tensor, Data Type is int64, Default Value is None. The numerical definition varies across different interfaces; see the interface descriptions above for details;
- output_dtype: Optional Input, Parameter Type is torch.dtype, optional values are: torch.int8, torch.bfloat16, torch.float16, used to specify the output data type. Default Value is None, in which case the output Type is torch.float16;
- act_type: Optional parameter, Parameter Type is int, used to specify the activation function type. Default Value is 0. The supported activation function types are as follows:
  - 0: No activation function;
  - 1: relu;
  - 2: gelu_tanh;
  - 3: gelu_err_func (not yet supported);
  - 4: fast_gelu;
  - 5: silu.

Outputs:

- y: Required output, data type int8, float16, bfloat16.

Constraints and Limitations:

- In the npu_quant_gmm interface, group_list must be a non-negative monotonically non-decreasing sequence, and its length cannot be 1;
- In the npu_quant_gmm_v2 interface, group_list must be a non-negative sequence, its length cannot be 1, and its data type only supports tensor;
- The size of the last dimension of each group tensor in x and weight must be less than 65536. The last dimension of $x_i$ refers to the K axis of $x_i$ when the attribute transpose_x is False, or the M axis of $x_i$ when transpose_x is True. The last dimension of $weight_i$ refers to the N axis of $weight_i$ when the attribute transpose_weight is False, or the K axis of $weight_i$ when transpose_weight is True;
- The size of each dimension of each group tensor in x and weight, after 32-byte alignment, must be less than the maximum int32 value 2147483647;
- When the output y data type needs to be int8, specify output_dtype as torch.int8, the scale type as int64, and per_token_scale as empty. In this case, only act_type=0 is supported, meaning no activation function; this scenario currently only supports single-operator mode, and graph mode is not supported;
- When the output y data type needs to be bfloat16, output_dtype is torch.bfloat16, and the scale type is bfloat16;
- When the output y data type is float16, output_dtype is torch.float16 or the default parameter None, and the scale type is float32.

## Example Call of the gmm Class

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import quant_gmm

num_expert, seq_len, hidden_dim, out_channel = 8, 32, 256, 128
group_list = torch.tensor([1, 3, 6, 10, 15, 21, 28, 32], dtype=torch.int64).npu()

x = torch.randint(-128, 128, (seq_len, hidden_dim), dtype=torch.int8).npu()
weight = torch.randint(-128, 128, (num_expert, hidden_dim, out_channel), dtype=torch.int8).npu()
scale = torch.rand(num_expert, out_channel, dtype=torch.float32).npu()
per_token_scale = torch.rand(seq_len, dtype=torch.float32).npu()

result = quant_gmm.npu_quant_gmm(x, weight, scale, per_token_scale=per_token_scale,
                                 bias=None, group_list=group_list, output_dtype=torch.float16)

# # weight transpose example
weight_trans = torch.randint(-128, 128, (num_expert, out_channel, hidden_dim), dtype=torch.int8).npu()
result = quant_gmm.npu_quant_gmm(x, weight_trans.transpose(-1,-2), scale, per_token_scale=per_token_scale,
                                 bias=None, group_list=group_list, output_dtype=torch.float16)
```

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import quant_gmm

num_expert, seq_len, hidden_dim, out_channel = 8, 32, 256, 128
group_list = torch.tensor([1, 3, 3, 4, 5, 6, 7, 4], dtype=torch.int64).npu()

x = torch.randint(-128, 128, (seq_len, hidden_dim), dtype=torch.int8).npu()
weight = torch.randint(-128, 128, (num_expert, hidden_dim, out_channel), dtype=torch.int8).npu()
scale = torch.rand(num_expert, out_channel, dtype=torch.float32).npu()
per_token_scale = torch.rand(seq_len, dtype=torch.float32).npu()

result = quant_gmm.npu_quant_gmm_v2(x, weight, scale, per_token_scale=per_token_scale,
                                    bias=None, group_list=group_list, output_dtype=torch.float16)

# # weight transpose example
weight_trans = torch.randint(-128, 128, (num_expert, out_channel, hidden_dim), dtype=torch.int8).npu()
result = quant_gmm.npu_quant_gmm_v2(x, weight_trans.transpose(-1,-2), scale, per_token_scale=per_token_scale,
                                    bias=None, group_list=group_list, output_dtype=torch.float16)
```
