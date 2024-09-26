# quant_gmm对外接口

npu_quant_gmm(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
output_dtype=None, act_type=0)

npu_quant_gmm_v2(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
output_dtype=None, act_type=0)

[npu_quant_gmm_v2]相较于[npu_quant_gmm]接口, group_list的含义不同, 在npu_quant_gmm接口中group_list中数值为分组轴大小的cumsum结果（累积和），npu_quant_gmm_v2接口中group_list中数值为分组轴上每组大小。

## 前向接口：

输入：

- x：必选输入，参数为tensor，数据类型int8；
- weight：必选输入，参数为tensor，数据类型int8；
- scale：必选输入，参数类型为tensor，数据类型int64，bfloat16，float32；
- offset：保留参数，当前未使能；
- per_token_scale：可选参数，参数类型为tensor，数据类型float32，默认值为none；
- bias：可选输入，参数类型为tensor，数据类型int32, 默认值为none；
- group_list：可选输入，参数类型为tensor，数据类型int64，默认值为none。不同接口中的数值定义不同，具体见上述接口说明中描述；
- output_dtype：可选输入，参数类型为torch.dtype，可选值为：torch.int8，torch.bfloat16，torch.float16，用于指定输出数据类型，默认值为None，此时输出类型为torch.float16;
- act_type：可选参数，参数类型为int，用于指定激活函数类型，默认值为0，支持的激活函数类型如下：
  - 0：无激活函数；
  - 1：relu;
  - 2：gelu_tanh;
  - 3：gelu_err_func（暂不支持）；
  - 4：fast_gelu;
  - 5：silu。

输出：

- y：必选输出，数据类型int8, float16, bfloat16。

约束与限制：

- npu_quant_gmm接口中，group_list必须为非负单调非递减数列，且长度不能为1；
- npu_quant_gmm_v2接口中，group_list必须为非负数列，长度不能为1，且数据类型仅支持tensor；
- x和weight中每一组tensor的最后一维大小都应小于65536.$x_i$的最后一维指当属性transpose_x为false时$x_i$的K轴或当transpose_x为true时$x_i$的M轴。$weight_i$的最后一维指当属性transpose_weight为false时$weight_i$的N轴或当transpose_weight为true时$weight_i$的K轴；
- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647；
- 当需要输出y数据类型为int8时，指定output_dtype为torch.int8，scale类型为int64，per_token_scale为空，此时只支持act_type=0，即无激活函数；该场景当前仅支持单算子模式，图模式不支持；
- 当需要输出y数据类型为bfloat16时，output_dtype为torch.bfloat16，scale类型为bfloat16；
- 当需要输出y数据类型为float16时，output_dtype为torch.float16或者默认参数None，scale类型为float32。

## gmm 类的调用方式

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

# weight转置案例
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

# weight转置案例
weight_trans = torch.randint(-128, 128, (num_expert, out_channel, hidden_dim), dtype=torch.int8).npu()
result = quant_gmm.npu_quant_gmm_v2(x, weight_trans.transpose(-1,-2), scale, per_token_scale=per_token_scale,
                                    bias=None, group_list=group_list, output_dtype=torch.float16)
```
