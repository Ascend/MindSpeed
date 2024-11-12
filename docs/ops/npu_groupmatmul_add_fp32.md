# npu_groupmatmul_add_fp32对外接口(只支持前向)
输入：
- x：必选输入，维度[m,k], 数据类型float16, bf16
- weight：必选输入，维度[k,n], 数据类型float16, bf16
- groupNum：必选输入， 维度[NumGroup], 数据类型int64
- output：必选输入，维度[w1,w2]，数据类型float32

输出：
- output：必选输出，数据类型float32

## 案例
```python
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch
import torch_npu
from mindspeed.ops import gmm
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32

x = torch.rand((1024, 4096), dtype=torch.float16).npu()
weight = torch.rand((1024, 2048), dtype=torch.float16).npu()
group_list = torch.tensor([500, 1024], dtype=torch.int64).npu()
out = torch.rand((8192, 2048), dtype=torch.float32).npu()
# 分开算子计算结果
dy = gmm.npu_gmm(x, weight, bias=None, group_list=group_list,group_type=0)
out.add_(dy)
#融合算子计算结果
npu_groupmatmul_add_fp32(x, weight, group_list, out)
```
