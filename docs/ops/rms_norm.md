# rms_norm对外接口
## 前向接口：
输入：
- x：必选输入，数据类型float16, bfloat16, float32	
- gamma：必选输入，数据类型float16, bfloat16, float32
输出
- rstd：中间变量，数据类型float32	
- out：必选输出，数据类型float16, bfloat16, float32

属性：
- epsilon：可选属性，数据类型float，缺省1e-6。

## 反向接口：
输入：
- dy：必选输入，数据类型float16, bfloat16, float32
- x：必选输入，数据类型float16, bfloat16, float32
- rstd：必选输入，数据类型float32	
- gamma：必选输入，数据类型float16, bfloat16, float32	
输出
- dx：必选输出，数据类型float16, bfloat16, float32
- dgammma：必选输出，数据类型float32

## 案例

```python
    import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.op_builder import RmsNormOpBuilder

x = torch.rand([2, 4]).to(torch.float).npu()
gamma = torch.rand([4]).to(torch.float).npu()
// 正向接口案例
mindspeed_ops = RmsNormOpBuilder.load()
out = mindspeed_ops.rms_norm(x, gamma)
// 反向接口案例
out.backward(torch.ones(out.shape).npu())

```
