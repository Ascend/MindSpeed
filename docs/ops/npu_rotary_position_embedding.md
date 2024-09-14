# npu_rotary_position_embedding对外接口

npu_rotary_position_embedding(x, cos, sin, mode=0)

小算子等价计算逻辑：
```python
import torch
from einops import rearrange

# mode = 0
def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# mode = 1
def rotate_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ...(d two)", two=2)

def fused_rotary_position_embedding(x, cos, sin, interleaved=False):
    if not interleaved:
        return x * cos + rotate_half(x) * sin
    else:
        return x * cos + rotate_interleaved(x) * sin
```

## 前向接口：

输入：

- x：必选输入，4维Tensor，数据类型float16, bfloat16, float32
- cos: 必选输入，4维Tensor，数据类型float16, bfloat16, float32
- sin: 必选输入，4维Tensor，数据类型float16, bfloat16, float32

输出：

- y：必选输出，数据类型float16, bfloat16, float32

属性：

- mode：可选属性，数据类型int64_t，用于选择计算模式，0: rotate_half（GPT-NeoX style）; 1: rotate_interleaved（GPT-J style）。缺省为0。


## 反向接口：

输入：

- dy：必选输入，4维Tensor，数据类型float16, bfloat16, float32
- cos：必选输入，4维Tensor，数据类型float16, bfloat16, float32
- sin：必选输入，4维Tensor，数据类型float16, bfloat16, float32
- x: 可选输入，4维Tensor，数据类型float16, bfloat16, float32

输出：

- dx：必选输出，4维Tensor，数据类型float16, bfloat16, float32
- dcos：可选输出，4维Tensor，数据类型float16, bfloat16, float32
- dsin：可选输出，4维Tensor，数据类型float16, bfloat16, float32

属性：

- mode：可选属性，数据类型int64_t，用于选择计算模式，0: rotate_half（GPT-NeoX style）; 1: rotate_interleaved（GPT-J style）。缺省为0。

## 输入约束：

| 输入 | RotateHalf(mode: 0) | RotateInterleaved(mode: 1) |
| :-: | :- | :- |
| x | layout支持: BNSD、BSND、SBND; <br> D < 896，且为2的倍数; <br> B, N < 1000;  <br> 当需要计算cos/sin的反向梯度时，B*N <= 1024 | layout支持: BNSD、BSND、SBND; <br> B * N < 1000; <br> D < 896, 且D为2的倍数; |
| cos | 数据范围：[-1, 1]; <br>对应x layout的支持情况：<br> x为BNSD: 11SD、B1SD、BNSD; <br> x为BSND: 1S1D、BS1D、BSND; <br> x为SBND: S11D、SB1D、SBND. | 数据范围：[-1, 1]; <br>对应x layout的支持情况：<br> x为BNSD: 11SD; <br> x为BSND: 1S1D; <br> x为SBND: S11D.|
| sin | 同cos | 同cos |

**备注**：
1. 输入不支持为None;
2. cos, sin如需计算反向梯度需要同时设置`requires_grad = True`, 若仅设置一个则均不计算;
3. RotaryHalf（mode=0）模式下，当输入layout是BNSD，且D为非32Bytes对齐时，建议不使用该融合算子（模型启动脚本中不开启`--use-fused-rotary-pos-emb`选项），否则可能出现性能下降。


## 案例

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

x = np.random.uniform(-2, 2, (4, 8192, 4, 128))
cos = np.random.uniform(-1, 1, (1, 8192, 1, 128))
sin = np.random.uniform(-1, 1, (1, 8192, 1, 128))

x_npu = torch.from_numpy(x).float().npu()
cos_npu = torch.from_numpy(cos).float().npu()
sin_npu = torch.from_numpy(sin).float().npu()

x_npu.requires_grad = True
cos_npu.requires_grad = True
sin_npu.requires_grad = True
# 正向接口案例
result = npu_rotary_position_embedding(x_npu, cos_npu, sin_npu, 0)

# 反向接口案例
result.backward(torch.ones_like(result).npu())
x_npu.grad
cos_npu.grad
sin_npu.grad
```