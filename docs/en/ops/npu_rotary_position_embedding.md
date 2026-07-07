# npu_rotary_position_embedding External Interface

npu_rotary_position_embedding(x, cos, sin, mode=0)

Equivalent calculation logic of the small operator:

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

## Forward API

Inputs:

- x: Required Input, 4-dimensional Tensor, data type float16, bfloat16, float32
- cos: Required Input, 4-dimensional Tensor, data type float16, bfloat16, float32
- sin: Required Input, 4-dimensional Tensor, data type float16, bfloat16, float32

Outputs:

- y: Required Output, data type float16, bfloat16, float32

Attributes:

- mode: Optional attribute, data type int64_t, used to select the computation mode. 0: rotate_half (GPT-NeoX style); 1: rotate_interleaved (GPT-J style). Defaults to 0.

## Backward API

Inputs:

- dy: Required input, 4-dimensional Tensor, data type float16, bfloat16, float32
- cos: Required Input, 4D Tensor, Data Type float16, bfloat16, float32
- sin: Required Input, 4D Tensor, Data Type float16, bfloat16, float32
- x: Optional Input, 4D Tensor, Data Type float16, bfloat16, float32

Outputs:

- dx: Required Output, 4D Tensor, Data Type float16, bfloat16, float32
- dcos: Optional output, 4-dimensional Tensor, data type float16, bfloat16, float32
- dsin: Optional output, 4-dimensional Tensor, data type float16, bfloat16, float32

Attributes:

- mode: Optional attribute, data type int64_t, used to select the computation mode, 0: rotate_half (GPT-NeoX style); 1: rotate_interleaved (GPT-J style). Defaults to 0.

## Input Constraints

| Input | RotateHalf(mode: 0) | RotateInterleaved(mode: 1) |
| :-: | :- | :- |
| x | Layout support: BNSD, BSND, SBND; <br> D < 896, and D must be a multiple of 2; <br> B, N < 1000; <br> When the backward gradient of cos/sin needs to be computed, B*N <= 1024 | Layout support: BNSD, BSND, SBND; <br> B * N < 1000; <br> D < 896, and D must be a multiple of 2; |
| cos | Data range: [-1, 1]; <br>Support for corresponding x layout: <br> x is BNSD: 11SD, B1SD, BNSD; <br> x is BSND: 1S1D, BS1D, BSND; <br> x is SBND: S11D, SB1D, SBND. | Data range: [-1, 1]; <br>Support for corresponding x layout: <br> x is BNSD: 11SD; <br> x is BSND: 1S1D; <br> x is SBND: S11D.|
| sin | Same as cos | Same as cos |

**Note**:

1. Inputs do not support being None;
2. If the backward gradient of cos and sin needs to be computed, both must have `requires_grad = True` set; if only one is set, neither will be computed;
3. In RotateHalf (mode=0) mode, when the input layout is BNSD and D is not 32-byte aligned, it is recommended not to use this fused operator (do not enable the `--use-fused-rotary-pos-emb` option in the model launch script), as performance degradation may occur.

## Example

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
# Forward call example
result = npu_rotary_position_embedding(x_npu, cos_npu, sin_npu, 0)

# Backward call example
result.backward(torch.ones_like(result).npu())
x_npu.grad
cos_npu.grad
sin_npu.grad
```
