# rms_norm External Interface

## Forward API

Inputs:

- x: Required Input, data type float16, bfloat16, float32
- gamma: Required Input, data type float16, bfloat16, float32

Outputs:

- rstd: Intermediate variable, data type float32
- out: Required Output, data type float16, bfloat16, float32

Attributes:

- epsilon: Optional attribute, data type float, defaults to 1e-6.

## Backward API

Inputs:

- dy: Required Input, data type float16, bfloat16, float32
- x: Required Input, data type float16, bfloat16, float32
- rstd: Required Input, data type float32
- gamma: Required input, data type float16, bfloat16, float32

Outputs:

- dx: Required output, data type float16, bfloat16, float32
- dgamma: Required output, data type float32

## Example

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.op_builder import RmsNormOpBuilder

x = torch.rand([2, 4]).to(torch.float).npu()
gamma = torch.rand([4]).to(torch.float).npu()
# Forward interface example
mindspeed_ops = RmsNormOpBuilder().load()
out = mindspeed_ops.rms_norm(x, gamma)
# Backward interface example
out.backward(torch.ones(out.shape).npu())

```
