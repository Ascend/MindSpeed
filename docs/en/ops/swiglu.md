# swiglu External API

## Forward API

Input:

- x: Required input, data type float16, bfloat16, float32

Output:

- y: Required output, data type float16, bfloat16, float32

Attributes:

- dim: Optional attribute, data type int32_t, default -1.

## Backward API

Inputs:

- dy: Required input, data type float16, bfloat16, float32
- x: Required input, data type float16, bfloat16, float32

Outputs:

- dx: Required output, data type float16, bfloat16, float32

Attributes:

- dim: Optional attribute, data type int32_t, default -1.

## Example

```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import math
    from mindspeed.op_builder import SwigluOpBuilder

    x = np.random.uniform(-2, 2, (8192,1,3904))
    x = torch.from_numpy(x).float().npu()
    y_grad = np.random.uniform(-2, 2, (8192,1,1952))
    y_grad = torch.from_numpy(y_grad).float().npu()

    x.requires_grad = True
    # Forward interface example
    mindspeed_ops = SwigluOpBuilder().load()
    result = mindspeed_ops.swiglu(x, dim=-1)
    # Backward interface example
    result.backward(y_grad)
