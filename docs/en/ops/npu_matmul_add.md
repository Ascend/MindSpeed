# npu_matmul_add_fp32 External API (Forward Only)

Inputs:

- x: Required Input, Data Type float16, bf16
- weight: Required Input, Data Type float16, bf16
- C: Required Input, Data Type float32

Output:

- output: Required output, data type float32

## Example

```python
    import torch
    import torch_npu
    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32

    x = torch.rand((4096, 8192),dtype=torch.float16).npu()
    weight = torch.rand((4096, 8192),dtype=torch.float16).npu()
    C = torch.rand((8192, 8192),dtype=torch.float32).npu()
    # Separate operator computation process
    product = torch.mm(x.T, weight)
    result = product + C
    # Fused operator computation process
    npu_matmul_add_fp32(weight, x, C)
```
