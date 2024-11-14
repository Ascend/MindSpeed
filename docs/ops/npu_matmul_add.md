# npu_matmul_add_fp32对外接口(只支持前向)
输入：
- x：必选输入，数据类型float16, bf16
- weight：必选输入，数据类型float16, bf16
- C：必选输入，数据类型float32

输出：
- output：必选输出，数据类型float32

## 案例
```python
    import torch
    import torch_npu
    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32

    x = torch.rand((4096, 8192),dtype=torch.float16).npu()
    weight = torch.rand((4096, 8192),dtype=torch.float16).npu()
    C = torch.rand((8192, 8192),dtype=torch.float).npu()
    # 分开算子计算过程
    product = torch.mm(x.T, weight)
    result = product + C
    # 融合算子计算过程
    npu_matmul_add_fp32(weight, x, C)
```
