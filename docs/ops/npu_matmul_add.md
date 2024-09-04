# npu_matmul_add对外接口(只支持前向)
输入：
- x：必选输入，数据类型float16, bf16
- weight：必选输入，数据类型float16, bf16
- C：必选输入，数据类型float

输出：
- output：必选输出，数据类型float

## 案例
```python
    import torch
    from mindspeed.ops.npu_matmul_add import npu_matmul_add

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand((4096, 8192),dtype=torch.float16).to(device)
    weight = torch.rand((4096, 8192),dtype=torch.float16).to(device)
    C = torch.rand((8192, 8192),dtype=torch.float).to(device)
    # 分开算子计算过程
    product = torch.mm(x.T, weight)
    result = product + C
    # 融合算子计算过程
    npu_matmul_add(x, weight, C)
```