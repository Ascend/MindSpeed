# npu_groupmatmul_add_fp32

此接口计算每组输入的权重梯度，并原地累加到 `grad`。接口直接调用 TorchNPU 的 `npu_grouped_matmul_add_`。

```python
npu_groupmatmul_add_fp32(x, dy, grouplist, grad, *, group_list_type=0)
```

- `x`：形状 `[m, k]`，类型为 float16 或 bfloat16。
- `dy`：形状 `[m, n]`，类型与 `x` 相同。
- `grouplist`：一维 int64 Tensor 或 Python list；数值含义由 `group_list_type` 指定。
- `grad`：连续 float32 Tensor，形状 `[groups, k, n]` 或 `[groups * k, n]`。
- `group_list_type=0`：`grouplist` 保存累计边界。例如 `[16, 64]` 表示两组各有 16、48 行。
- `group_list_type=1`：`grouplist` 保存每组行数。例如 `[16, 48]`。

接口返回 `None`。接口保留 `grad` 的原始存储。空组的累加值为零。

接口将每组行数转换为累计边界后，调用原地累加算子。

## 示例

```python
import torch
import torch_npu
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32

x = torch.randn(64, 32, dtype=torch.float16).npu()
dy = torch.randn(64, 16, dtype=torch.float16).npu()
groups = torch.tensor([16, 48], dtype=torch.int64).npu()
grad = torch.zeros(2, 32, 16, dtype=torch.float32).npu()
npu_groupmatmul_add_fp32(x, dy, groups, grad, group_list_type=1)
```
