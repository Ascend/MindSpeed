# npu_groupmatmul_add_fp32

This interface computes each group's weight gradient and adds it to `grad` in place. It calls TorchNPU `npu_grouped_matmul_add_`.

```python
npu_groupmatmul_add_fp32(x, dy, grouplist, grad, *, group_list_type=0)
```

- `x`: shape `[m, k]`, float16 or bfloat16.
- `dy`: shape `[m, n]`, same dtype as `x`.
- `grouplist`: one-dimensional int64 Tensor or Python list, interpreted according to `group_list_type`.
- `grad`: contiguous float32 Tensor, shape `[groups, k, n]` or `[groups * k, n]`.
- `group_list_type=0`: cumulative boundaries. `[16, 64]` describes groups with 16 and 48 rows.
- `group_list_type=1`: row counts. `[16, 48]` describes the same groups.

The interface returns `None` and preserves the original storage of `grad`. Empty groups add zero.

Row counts are converted to cumulative boundaries before calling the in-place operator.

## Example

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
