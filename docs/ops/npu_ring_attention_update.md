# npu_ring_attention_update对外接口

npu_ring_attention_update(
        prev_attn_out: torch.Tensor,
        prev_softmax_max: torch.Tensor,
        prev_softmax_sum: torch.Tensor,
        cur_attn_out: torch.Tensor,
        cur_softmax_max: torch.Tensor,
        cur_softmax_sum: torch.Tensor,
        actual_seq_qlen: torch.Tensor = None,
        layout: str = "SBH",
)

小算子等价计算逻辑：
```python
import torch


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    # SBH layout
    n = prev_out_scale.shape[1]
    h = prev_attn_out.shape[-1]
    d = h // n
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    prev_out_scale = rearrange(prev_out_scale, 'b n s d -> s b (n d)').contiguous()
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    cur_out_scale = rearrange(cur_out_scale, 'b n s d -> s b (n d)').contiguous()

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    return attn_out, softmax_max, softmax_sum

```

## 前向接口：

输入：

- prev_attn_out：必选输入，数据类型torch.bfloat16, torch.float, torch.float16
- prev_softmax_max: 必选输入，数据类型torch.float
- prev_softmax_sum: 必选输入，数据类型torch.float
- cur_attn_out: 必选输入，数据类型torch.bfloat16, torch.float, torch.float16
- cur_softmax_max: 必选输入，数据类型torch.float
- cur_softmax_sum: 必选输入，数据类型torch.float


输出：

- attn_out：必选输出，数据类型torch.bfloat16, torch.float, torch.float16
- softmax_max：必选输出，数据类型torch.float
- softmax_max：必选输出，数据类型torch.float

属性：

- actual_seq_qlen：可选属性，数据类型torch.int64
- layout：必选属性，数据类型str



## 案例

```python
import torch
import torch_npu
from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update

prev_attn_out = torch.randn(2048, 1, 12, dtype=torch.bfloat16).npu()
prev_softmax_max = torch.randn(1, 12, 2048, 8, dtype=torch.float32).npu()
prev_softmax_sum = torch.randn(1, 12, 2048, 8, dtype=torch.float32).npu()
cur_attn_out = torch.randn(2048, 1, 12, dtype=torch.bfloat16).npu()
cur_softmax_max = torch.randn(1, 12, 2048, 8, dtype=torch.float32).npu()
cur_softmax_sum = torch.randn(1, 12, 2048, 8, dtype=torch.float32).npu()

attn_out, softmax_max, softmax_sum = forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                                cur_attn_out, cur_softmax_max, cur_softmax_sum)


```