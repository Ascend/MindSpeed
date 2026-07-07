# npu_mm_all_reduce_add_rms_norm_ External Interface

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-30T02:30:05.901Z pushedAt=2026-06-30T02:45:13.039Z -->

CLASS InplaceMatmulAllReduceAddRmsNorm()

Computation logic:
$$
mmOut = allReduce(x1*x2 + bias)
$$
$$
y = mmOut + residual
$$
$$
normOut = \frac{y}{RMS(y)}*gamma, RMS(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} y_{i}^{2} + epsilon}
$$

## Non-Quantization Scenario

Inputs:

- x1: Required Input, data type float16, bfloat16
- x2: Required Input, data type float16, bfloat16
- residual: Required Input, data type float16, bfloat16
- gamma: Required Input, data type float16, bfloat16
- hcom: Required Input, data type string,
- reduce_op: Optional input, data type string, currently only supports sum
- epsilon: Optional input, data type float, default case is 1e-06
- bias: Optional input, data type float16, bfloat16
- antiquant_scale: Optional input, default case is nullptr
- antiquant_offset: Optional input, default case is nullptr
- dequant_scale: Optional Input, defaults to nullptr in this scenario
- antiquant_group_size: Optional Input, defaults to 0 in this scenario
- comm_turn: Optional Input, Data Type int, defaults to 0

Outputs:

- residual: Required Output, Reuse residual, Data Type float16, bfloat16
- normOut: Required Output, Data Type float16, bfloat16

## Full Quantization Scenario

Inputs:

- x1: Required Input, Data Type int8
- x2: Required Input, Data Type int8
- residual: Required Input, data type float16, bfloat16
- gamma: Required Input, data type float16, bfloat16
- hcom: Required Input, data type string,
- reduce_op: Optional Input, data type string, currently only supports sum
- epsilon: Optional Input, data type float, default case is 1e-06
- bias: Optional input, data type int32
- antiquant_scale: Optional input, defaults to nullptr in this scenario
- antiquant_offset: Optional input, defaults to nullptr in this scenario
- dequant_scale: Optional input, data type int64, uint64, bfloat16
- antiquant_group_size: Optional input, defaults to 0 in this scenario
- comm_turn: Optional input, data type int, default case is 0

Outputs:

- residual: Required output, reuse residual, data type float16, bfloat16
- normOut: Required output, data type float16, bfloat16

## Pseudo-Quantization Scenario

Inputs:

- x1: Required Input, Data Type float16, bfloat16
- x2: Required Input, Data Type int8
- residual: Required Input, Data Type float16, bfloat16
- gamma: Required Input, Data Type float16, bfloat16
- hcom: Required Input, data type string,
- reduce_op: Optional Input, data type string, currently only supports sum
- epsilon: Optional Input, data type float, default case is 1e-06
- bias: Optional Input, data type float16, bfloat16
- antiquant_scale: Optional Input, data type float16, bfloat16
- antiquant_offset: Optional input, data type float16, bfloat16
- dequant_scale: Optional input, default is nullptr in this scenario
- antiquant_group_size: Optional input, data type int, default is 0
- comm_turn: Optional input, data type int, default is 0

Outputs:

- residual: Required Output, reuse residual, data type float16, bfloat16
- normOut: Required Output, data type float16, bfloat16

## Input Constraints

- ``x2`` only supports non-contiguous tensor input with the last two axes transposed; inputs such as ``x1``, ``residual``, and ``gamma`` only support contiguous tensors
- Only the ND data format is supported
- ``x1`` supports two or three dimensions, with dimensions ``(b, s, k)`` or ``(s, k)``
- ``x2`` only supports two dimensions, with dimensions ``(k, n)``. The axes of ``x1`` and ``x2`` must meet the input requirements of the matmul operator, with the k-axis being equal
- ``bias``, when non-empty, is 1-dimensional with dimensions ``(n)``
- ``residual`` only supports three dimensions, with dimensions ``(b, s, n)``. When ``x1`` is two-dimensional, ``(b * s)`` of ``residual`` equals ``s`` of ``x1``. When ``x1`` is three-dimensional, ``(b * s)`` of ``residual`` equals ``(b * s)`` of ``x1``; the last dimension of ``residual`` equals the last dimension of ``x2``
- ``gamma`` only supports one dimension, with dimensions ``(n)``. The last dimension of ``gamma`` equals the last dimension of ``residual``
- ``reduce_op`` only supports ``sum``
- Atlas A2 AI processors support 1, 2, 4, or 8 cards, and only support hccs link all-mesh networking
- Atlas A2 AI processors support empty tensors where ``(b * s)`` and ``n`` are 0, but do not support empty tensors where ``k`` is 0
- In the non-quantization scenario, the data types of the compute inputs ``x1``, ``x2``, ``bias`` (if supported), ``residual``, and ``gamma`` must be consistent
- For Atlas A2 AI processors, in the non-quantization scenario, the ranges of ``(b * s)``, ``k``, and ``n`` are ``[1, 2147483647]``
- In the Full Quantization Scenario, if the output ``residual`` type is ``FLOAT16``, the type of ``dequant_scale`` is ``INT64`` or ``UINT64`` (``dequant_scale`` needs to be processed via the ``torch_npu.npu_trans_quant_param()`` interface); if the output ``residual`` type is ``BFLOAT16``, the type of ``dequant_scale`` is ``BFLOAT16``. ``dequant_scale`` supports two modes:
    - ``per_tensor`` mode: ``(1,)``
    - ``per_channel`` mode: ``(1, n)`` or ``(n,)``
- In the Full Quantization Scenario, the data types of ``x1`` and ``x2`` are ``int8``, the data type of ``bias`` (if supported) is ``int32``, and the data types of the ``residual`` and ``gamma`` compute inputs must be consistent.
- In the Full Quantization Scenario, the size of ``m`` does not exceed 2147483647, the size of the last dimension of ``x1`` and ``x2`` does not exceed 65535, where the last dimension of ``x1`` refers to ``k``, and the last dimension of ``x2`` refers to ``k`` when transposed or ``n`` when not transposed.
- In the pseudo-quantization scenario, the range of ``m`` is ``[1, 2147483647]``, and the ranges of ``k`` and ``n`` are ``[1, 65535]``.
- In the pseudo-quantization scenario, ``antiquant_scale`` supports three modes:
    - ``per_tensor`` mode: ``(1,)``
    - ``per_channel`` mode: ``(1, n)`` or ``(n,)``
    - ``per_group`` mode: ``(ceil(k,antiquant_group_size),n)``
- If ``antiquant_offset`` is not empty, its shape is consistent with ``antiquant_scale``.
- In the pseudo-quantization scenario, the data type of ``x2`` must be ``int8``, and the data types of the compute inputs ``x1``, ``bias`` (if supported), ``residual``, ``gamma``, ``antiquant_scale``, and ``antiquant_offset`` must be consistent.
- In the pseudo-quantization scenario, the value of ``antiquant_group_size`` must satisfy the range ``[32, min(k-1, INT_MAX)]`` and be a multiple of 32.
- The MC2 fused operators within a model only support the same communication domain.

## Example Call of npu_mm_all_reduce_add_rms_norm

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from mindspeed.ops.npu_mm_all_reduce_add_rms_norm_ import npu_mm_all_reduce_add_rms_norm_


def run_mm_all_reduce_add_rms_norm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, residual_shape,
                                   gamma_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)

    x1 = torch.randn(x1_shape, dtype=dtype).npu()
    x2 = torch.randn(x2_shape, dtype=dtype).npu()
    residual = torch.randn(residual_shape, dtype=dtype).npu()
    gamma = torch.randn(gamma_shape, dtype=dtype).npu()
    epsilon = 0.000001
    y, norm_out = npu_mm_all_reduce_add_rms_norm_(x1=x1, x2=x2, residual=residual, gamma=gamma, hcom=hcom_info,
                                                  reduce_op='sum', epsilon=epsilon)
    print("y:", y)
    print("norm_out:", norm_out)


if __name__ == "__main__":
    worksize = 8
    master_ip = "127.0.0.1"
    master_port = '50001'
    b, s, k, n = 4, 1024, 1024, 8192
    x1_shape = (b, s, k)
    x2_shape = (k, n)
    residual_shape = (b, s, n)
    gamma_shape = (n)
    dtype = torch.float16

    mp.spawn(run_mm_all_reduce_add_rms_norm,
             args=(worksize, master_ip, master_port, x1_shape, x2_shape, residual_shape, gamma_shape, dtype),
             nprocs=worksize)
```
