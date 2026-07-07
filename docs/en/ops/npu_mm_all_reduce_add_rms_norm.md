# npu_mm_all_reduce_add_rms_norm External Interface

CLASS MatmulAllReduceAddRmsNorm()

Computation Logic:
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
- epsilon: Optional input, data type float, default is 1e-06
- bias: Optional input, data types float16, bfloat16
- antiquant_scale: Optional input, default is nullptr in this scenario
- antiquant_offset: Optional input, default is nullptr in this scenario
- dequant_scale: Optional Input, default is nullptr in this scenario
- antiquant_group_size: Optional Input, default is 0 in this scenario
- comm_turn: Optional Input, Data Type is int, default is 0

Outputs:

- y: Required Output, Data Type float16, bfloat16
- normOut: Required Output, Data Type float16, bfloat16

## Full Quantization Scenario

Inputs:

- x1: Required Input, Data Type int8
- x2: Required Input, Data Type int8
- residual: Required Input, data type float16, bfloat16
- gamma: Required Input, data type float16, bfloat16
- hcom: Required Input, data type string,
- reduce_op: Optional Input, data type string, currently only supports sum
- epsilon: Optional Input, data type float, 1e-06 by default
- bias: Optional Input, Data Type int32
- antiquant_scale: Optional Input, default is nullptr in this scenario
- antiquant_offset: Optional Input, default is nullptr in this scenario
- dequant_scale: Optional Input, Data Type int64, uint64, bfloat16
- antiquant_group_size: Optional Input, default is 0 in this scenario
- comm_turn: Optional Input, Data Type int, 0 in Default Case

Outputs:

- y: Required Output, Data Type float16, bfloat16
- normOut: Required Output, Data Type float16, bfloat16

## Pseudo-Quantization Scenario

Inputs:

- x1: Required Input, data type float16, bfloat16
- x2: Required Input, data type int8
- residual: Required Input, data type float16, bfloat16
- gamma: Required Input, data type float16, bfloat16
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

- y: Required Output, data type float16, bfloat16
- normOut: Required Output, data type float16, bfloat16

## Input Constraints

- ``x2`` only supports non-contiguous tensor input when the last two axes are transposed. Inputs such as ``x1``, ``residual``, and ``gamma`` only support contiguous tensors.
- Only supports ND data format.
- ``x1`` supports two or three dimensions, with dimensions ``(b, s, k)`` or ``(s, k)``
- ``x2`` only supports two dimensions, with dimensions ``(k, n)``. The axes of ``x1`` and ``x2`` must satisfy the input requirements of the matmul operator, with the k-axis being equal
- ``bias``, when non-empty, is 1-dimensional with dimensions ``(n)``
- ``residual`` only supports three dimensions, with dimensions ``(b, s, n)``. When ``x1`` is two-dimensional, ``(b * s)`` of ``residual`` equals ``s`` of ``x1``; when ``x1`` is three-dimensional, ``(b * s)`` of ``residual`` equals ``(b * s)`` of ``x1``. The last dimension of ``residual`` is equal to the last dimension of ``x2``
- ``gamma`` only supports one dimension, with dimensions ``(n)``. The last dimension of ``gamma`` is equal to the last dimension of ``residual``
- ``reduce_op`` only supports ``sum``
- 昇腾 Atlas A2 AI processors support 1, 2, 4, and 8 cards, and only support hccs link all-mesh networking
- 昇腾 Atlas A2 AI processors support empty tensors where ``(b * s)`` and ``n`` are 0, but do not support empty tensors where ``k`` is 0
- In the non-quantization scenario, the data types of the compute inputs ``x1``, ``x2``, ``bias`` (if supported), ``residual``, and ``gamma`` must be consistent
- For 昇腾 Atlas A2 AI processors, in the non-quantization scenario, the range of ``(b * s)``, ``k``, and ``n`` is ``[1, 2147483647]``
- In the full quantization scenario, if the output ``residual`` type is ``FLOAT16``, the type of ``dequant_scale`` is ``INT64`` or ``UINT64`` (``dequant_scale`` needs to be processed through the ``torch_npu.npu_trans_quant_param()`` interface); if the output ``residual`` type is ``BFLOAT16``, the type of ``dequant_scale`` is ``BFLOAT16``. ``dequant_scale`` supports two modes:
    - ``per_tensor`` mode: ``(1,)``
    - ``per_channel`` mode: ``(1, n)`` or ``(n,)``
- In the full quantization scenario, the data types of ``x1`` and ``x2`` are ``int8``, the data type of ``bias`` (if supported) is ``int32``, and the data types of the ``residual`` and ``gamma`` compute inputs must be consistent.
- In the full quantization scenario, the size of ``m`` does not exceed 2147483647, the last dimension size of ``x1`` and ``x2`` does not exceed 65535, where the last dimension of ``x1`` refers to ``k``, and the last dimension of ``x2`` refers to ``k`` when transposed or ``n`` when not transposed.
- In the pseudo-quantization scenario, the range of ``m`` is ``[1, 2147483647]``, and the ranges of ``k`` and ``n`` are ``[1, 65535]``.
- In the pseudo-quantization scenario, ``antiquant_scale`` supports three modes:
    - ``per_tensor`` mode: ``(1,)``
    - ``per_channel`` mode: ``(1, n)`` or ``(n,)``
    - ``per_group`` mode: ``(ceil(k, antiquant_group_size), n)``
- If ``antiquant_offset`` is not empty, its shape is consistent with ``antiquant_scale``.
- In the pseudo-quantization scenario, the data type of ``x2`` must be ``int8``, and the data types of ``x1``, ``bias`` (if supported), ``residual``, ``gamma``, ``antiquant_scale``, and ``antiquant_offset`` compute inputs must be consistent.
- In the pseudo-quantization scenario, the value of ``antiquant_group_size`` must fall within the range ``[32, min(k-1, INT_MAX)]`` and be a multiple of 32.
- The fused compute MC2 operator within a model only supports the same communication domain.

## Example Call of npu_mm_all_reduce_add_rms_norm

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from mindspeed.ops.npu_mm_all_reduce_add_rms_norm import npu_mm_all_reduce_add_rms_norm


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
    y, norm_out = npu_mm_all_reduce_add_rms_norm(x1=x1, x2=x2, residual=residual, gamma=gamma, hcom=hcom_info,
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
