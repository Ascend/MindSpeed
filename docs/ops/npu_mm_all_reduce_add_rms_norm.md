# npu_mm_all_reduce_add_rms_norm对外接口

CLASS MatmulAllReduceAddRmsNorm()

计算逻辑：
$$
mmOut = allReduce(x1*x2 + bias)
$$
$$
y = mmOut + residual
$$
$$
normOut = \frac{y}{RMS(y)}*gamma, RMS(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} y_{i}^{2} + epsilon}
$$

## 非量化场景：
输入：
- x1：必选输入，数据类型float16, bfloat16	
- x2：必选输入，数据类型float16, bfloat16	
- residual：必选输入，数据类型float16, bfloat16
- gamma：必选输入，数据类型float16, bfloat16
- hcom：必选输入，数据类型string,
- reduce_op：可选输入，数据类型为string，当前仅支持sum
- epsilon：可选输入，数据类型为float，缺省情况下为1e-06
- bias：可选输入，数据类型float16, bfloat16
- antiquant_scale：可选输入，该场景默认为nullptr
- antiquant_offset：可选输入，该场景默认为nullptr
- dequant_scale：可选输入，该场景默认为nullptr
- antiquant_group_size：可选输入，该场景默认为0
- comm_turn：可选输入，数据类型为int,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 全量化场景
输入：
- x1：必选输入，数据类型int8
- x2：必选输入，数据类型int8
- residual：必选输入，数据类型float16, bfloat16
- gamma：必选输入，数据类型float16, bfloat16
- hcom：必选输入，数据类型string,
- reduce_op：可选输入，数据类型为string，当前仅支持sum
- epsilon：可选输入，数据类型为float，缺省情况下为1e-06
- bias：可选输入，数据类型int32
- antiquant_scale：可选输入，该场景默认为nullptr
- antiquant_offset：可选输入，该场景默认为nullptr
- dequant_scale：可选输入，数据类型int64，uint64，bfloat16
- antiquant_group_size：可选输入，该场景默认为0
- comm_turn：可选输入，数据类型为int,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 伪量化场景
输入：
- x1：必选输入，数据类型float16, bfloat16	
- x2：必选输入，数据类型int8
- residual：必选输入，数据类型float16, bfloat16
- gamma：必选输入，数据类型float16, bfloat16
- hcom：必选输入，数据类型string,
- reduce_op：可选输入，数据类型为string，当前仅支持sum
- epsilon：可选输入，数据类型为float，缺省情况下为1e-06
- bias：可选输入，数据类型float16, bfloat16
- antiquant_scale：可选输入，数据类型float16, bfloat16
- antiquant_offset：可选输入，数据类型float16, bfloat16
- dequant_scale：可选输入，该场景默认为nullptr
- antiquant_group_size：可选输入，数据类型为int，缺省情况下为0
- comm_turn：可选输入，数据类型为int,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 输入限制
- ``x2`` 仅支持最后两轴转置情况下的非连续tensor传入，``x1``、``residual``、``gamma`` 等输入仅支持连续tensor 
- 仅支持ND数据格式
- ``x1`` 支持两维或者三维，其维度为 ``(b, s, k)`` 或者 ``(s, k)``
- ``x2`` 仅支持两维，其维度为 ``(k, n)``，``x1`` 和 ``x2`` 的轴满足matmul算子入参要求，k轴相等
- ``bias`` 在非空情况下为1维，其维度为 ``(n)``
- ``residual`` 仅支持三维，其维度为 ``(b, s, n)``，当 ``x1`` 为两维时，``residual`` 的 ``(b * s)`` 等于 ``x1`` 的 ``s``，当 ``x1`` 为三维时，``residual`` 的 ``(b * s)`` 等于 ``x1`` 的 ``(b * s)``;``residual`` 的最后一维与``x2`` 的最后一维相等
- ``gamma`` 仅支持一维，其维度为 ``(n)``，``gamma`` 的最后一维与 ``residual`` 的最后一维相等
- ``reduce_op`` 仅支持 ``sum``
- 昇腾910B AI处理器支持1、2、4、8卡
- 昇腾910B AI处理器支持``(b * s)``，``n``为0的空tensor，不支持``k``为0的空tensor
- 非量化场景下，``x1``、``x2``、``bias``（若支持）、``residual``、``gamma`` 计算输入的数据类型要一致
- 昇腾910B AI处理器，在非量化场景下，``(b * s)``、``k``、``n``的范围为``[1, 2147483647]``
- 全量化场景下，若输出 ``residual`` 类型为 ``FLOAT16``，``dequant_scale`` 的类型为 ``INT64``、``UINT64``（需通过 ``torch_npu.npu_trans_quant_param()`` 接口对 ``dequant_scale`` 进行处理）；若输出 ``residual`` 类型为 ``BFLOAT16``，``dequant_scale`` 的类型为 ``BFLOAT16``。``dequant_scale`` 满足两种模式：
    - ``per_tensor`` 模式：``(1,)``
    - ``per_channel`` 模式：``(1, n)`` 或 ``(n,)``
- 全量化场景下，``x1``、``x2`` 数据类型为 ``int8``，``bias``（若支持）数据类型为 ``int32``，``residual``、``gamma``计算输入的数据类型要一致。
- 全量化场景下，``x1``与``x2``的最后一维大小不超过65535，``x1``的最后一维指``k``，``x2``的最后一维指转置时的``k``或非转置时的``n``。
- 伪量化场景下，``k``、``n``的范围为``[1,65535]``
- 伪量化场景下，``antiquant_scale`` 满足三种模式：
    - ``per_tensor`` 模式：``(1,)``
    - ``per_channel`` 模式：``(1, n)`` 或 ``(n,)``
    - ``per_group`` 模式：``(ceil(k,antiquant_group_size),n)``
- ``antiquantOffset`` 若非空，shape 与 ``antiquant_scale``一致。
- 伪量化场景下，``x2`` 的数据类型需为 ``int8``，``x1``、``bias``（若支持）、``residual``、``gamma``、``antiquant_scale``、``antiquant_offset``计算输入的数据类型要一致。
- 伪量化场景下，``antiquant_group_size`` 取值满足取值范围``[32, min(k-1, INT_MAX)]``且为32倍数。

## npu_mm_all_redcue_add_rms_norm 接口的调用方式

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