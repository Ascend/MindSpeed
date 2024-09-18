# npu_bmm_reducescatter_alltoall对外接口
```
def npu_bmm_reducescatter_alltoall(x: Tensor,
                                   weight: Tensor,
                                   *,
                                   bias: Optional[Tensor] = None,
                                   group_ep: str,
                                   group_ep_worldsize: int,
                                   group_tp: str,
                                   group_tp_worldsize: int,
                                   shard_type: Optional[int] = 0) -> Tensor:
```

计算逻辑：
BatchMatMulReduceScatterAllToAll是实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。
大体计算流程为：BatchMatMul计算-->转置（shard_type等于0时需要）-->ReduceScatter集合通信-->Add-->AllToAll集合通信

$$
计算逻辑如下，其中y为输出
temp1 = BatchMatMul(x，weight)
temp2 = ReduceScatter(temp1)
temp3 = Add(temp2, bias)
y = AllToAll(temp3)
$$

## 输入输出及属性说明：
输入：
- x：必选输入，Tensor，数据类型float16，bfloat16，必须为3维。BatchMatMul计算的左矩阵；
- weight：必选输入，Tensor，数据类型float16, bfloat16，必须为3维，类型与x保持一致。BatchMatMul计算的右矩阵
- bias：可选输入，Tensor，数据类型float16, float32。x为float16时，bias需为float16；x为bfloat16时，bias需为float32。支持两维或三维。BatchMatMul计算的bias。(由于要进行ReduceScatter通信，因此需要在通信之后再Add)

输出：
- y：Tensor，数据类型float16, bfloat16，必须为3维。最终计算结果，类型与输入x保持一致。

属性：
- group_ep：必选属性，str。ep通信域名称，专家并行的通信域。
- group_ep_worldsize：必选属性，int。ep通信域size，支持2/4/8/16。
- group_tp：必选属性，str。tp通信域名称，Tensor并行的通信域。
- group_tp_worldsize：必选属性，int。tp通信域size，支持2/4/8/16。
- shard_type：可选属性，int，默认值为0。0表示输出在H维度按tp分片，1表示输出在C维度按tp分片。当前仅支持shard_type等于1的场景。


## 输入限制
因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=ep_world_size，tp=tp_world_size）
按H轴进行ReduceScatter场景，即shard_type为0场景（暂不支持该场景）：
- x: (E/ep, ep*C, M/tp) 
- weight：(E/ep, M/tp, H)
- bias：(E/ep, 1, H/tp)  两维时为(E/ep, H/tp)
- y：(E, C, H/tp)

按C轴进行ReduceScatter场景，即shard_type为1场景：
- x: (E/ep, ep*tp*C/tp, M/tp) 
- weight：(E/ep, M/tp, H)
- bias：(E/ep, 1, H)    两维时为(E/ep, H)
- y：(E, C, H)

数据关系说明：
1、比如x.size(0)等于E/tp，y.size(0)等于E，则表示，y.size(0) = ep*x.size(0)，y.size(0)是ep的整数倍；其他关系类似
2、E的取值范围为[2, 2048]，且E是ep的整数倍；
3、H的取值范围为：[1, 65535]；
4、M/tp的取值为：[1, 65535]；
5、ep、tp均仅支持2、4、8、16；
6、C大于0，上限为算子device内存上限；

## npu_bmm_reducescatter_alltoall 类的调用示例(待验证)

```python
import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall

world_size = 8
ep_size = 4
tp_size = 2
def get_hcomm_info(n, i):
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(i)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(i)
    return hcomm_info

def setup_ep_tp(rank, tp_size, ep_size, backend_type):
    # 初始化EP域
    print("device %d initialize ep group" % rank, flush=True)
    for i in range(tp_size):
        ep_ranks = [x + ep_size * i for x in range(ep_size)]
        ep_group = dist.new_group(backend=backend_type, ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_tmp = ep_group
    print("device %d initialize tp group" % rank, flush=True)
    for i in range(ep_size):
        tp_ranks = [x * ep_size + i for x in range(tp_size)]
        tp_group = dist.new_group(backend=backend_type, ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_tmp = tp_group
    return ep_group_tmp, tp_group_tmp

def get_ep_tp_hcomm_info(rank, ep_size, tp_size):
    ep_group, tp_group = setup_ep_tp(rank, tp_size, ep_size, "hccl")
    if torch.__version__ > '2.0.1':
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        tp_hcomm_info = tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        ep_hcomm_info = ep_group.get_hccl_comm_name(rank)
        tp_hcomm_info = tp_group.get_hccl_comm_name(rank)
    return ep_hcomm_info, tp_hcomm_info

def test_npu_bmm_reducescatter_alltoall(dtype, y_shard_type, transpose_weight):
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    ep_group, tp_group = get_ep_tp_hcomm_info(rank, ep_size, tp_size)
    hcomm_info = get_hcomm_info(world_size, rank)
    print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {ep_group}, {tp_group}')
    E, C, H, M = 4, 1024, 1024, 8192
    if y_shard_type == 0:
        x_shape = (E / ep_size, ep_size * C, M / tp_size)
        bias_shape = (E / ep_size, 1, H / tp_size)
    else:
        x_shape = (E / ep_size, tp_size * ep_size * C, M / tp_size)
        bias_shape = (E / ep_size, 1, H)
    weight_shape = (E / ep_size, M / tp_size, H)
    if transpose_weight == True:
        weight_shape = (E / ep_size, H, M / tp_size)
    
    x_shape = tuple(int(item) for item in x_shape)
    weight_shape = tuple(int(item) for item in weight_shape)
    bias_shape = tuple(int(item) for item in bias_shape)
    x = torch.rand(x_shape)
    weight = torch.rand(weight_shape)
    bias = torch.rand(bias_shape)
    x_npu = x.npu().to(dtype)
    weight_npu = weight.npu().to(dtype)
    if transpose_weight == True:
        print(f'!!!!before transpose, weight_npu.size()={weight_npu.size()}')
        weight_npu = weight_npu.transpose(1, 2)
        print(f'!!!!after transpose, weight_npu.size()={weight_npu.size()}')
        print(f'!!!!after transpose, weight_npu.is_contiguous()={weight_npu.is_contiguous()}')
    bias_npu = bias.npu().to(dtype)
    
    y = npu_bmm_reducescatter_alltoall(x_npu,
                                       weight_npu,
                                       bias=bias_npu,
                                       group_ep=ep_group,
                                       group_ep_worldsize=ep_size,
                                       group_tp=tp_group,
                                       group_tp_worldsize=tp_size,
                                       shard_type=y_shard_type)
    print(f'y_shape = {y.size()}')
    if y_shard_type == 0:
        assert y.size() == (E, C, int(H / tp_size))
    else:
        assert y.size() == (E, C, H)
    return y

if __name__ == '__main__':
    dtype = torch.float16
    shard_type = 1
    transpose_weight = False
    y_npu = test_npu_bmm_reducescatter_alltoall(dtype, shard_type, transpose_weight)
    if rank == 0:
        for i, y in enumerate(y_npu):
            y.cpu().numpy().tofile(f"./y_{i}.bin")

```
