# npu_alltoall_allgather_bmm对外接口
```
def npu_alltoall_allgather_bmm(
    x: Tensor,
    weight: Tensor,
    *,
    bias: Optional[Tensor] = None,
    group_ep: str,
    group_ep_worldsize: int,
    group_tp: str,
    group_tp_worldsize: int,
    shard_type: Optional[int] = 0,
    act_type: Optional[str] = "None",
    need_allgather_out: Optional[bool] = False,
    need_activation_feature: Optional[bool] = False
) -> (Tensor, Tensor, Tensor):

```

计算逻辑：
bmm指BatchMatMul，AllToAllAllGahterBatchMatMul算子是实现AllToAll、AllGather集合通信与BatchMatMul计算并行的算子。
大体计算流程为：AllToAll集合通信-->AllGather集合通信-->BatchMatMul-->激活（可选，可以没有）

$$
计算逻辑如下，其中y1 y2 y3为输出
x1 = AllToAll(x)
y2 = AllGather(x1)
y3 = BatchMatMul(y2, weight, bias)
y1 = 激活函数(y3)
$$

## 输入输出及属性说明：
输入：
- x：必选输入，Tensor，数据类型支持float16，bfloat16。该输入进行AllToAll、AllGather集合通信，必须为3维，数据格式支持ND，通信后结果作为BatchMatMul计算的左矩阵；
- weight：必选输入，Tensor，数据类型支持float16, bfloat16，类型需与x保持一致，必须为3维，数据格式支持ND。BatchMatMul计算的右矩阵
- bias：可选输入，Tensor，数据类型支持float16, float32。x为float16时，bias需为float16；x为bfloat16时，bias需为float32，必须为两维或三维，数据格式支持ND。BatchMatMul计算的bias。

输出：
- y1：Tensor，数据类型支持float16, bfloat16，仅支持3维。最终计算结果，如果有激活函数则为激活函数的输出，否则为BatchMatMul的输出。数据类型与输入x保持一致。
- y2：Tensor，可选输出，数据类型支持float16, bfloat16，仅支持3维。AllGather的输出，数据类型与输入x保持一致。反向可能需要。
- y3：Tensor，可选输出，数据类型支持float16, bfloat16，仅支持3维。有激活函数时，BatchMatMul的输出，类型与输入x保持一致。

属性：
- group_ep：必选属性，str。ep通信域名称，专家并行的通信域。
- group_ep_worldsize：必选属性，int。ep通信域size，支持2/4/8/16。
- group_tp：必选属性，str。tp通信域名称，Tensor并行的通信域。
- group_tp_worldsize：必选属性，int。tp通信域size，支持2/4/8/16。
- shard_type：可选属性，int，默认值为0，0表示在H维度按tp域进行allgather，1表示在C维度上按tp域进行allgather。当前仅支持shard_type等于1的场景。
- act_type：可选属性，str，激活函数类型，默认值为None，表示无激活函数。支持GELU/Silu/FastGELU/Relu/None等。
- need_allgather_out：是否需要输出allgather后的结果，默认False，表示不需要输出。
- need_activation_feature：是否需要输出执行激活函数前的结果（BatchMatMul后），默认False，表示不需要输出。仅在act_type不为0的时候有意义。


## 输入shape限制
因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=ep_world_size，tp=tp_world_size）
- x: (E,C/tp,H)；
- weight：(E/ep,H,M/tp)；
- bias：(E/ep,1,M/tp)；  支持两维或三维，两维时shape为：(E/ep, M/tp)
- y1：(E/ep,ep*tp*C/tp,M/tp)；
- y2：(E/ep,ep*tp*C/tp,H)；
- y3：(E/ep,ep*tp*C/tp,M/tp)
数据关系说明：
1、比如x.size(0)等于E，weight.size(0)等于E/ep，则表示，x.size(0) = ep*weight.size(0)，x.size(0)是ep的整数倍；其他关系类似
2、E的取值范围为[2, 2048]，且E是ep的整数倍；
3、H的取值范围为：[1, 65535]；
4、M/tp的取值为：[1, 65535]；
5、ep、tp均仅支持2、4、8、16；
6、C大于0，上限为算子device内存上限；

## npu_alltoall_allgather_bmm 的调用示例

```python
import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm

world_size = 8
ep_size = 4
tp_size = 2
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

if __name__ == '__main__':
    dtype = torch.float16
    x_shard_type = 1
    out_y2_flag = True
    out_y3_flag = False
    act_type = "None"
    transpose_weight = False
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    ep_group, tp_group = get_ep_tp_hcomm_info(rank, ep_size, tp_size)
    print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {ep_group}, {tp_group}')
    E, C, H, M = 4, 1024, 1024, 8192
    if x_shard_type == 0:
        x_shape = (E, C, H / tp_size)
    elif x_shard_type == 1:
        x_shape = (E, C / tp_size, H)
    else:
        x_shape = (E / ep_size, tp_size * ep_size * C, M / tp_size)
    weight_shape = (E / ep_size, H, M / tp_size)
    if transpose_weight == True:
        weight_shape = (E / ep_size, M / tp_size, H)
    bias_shape = (E / ep_size, 1, M / tp_size)
    
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
    
    y_npu = npu_alltoall_allgather_bmm(x_npu,
                                       weight_npu,
                                       bias=bias_npu,
                                       group_ep=ep_group,
                                       group_ep_worldsize=ep_size,
                                       group_tp=tp_group,
                                       group_tp_worldsize=tp_size,
                                       shard_type=x_shard_type,
                                       act_type=act_type,
                                       need_allgather_out=out_y2_flag,
                                       need_activation_feature=out_y3_flag)
    if rank == 0:
        for i, y in enumerate(y_npu):
            y.cpu().numpy().tofile(f"./y_{i}.bin")

```
