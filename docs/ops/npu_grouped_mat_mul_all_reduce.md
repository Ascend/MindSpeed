# npu_grouped_mat_mul_all_reduce对外接口
```
def npu_grouped_mat_mul_all_reduce(x: List[torch.Tensor],
                                      weight: List[torch.Tensor],
                                      hcomm: str,
                                      *,
                                      bias: Optional[List[torch.Tensor]] = None,
                                      group_list: Optional[List[int]] = None,
                                      split_item: Optional[int] = 0,
                                      reduce_op: str = "sum",
                                      comm_turn: int = 0) -> List[torch.Tensor]
```

计算逻辑：
GroupedMatMulAllReduce算子是GroupedMatmul算子的多卡通信版本。它可以实现分组矩阵计算，每组矩阵乘的维度大小可以不同，是一种灵活的组合方式。输入数据x和输出数据y均支持切分或不切分模式，可以根据参数split_item来确定是否切分。当x需要切分时，使用group_list参数来描述x的m轴切分配置。本算子增加了AllReduce集合通信操作，可以把矩阵乘任务切分到多张卡上并行计算，然后通过AllReduce集合通信操作把所有卡的计算结果加和到一起，最终完成整个任务。根据输入x、weight和输出y的Tensor数量，本算子可以支持如下四种场景：
- x、weight、y的tensor数量均等于组数group_num,即每组的数据对应的tensor是独立的。
- x的tensor数量为1， weight和y的tensor数量等于组数，此时需要通过group_list来说明x在m轴方向上的分组情况。如group_list[0]=10说明x矩阵的前10行参与第一组矩阵乘计算。
- x、weight的tensor数量均等于组数group_num, y的tensor数量为1，此时每组矩阵乘的结果放在同一个输出tensor中连续存放。
- x、y的tensor数量均为1，weight的tensor数量等于组数，属于前两种情况的组合。

计算公式为：
对于每一组矩阵乘任务i: $$y_i = x_i * weight_i + bias_i$$
切分到n张卡上后，计算形式可表示为：

$$
y_i = [x_{i1}, x_{i2}, ..., x_{in}] *
\begin{bmatrix}
{weight_{i1}} \\
{weight_{i2}} \\
{...} \\
{weight_{in}}
\end{bmatrix}+\sum^{n}{bias_i/n}
$$

## 前向接口：
输入：
- x：必选输入，List[Tensor]，数据类型float16，bfloat16。支持的最大长度为64个。
- weight：必选输入，List[Tensor]，数据类型float16, bfloat16。支持的最大长度为64个。
- bias：可选输入，List[Tensor]，数据类型float16, float32。支持的最大长度为64个。对于实际无bias的场景，可以直接不指定bias参数或设置为None。
- group_list：可选输入，Optional[List[int64]]，缺省None。代表输入和输出M方向的matmul大小分布，支持的最大长度为64个。

输出：
- y：List[Tensor]，数据类型float16, bfloat16。支持的最大长度为64个。

属性：
- split_item：可选属性，int64。代表输入和输出是否要做tensor切分，0代表输入和输出都不用切分；1代表输入需要切分，输出不需要切分；2代表输入不需要切分，输出需要切分；3代表输入和输出都需要切分。缺省0。
- hcomm：必选属性，数据类型支持：string。表示通信域名称，Host侧标识列组的字符串。通过Hccl提供的接口获取。
- reduce_op：可选属性，数据类型支持：string。reduce操作类型。**当前版本仅支持输入"sum"。**
- comm_turn：可选属性，int64。Host侧的整型，通信数据切分数，即总数据量/单次通信量。**当前版本仅支持输入0。**


## 反向接口
无

## 输入限制
- List支持的最大长度为64； 
- 属性reduce_op仅支持输入"sum"；
- 属性comm_turn仅支持输入0；
- 通信支持2、4、8卡。
- 当split_item取0或2时，x的tensor数量与weight的相同；当split_item取1或3时，x的tensor数量为1。
- 当split_item取0或2时，group_list为空；当split_item取1或3时，group_list的长度与weight的tensor个数相同。
- 若bias不是空，其tensor数量要与weight的相同。
- 满足矩阵乘的m/k/n维度关系。
- 支持输入元素类型：
  1. x中元素类型为float16，weight中元素类型为float16，bias中元素类型为float16，输出y中元素类型为float16；

  2. x中元素类型为bfloat16，weight中元素类型为bfloat16，bias中元素类型为float32，输出y中元素类型为bfloat16；
- 输入中须包含通信域hcomm字符串，它需要从torch.distributed包中的接口获取。
- 当前只支持PyTorch2.1版本。

## npu_grouped_mat_mul_all_reduce 类的调用方式

```python
import os
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import math
from mindspeed.ops.npu_grouped_mat_mul_all_reduce import npu_grouped_mat_mul_all_reduce


def get_hcomm_info(world_size, rank):
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method='tcp://127.0.0.1:5001')
    print(f'device_{rank} init_process_group success.')
    if dist.is_available():
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    print(f'device_{rank} get_hccl_comm_name success.')
    return dist, hcomm_info


def run(rank, n_device):
    torch_npu.npu.set_device(rank)
    dist, hcomm_info = get_hcomm_info(n_device, rank)
  
    x1 = torch.randn(32, 256, dtype=torch.float16)
    x1_npu = x1.npu()
    w1 = torch.randn(256, 32, dtype=torch.float16)
    w1_npu = w1.npu()
  
    x2 = torch.randn(64, 128, dtype=torch.float16)
    x2_npu = x2.npu()
    w2 = torch.randn(128, 64, dtype=torch.float16)
    w2_npu = w2.npu()
  
    group_list = None
    split_item = 0
    hccl_group = hcomm_info
    reduce_op = "sum"
    comm_turn = 0
  
    # 正向接口案例
    x1_npu.requires_grad = False
    x2_npu.requires_grad = False
    w1_npu.requires_grad = False
    w2_npu.requires_grad = False
    y_npu = npu_grouped_mat_mul_all_reduce([x1_npu, x2_npu], [w1_npu, w2_npu], hccl_group, bias=[],
                                           group_list=group_list, split_item=split_item, reduce_op=reduce_op,
                                           comm_turn=comm_turn)
    if rank == 0:
        for i, y in enumerate(y_npu):
            y.cpu().numpy().tofile(f"./y_{i}.bin")


if __name__ == '__main__':
    n_device = 2
    mp.spawn(run, args=(n_device,), nprocs=n_device)

```
