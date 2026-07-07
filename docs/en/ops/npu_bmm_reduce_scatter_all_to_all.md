# npu_bmm_reducescatter_alltoall External API

```python
def npu_bmm_reducescatter_alltoall(x: Tensor,
                                   weight: Tensor,
                                   group_ep: str,
                                   group_ep_worldsize: int,
                                   group_tp: str,
                                   group_tp_worldsize: int,
                                   *,
                                   bias: Optional[Tensor] = None,
                                   shard_type: Optional[int] = 0) -> Tensor:
```

Computation logic:
BatchMatMulReduceScatterAllToAll is an operator that implements BatchMatMul computation in parallel with ReduceScatter and AllToAll collective communication.
The general computation flow is: BatchMatMul computation --> Transpose (required when shard_type equals 0) --> ReduceScatter collective communication --> Add --> AllToAll collective communication

The computation logic is as follows, where out is the final output, and x, weight, and bias are inputs:
$$
 bmmOut = BatchMatMul(x, weight)
$$
$$
 reduceScatterOut = ReduceScatter(bmmOut)
$$
$$
 addOut = Add(reduceScatterOut, bias)
$$
$$
 out = AllToAll(addOut)
$$

## Input, Output, and Attribute Description

Inputs:

- x: Required input, Tensor, data type float16, bfloat16, must be 3-dimensional. The left matrix for the BatchMatMul computation.
- weight: Required input, Tensor, data type float16, bfloat16, must be 3-dimensional, type consistent with x. The right matrix for the BatchMatMul computation.
- bias: Optional input, Tensor, data type float16, float32. When x is float16, bias must be float16; when x is bfloat16, bias must be float32. Supports two-dimensional or three-dimensional. The bias for the BatchMatMul computation. (Since ReduceScatter communication is required, the Add operation is performed after the communication).

Outputs:

- out: Tensor, data type float16, bfloat16, must be 3-dimensional. The final computation result, with type consistent with the input x.

Attributes:

- group_ep: required attribute, str. Name of the ep communication domain, the communication domain for expert parallelism.
- group_ep_worldsize: required attribute, int. Size of the ep communication domain, supporting 2/4/8/16/32.
- group_tp: required attribute, str. Name of the tp communication domain, the communication domain for tensor parallelism.
- group_tp_worldsize: required attribute, int. Size of the tp communication domain, supporting 2/4/8/16/32.
- shard_type: optional attribute, int, default value is 0. 0 indicates the output is sharded along the H dimension by tp, and 1 indicates the output is sharded along the C dimension by tp.

## Input Constraints

Due to the requirements of collective communication and BatchMatMul computation, the input and output shapes must satisfy the following mathematical relationships: (where ep=group_ep_worldsize, tp=group_tp_worldsize)

ReduceScatter scenario along the H axis, i.e., the shard_type is 0 scenario:

- x: (E/ep, ep\*C, M/tp)
- weight: (E/ep, M/tp, H)
- bias: (E/ep, 1, H/tp) or (E/ep, H/tp) when two-dimensional
- out: (E, C, H/tp)

ReduceScatter scenario along the C axis, i.e., when shard_type is 1:

- x: (E/ep, ep\*tp\*C/tp, M/tp)
- weight: (E/ep, M/tp, H)
- bias: (E/ep, 1, H)    For two dimensions: (E/ep, H)
- out: (E, C/tp, H)

Data relationship description:

- For example, if x.size(0) equals E/tp and out.size(0) equals E, then out.size(0) = ep\*x.size(0), meaning out.size(0) is an integer multiple of ep; other relationships are similar.
- The value range of E is [2, 512], and E must be an integer multiple of ep;
- The value range of H is [1, 65535]. When shard_type is 0, H must be an integer multiple of tp;
- The value range of M/tp is [1, 65535];
- The value range of E/ep is [1, 32];
- Both ep and tp only support 2, 4, 8, 16, and 32;
- The names of group_ep and group_tp cannot be the same;
- C must be greater than 0, with the upper limit being the operator's device memory limit. When shard_type is 1, C must be an integer multiple of tp;
- Cross-supernode is not supported; only intra-supernode is supported.

## Example Call of the npu_bmm_reducescatter_alltoall Class (to be verified)

The terminal invocation command is as follows:

```bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_addr 127.0.0.1  --master_port 29500 demo_test.py
```

Note that the master_addr and master_port parameters must be set by the user according to the actual situation. 8 represents ep_size*tp_size, modify it accordingly.

The sample code of demo_test.py is as follows:

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
    # Initialize the EP domain
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
                                       ep_group,
                                       ep_size,
                                       tp_group,
                                       tp_size,
                                       bias=bias_npu,
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
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        for i, y in enumerate(y_npu):
            y.cpu().numpy().tofile(f"./y_{i}.bin")

```
