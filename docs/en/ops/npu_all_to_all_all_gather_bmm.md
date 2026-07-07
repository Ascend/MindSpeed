# npu_alltoall_allgather_bmm External API

```python
def npu_alltoall_allgather_bmm(
    x: Tensor,
    weight: Tensor,
    group_ep: str,
    group_ep_worldsize: int,
    group_tp: str,
    group_tp_worldsize: int,
    *,
    bias: Optional[Tensor] = None,
    shard_type: Optional[int] = 0,
    act_type: Optional[str] = "None",
    need_allgather_out: Optional[bool] = False,
    need_activation_feature: Optional[bool] = False
) -> Tuple[Tensor, Tensor, Tensor]:

```

Computation logic:
bmm refers to BatchMatMul. The AllToAllAllGatherBatchMatMul operator implements the parallelization of AllToAll and AllGather collective communication with BatchMatMul computation.
The general computation flow is: AllToAll collective communication --> AllGather collective communication --> BatchMatMul --> activation (optional, can be omitted)

The computation logic is as follows, where y1Out, y2OutOptional, and y3OutOptional are outputs, x, weight, and bias are inputs, and activating is the activation function (determined by act_type; when act_type is None, it means no activation function is called)
$$
 alltoallOut = AllToAll(x)
$$
$$
 y2OutOptional = AllGather(alltoallOut)
$$
$$
 y3OutOptional = BatchMatMul(y2OutOptional, weight, bias)
$$
$$
 y1Out = activating(y3OutOptional)
$$

## Input, Output, and Attribute Description

Inputs:

- x: Required input, Tensor, data type supports float16, bfloat16. This input undergoes AllToAll and AllGather collective communication, must be 3-dimensional, data format supports ND, and the result after communication serves as the left matrix for the BatchMatMul computation.
- weight: Required input, Tensor, data type supports float16, bfloat16, must be consistent with x, must be 3-dimensional, data format supports ND, and serves as the right matrix for the BatchMatMul computation.
- bias: Optional input, Tensor, data type supports float16, float32. When x is float16, bias must be float16; when x is bfloat16, bias must be float32, must be 2-dimensional or 3-dimensional, data format supports ND. The bias for the BatchMatMul computation.

Outputs:

- y1Out: Tensor, data type supports float16, bfloat16, only supports 3 dimensions. The final computation result. If there is an activation function, it is the output of the activation function; otherwise, it is the output of BatchMatMul. The data type is consistent with the input x.
- y2OutOptional: Tensor, optional output, data type supports float16, bfloat16, only supports 3 dimensions. The output of AllGather, with the data type consistent with the input x. May be needed for backward pass.
- y3OutOptional: Tensor, optional output, data type supports float16, bfloat16, only supports 3 dimensions. When an activation function is present, the output of BatchMatMul, with the type consistent with the input x.

Attributes:

- group_ep: Required attribute, str. The name of the ep communication domain, which is the communication domain for expert parallelism.
- group_ep_worldsize: Required attribute, int. The size of the ep communication domain, supports 2/4/8/16/32.
- group_tp: Required attribute, str. Name of the tp communication domain, the communication domain for tensor parallelism.
- group_tp_worldsize: Required attribute, int. Size of the tp communication domain, supporting 2/4/8/16/32.
- shard_type: Optional attribute, int, default value is 0. 0 indicates performing allgather on the H dimension by tp domain, and 1 indicates performing allgather on the C dimension by tp domain.
- act_type: Optional attribute, str, activation function type, default value is None, indicating no activation function. Supports GELU/Silu/FastGELU/Relu/None, etc.
- need_allgather_out: Whether to output the result after allgather, default is False, indicating no output is required.
- need_activation_feature: Whether to output the result before the activation function (after BatchMatMul). Defaults to False, meaning no output. This is only meaningful when act_type is not None.

## Input Shape Constraints

Due to the requirements of collective communication and BatchMatMul computation, the input and output shapes must satisfy the following mathematical relationships: (where ep=group_ep_worldsize, tp=group_tp_worldsize)

For the AllGather scenario along the H axis, when shard_type is 0:

- x: (E, C, H/tp)
- weight: (E/ep, H, M/tp)
- bias: Supports two or three dimensions. When three-dimensional, the shape is: (E/ep, 1, M/tp); when two-dimensional, the shape is: (E/ep, M/tp)
- y1Out: (E/ep, ep\*C, M/tp)
- y2OutOptional: (E/ep, ep\*C, H)
- y3OutOptional: (E/ep, ep\*C, M/tp)

AllGather scenario along the C axis, when shard_type is 1:

- x: (E, C/tp, H);
- weight: (E/ep, H, M/tp);
- bias: supports two or three dimensions. When three-dimensional, the shape is: (E/ep, 1, M/tp); when two-dimensional, the shape is: (E/ep, M/tp)
- y1Out: (E/ep, ep\*tp\*C/tp, M/tp);
- y2OutOptional: (E/ep, ep\*tp\*C/tp, H);
- y3OutOptional: (E/ep, ep\*tp\*C/tp, M/tp)

Data relationship description:

- For example, x.size(0) equals E, and weight.size(0) equals E/ep, which means x.size(0) = ep\*weight.size(0), and x.size(0) is an integer multiple of ep; other relationships are similar.
- The value range of E is [2, 512], and E must be an integer multiple of ep;
- The value range of H is [1, 65535]. When shard_type is 0, H must be an integer multiple of tp.
- The value range of M/tp is [1, 65535].
- The value range of E/ep is [1, 32].
- Both ep and tp only support 2, 4, 8, 16, and 32.
- The names of group_ep and group_tp cannot be the same.
- C must be greater than 0, with the upper limit being the device memory limit of the operator. When shard_type is 1, C must be an integer multiple of tp.
- Cross-super-node is not supported; only intra-super-node is supported.

## Example Call of npu_alltoall_allgather_bmm

The terminal command is as follows:

```bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_addr 127.0.0.1  --master_port 29500 demo_test.py
```

Note: The master_addr and master_port parameters must be set by the user according to the actual situation.

The sample code of demo_test.py is as follows:

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
    # Assigning None can verify the scenario where bias is empty
    bias_npu = None

    y_npu = npu_alltoall_allgather_bmm(x_npu,
                                       weight_npu,
                                       ep_group,
                                       ep_size,
                                       tp_group,
                                       tp_size,
                                       bias=bias_npu,
                                       shard_type=x_shard_type,
                                       act_type=act_type,
                                       need_allgather_out=out_y2_flag,
                                       need_activation_feature=out_y3_flag)
    if rank == 0:
        for i, y in enumerate(y_npu[0]):
            y.cpu().numpy().tofile(f"./y_{i}.bin")

```
