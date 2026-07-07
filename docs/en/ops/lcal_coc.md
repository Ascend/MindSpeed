# LCAL_COC External API

## matmul_all_reduce

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.matmul_all_reduce(input1, input2, output, bias)
```

### Function

This interface performs a Matmul operation on the input left and right matrices, performs All-Reduce communication on the result, and finally adds a bias (if the bias is not None). The final result is assigned to the output memory region.

### Input/Output

Assume the shapes corresponding to the Matmul operation are [m, k] and [k, n]:

Input:

- input1: Left Matrix (required input, data type float16/bfloat16, shape supports only two dimensions, no transpose, [m, k]);
- input2: Right Matrix (required input, data type float16/bfloat16, shape supports only two dimensions, transpose supported, [k, n]/[n, k]);
- output: Output Matrix, requires pre-allocated memory as an interface input (required input, data type float16/bfloat16, shape supports only two dimensions, [m, n]);
- bias: bias vector (optional input, data type float16/bfloat16, shape supports [1, n]);

Output:

- None

### Use Example

```python
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu
import megatron.core.parallel_state as ps


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def test_coc_matmul_all_reduce(rank, world_size, master_ip, master_port):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)
    initialize_model_parallel(world_size)
    from mindspeed.ops.lcal_functional import coc_ops

    m, k, n = 2048, 4096, 8192
    dtype = torch.float16
    input1 = torch.rand(m, k, dtype=dtype, device=torch.npu.current_device())
    input2 = torch.rand(k, n, dtype=dtype, device=torch.npu.current_device())
    bias = torch.rand(1, n, dtype=dtype, device=torch.npu.current_device())
    output = torch.zeros(m, n, dtype=dtype, device=torch.npu.current_device())
    coc_ops.matmul_all_reduce(input1, input2, output, bias)
    torch.npu.synchronize()
    print(output)

if __name__ == "__main__":
    world_size = 8
    master_ip = "127.0.0.1"
    master_port = "50001"
    mp.spawn(test_coc_matmul_all_reduce, args=(world_size, master_ip, master_port), nprocs=world_size)
```

## all_gather_matmul

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.all_gather_matmul(input1, input2, output, bias)
```

### Function

This interface performs an All-Gather operation on the input left matrix, then performs a Matmul operation with the right matrix, and finally adds the bias (if bias is not None). The final result is assigned to the output memory region.

### Input/Output

Assume the shapes corresponding to the Matmul operation are [m, k] and [k, n] (m must be a multiple of world_size):

Input:

- input1: Left Matrix (required input, data type float16/bfloat16, shape supports only 2D, no transpose, \[m // world_size, k\]);
- input2: Right Matrix (required input, data type float16/bfloat16, shape supports only 2D, transpose supported, \[k, n\]/\[n, k\]);
- output: Output Matrix, requires memory allocation in advance as an interface input (required input, data type float16/bfloat16, shape supports only 2D, \[m, n\]);
- bias: Bias Vector (optional input, data type float16/bfloat16, shape supports \[1, n\]);

Output:

- None

### Use Example

```python
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu
import megatron.core.parallel_state as ps


def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def test_coc_all_gather_matmul(rank, world_size, master_ip, master_port):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)
    initialize_model_parallel(world_size)
    from mindspeed.ops.lcal_functional import coc_ops

    m, k, n = 2048, 4096, 8192
    dtype = torch.float16
    input1 = torch.rand(m // world_size, k, dtype=dtype, device=torch.npu.current_device())
    input2 = torch.rand(k, n, dtype=dtype, device=torch.npu.current_device())
    bias = torch.rand(1, n, dtype=dtype, device=torch.npu.current_device())
    output = torch.zeros(m, n, dtype=dtype, device=torch.npu.current_device())
    coc_ops.all_gather_matmul(input1, input2, output, bias)
    torch.npu.synchronize()
    print(output)


if __name__ == "__main__":
    world_size = 8
    master_ip = "127.0.0.1"
    master_port = "50001"
    mp.spawn(test_coc_all_gather_matmul, args=(world_size, master_ip, master_port), nprocs=world_size)
```

## ALL_GATHER_MATMUL_V2

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.all_gather_matmul_v2(input1, input2, output, comm_output, bias)
```

### Function

This interface performs an All-Gather operation on the input left matrix, then performs a Matmul operation with the right matrix, and finally adds the bias (if bias is not None). The final result is assigned to the output memory region, and the result obtained after the All-Gather operation on the left matrix is assigned to the comm_output memory region.

### Input/Output

Assume the shapes corresponding to the Matmul operation are [m, k] and [k, n] (m must be a multiple of world_size):

Input:

- input1: Left Matrix (Required Input, data type float16/bfloat16, shape supports only 2D, no transpose, [m // world_size, k]);
- input2: Right Matrix (Required Input, data type float16/bfloat16, shape supports only 2D, transpose supported, [k, n]/[n, k]);
- output: Output matrix, requires memory allocation in advance as interface input (required input, data type float16/bfloat16, shape supports only two dimensions, \[m,n\]);
- comm_output: Output matrix, requires memory allocation in advance as interface input (required input, data type float16/bfloat16, shape supports only two dimensions, \[m,k\]);
- bias: Bias vector (optional input, data type float16/bfloat16, shape supports \[1, n\]);

Output:

- None

### Use Example

```python
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu
import megatron.core.parallel_state as ps


def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def test_coc_all_gather_matmul_v2(rank, world_size, master_ip, master_port):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)
    initialize_model_parallel(world_size)
    from mindspeed.ops.lcal_functional import coc_ops

    m, k, n = 2048, 4096, 8192
    dtype = torch.float16
    input1 = torch.rand(m // world_size, k, dtype=dtype, device=torch.npu.current_device())
    input2 = torch.rand(k, n, dtype=dtype, device=torch.npu.current_device())
    bias = torch.rand(1, n, dtype=dtype, device=torch.npu.current_device())
    output = torch.zeros(m, n, dtype=dtype, device=torch.npu.current_device())
    comm_output = torch.zeros(m, k, dtype=dtype, device=torch.npu.current_device())
    coc_ops.all_gather_matmul_v2(input1, input2, output, comm_output, bias)
    torch.npu.synchronize()
    print(output)


if __name__ == "__main__":
    world_size = 8
    master_ip = "127.0.0.1"
    master_port = "50001"
    mp.spawn(test_coc_all_gather_matmul_v2, args=(world_size, master_ip, master_port), nprocs=world_size)
```

## MATMUL_REDUCE_SCATTER

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.matmul_reduce_scatter(input1, input2, output, bias)
```

### Function

This interface performs a Matmul operation on the input left and right matrices, performs Reduce-Scatter communication on the result, and finally adds a bias (if bias is not None). The final result is assigned to the output memory region.

### Input/Output

Assume the shapes corresponding to the Matmul operation are [m, k] and [k, n] (m must be a multiple of world_size):

Input:

- input1: Left Matrix (required input, data type float16/bfloat16, shape supports only two dimensions, no transpose, [m, k]);
- input2: Right Matrix (required input, data type float16/bfloat16, shape supports only two dimensions, transpose supported, [k, n]/[n, k]);
- output: Output Matrix, requires memory allocation in advance as an interface input (required input, data type float16/bfloat16, shape supports only two dimensions, [m // world_size, n]);
- bias: bias vector (optional input, data type float16/bfloat16, shape supports \[1, n\]);

Output:

- None

### Use Example

```python
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu
import megatron.core.parallel_state as ps


def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def test_coc_matmul_reduce_scatter(rank, world_size, master_ip, master_port):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)
    initialize_model_parallel(world_size)
    from mindspeed.ops.lcal_functional import coc_ops

    m, k, n = 2048, 4096, 8192
    dtype = torch.float16
    input1 = torch.rand(m, k, dtype=dtype, device=torch.npu.current_device())
    input2 = torch.rand(k, n, dtype=dtype, device=torch.npu.current_device())
    bias = torch.rand(1, n, dtype=dtype, device=torch.npu.current_device())
    output = torch.zeros(m // world_size, n, dtype=dtype, device=torch.npu.current_device())
    coc_ops.matmul_reduce_scatter(input1, input2, output, bias)
    torch.npu.synchronize()
    print(output)


if __name__ == "__main__":
    world_size = 8
    master_ip = "127.0.0.1"
    master_port = "50001"
    mp.spawn(test_coc_matmul_reduce_scatter, args=(world_size, master_ip, master_port), nprocs=world_size)
```

## PURE_MATMUL

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.pure_matmul(input1, input2, output, bias)
```

### Function

This interface performs an Lcal Matmul operation on the input left and right matrices, and finally adds the bias (if bias is not None). The final result is assigned to the output memory region.

### Input/Output

Assume the shapes corresponding to the Matmul operation are [m, k] and [k, n]:

Input:

- input1: Left Matrix (required input, data type float16/bfloat16, shape supports only two dimensions, no transpose, \[m,k\]);
- input2: Right Matrix (required input, data type float16/bfloat16, shape supports only two dimensions, transpose supported, \[k,n\]/\[n,k\]);
- output: Output Matrix, requires memory allocation in advance as an interface input (required input, data type float16/bfloat16, shape supports only two dimensions, \[m,n\]);
- bias: Bias Vector (optional input, data type float16/bfloat16, shape supports \[1, n\]);

Interface Output:

- None

### Example Call

```python
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu
import megatron.core.parallel_state as ps


def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )


def test_coc_pure_matmul(rank, world_size, master_ip, master_port):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)
    initialize_model_parallel(world_size)
    from mindspeed.ops.lcal_functional import coc_ops

    m, k, n = 2048, 4096, 8192
    dtype = torch.float16
    input1 = torch.rand(m, k, dtype=dtype, device=torch.npu.current_device())
    input2 = torch.rand(k, n, dtype=dtype, device=torch.npu.current_device())
    bias = torch.rand(1, n, dtype=dtype, device=torch.npu.current_device())
    output = torch.zeros(m, n, dtype=dtype, device=torch.npu.current_device())
    coc_ops.pure_matmul(input1, input2, output, bias)
    torch.npu.synchronize()
    print(output)


if __name__ == "__main__":
    world_size = 8
    master_ip = "127.0.0.1"
    master_port = "50001"
    mp.spawn(test_coc_pure_matmul, args=(world_size, master_ip, master_port), nprocs=world_size)
```
