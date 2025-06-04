# LCAL_COC对外接口

## matmul_all_reduce接口

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.matmul_all_reduce(input1, input2, output, bias)
```

### 接口功能

该接口对输入的左右矩阵进行Matmul操作，并对其结果进行All-Reduce通信，最后加上bias（如果bias不为None）。将最终结果赋值到output内存区域中。

### 接口输入输出

假设Matmul操作对应的shape为[m, k]和[k, n]：

接口输入：
- input1：左矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，不支持转置，\[m,k\]）；
- input2：右矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，支持转置，\[k,n\]/\[n,k\]）；
- output：输出矩阵，需要提前申请内存作为接口的输入（必选输入，数据类型float16/bfloat16，shape只支持二维，\[m,n\]）；
- bias：偏置向量（可选输入，数据类型float16/bfloat16，shape支持\[1, n\]）；

接口输出：
- 无

### 使用案例

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


## all_gather_matmul接口

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.all_gather_matmul(input1, input2, output, bias)
````

### 接口功能

该接口对输入的左矩阵进行All-Gather操作，然后将其与右矩阵做Matmul操作，最后加上bias（如果bias不为None）。将最终结果赋值到output内存区域中。

### 接口输入输出

假设Matmul操作对应的shape为[m, k]和[k, n]（m必须为world_size的倍数）：

接口输入：
- input1：左矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，不支持转置，\[m // world_size,k\]）；
- input2：右矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，支持转置，\[k,n\]/\[n,k\]）；
- output：输出矩阵，需要提前申请内存作为接口的输入（必选输入，数据类型float16/bfloat16，shape只支持二维，\[m,n\]）；
- bias：偏置向量（可选输入，数据类型float16/bfloat16，shape支持\[1, n\]）；

接口输出：
- 无

### 使用案例

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


## ALL_GATHER_MATMUL_V2接口

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.all_gather_matmul_v2(input1, input2, output, comm_output, bias)
```

### 接口功能

该接口对输入的左矩阵进行All-Gather操作，然后将其与右矩阵做Matmul操作，最后加上bias（如果bias不为None）。将最终结果赋值到output内存区域中，并将左矩阵进行All-Gather操作后得到的结果赋值到comm_output内存区域中。

### 接口输入输出

假设Matmul操作对应的shape为[m, k]和[k, n]（m必须为world_size的倍数）：

接口输入：
- input1：左矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，不支持转置，\[m // world_size,k\]）；
- input2：右矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，支持转置，\[k,n\]/\[n,k\]）；
- output：输出矩阵，需要提前申请内存作为接口的输入（必选输入，数据类型float16/bfloat16，shape只支持二维，\[m,n\]）；
- comm_output：输出矩阵，需要提前申请内存作为接口的输入（必选输入，数据类型float16/bfloat16，shape只支持二维，\[m,k\]）；
- bias：偏置向量（可选输入，数据类型float16/bfloat16，shape支持\[1, n\]）；

接口输出：
- 无

### 使用案例

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

## MATMUL_REDUCE_SCATTER接口

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.matmul_reduce_scatter(input1, input2, output, bias)
````

### 接口功能

该接口对输入的左右矩阵进行Matmul操作，并对其结果进行Reduce-Scatter通信，最后加上bias（如果bias不为None）。将最终结果赋值到output内存区域中。

### 接口输入输出

假设Matmul操作对应的shape为[m, k]和[k, n]（m必须为world_size的倍数）：

接口输入：
- input1：左矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，不支持转置，\[m,k\]）；
- input2：右矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，支持转置，\[k,n\]/\[n,k\]）；
- output：输出矩阵，需要提前申请内存作为接口的输入（必选输入，数据类型float16/bfloat16，shape只支持二维，\[m // world_size,n\]）；
- bias：偏置向量（可选输入，数据类型float16/bfloat16，shape支持\[1, n\]）；

接口输出：
- 无

### 使用方法

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


## PURE_MATMUL接口

```python
from mindspeed.ops.lcal_functional import coc_ops

coc_ops.pure_matmul(input1, input2, output, bias)
````

### 接口功能

该接口对输入的左右矩阵进行Lcal Matmul操作，最后加上bias（如果bias不为None）。将最终结果赋值到output内存区域中。

### 接口输入输出

假设Matmul操作对应的shape为[m, k]和[k, n]：

接口输入：
- input1：左矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，不支持转置，\[m,k\]）；
- input2：右矩阵（必选输入，数据类型float16/bfloat16，shape只支持二维，支持转置，\[k,n\]/\[n,k\]）；
- output：输出矩阵，需要提前申请内存作为接口的输入（必选输入，数据类型float16/bfloat16，shape只支持二维，\[m,n\]）；
- bias：偏置向量（可选输入，数据类型float16/bfloat16，shape支持\[1, n\]）；

接口输出：
- 无

### 使用方法

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
