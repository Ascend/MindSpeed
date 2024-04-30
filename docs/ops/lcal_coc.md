# LCAL_COC对外接口

## MATMUL_ALL_REDUCE接口
```python
from ascendspeed.ops.lcal_functional import coc_ops
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
from ascendspeed.ops.lcal_functional import coc_ops

m, k, n = 2048, 4096, 8192
dtype = torch.float16
input1 = torch.rand(m, k, dtype, device=torch.npu.current_device())
input2 = torch.rand(k, n, dtype, device=torch.npu.current_device())
bias = torch.rand(1, n, dtype, device=torch.npu.current_device())
output = torch.zeros(m, n, dtype)
coc_ops.matmul_all_reduce(input1, input2, output, bias)
```


## ALL_GATHER_MATMUL接口
```python
from ascendspeed.ops.lcal_functional import coc_ops
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
from ascendspeed.ops.lcal_functional import coc_ops

m, k, n = 2048, 4096, 8192
world_size = 8
dtype = torch.float16
input1 = torch.rand(m // world_size,  k, dtype, device=torch.npu.current_device())
input2 = torch.rand(k, n, dtype, device=torch.npu.current_device())
bias = torch.rand(1, n, dtype, device=torch.npu.current_device())
output = torch.zeros(m, n, dtype, device=torch.npu.current_device())
coc_ops.all_gather_matmul(input1, input2, output, bias)
```


## ALL_GATHER_MATMUL_V2接口
```python
from ascendspeed.ops.lcal_functional import coc_ops
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
from ascendspeed.ops.lcal_functional import coc_ops

m, k, n = 2048, 4096, 8192
world_size = 8
dtype = torch.float16
input1 = torch.rand(m // world_size, k, dtype, device=torch.npu.current_device())
input2 = torch.rand(k, n, dtype, device=torch.npu.current_device())
bias = torch.rand(1, n, dtype, device=torch.npu.current_device())
output = torch.zeros(m, n, dtype, device=torch.npu.current_device())
comm_output= torch.zeros(m, k, dtype, device=torch.npu.current_device())
coc_ops.all_gather_matmul_v2(input1, input2, output, comm_output, bias)
```

## MATMUL_REDUCE_SCATTER接口
```python
from ascendspeed.ops.lcal_functional import coc_ops
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
from ascendspeed.ops.lcal_functional import coc_ops

m, k, n = 2048, 4096, 8192
world_size = 8
dtype = torch.float16
input1 = torch.rand(m, k, dtype, device=torch.npu.current_device())
input2 = torch.rand(k, n, dtype, device=torch.npu.current_device())
bias = torch.rand(1, n, dtype, device=torch.npu.current_device())
output = torch.zeros(m // word_size, n, dtype, device=torch.npu.current_device())
coc_ops.matmul_reduce_scatter(input1, input2, output, bias)
```


## PURE_MATMUL接口
```python
from ascendspeed.ops.lcal_functional import coc_ops
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
from ascendspeed.ops.lcal_functional import coc_ops

m, k, n = 2048, 4096, 8192
world_size = 8
dtype = torch.float16
input1 = torch.rand(m, k, dtype, device=torch.npu.current_device())
input2 = torch.rand(k, n, dtype, device=torch.npu.current_device())
bias = torch.rand(1, n, dtype, device=torch.npu.current_device())
output = torch.zeros(m, n, dtype, device=torch.npu.current_device())
coc_ops.pure_matmul(input1, input2, output, bias)
```
