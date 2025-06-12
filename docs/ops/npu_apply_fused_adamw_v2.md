# npu_apply_fused_adamw_v2 对外接口

## 接口原型
```
npu_apply_fused_adamw_v2(var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
```
npu_apply_fused_adamw_v2接口用于更新adamw优化器中的var(模型参数), m(一阶矩动量), v(二阶矩动量),max_grad_norm(训练过程中最大的二阶矩动量)这四个参数。<br>

```python
import math
import torch
import torch_npu
import numpy as np
# 接口内部计算逻辑示例如下  

def npu_apply_fused_adamw_v2(var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize):
    var_dtype, m_dtype, v_dtype, grad_dtype, step_dtype, max_grad_norm_dtype = \
        var.dtype, m.dtype, v.dtype, grad.dtype, step.dtype, max_grad_norm.dtype
    is_var_dtype_bf16_fp16 = "bfloat16" in str(var_dtype) or "float16" in str(var_dtype)
    is_grad_dtype_bf16_fp16 = "bfloat16" in str(grad_dtype) or "float16" in str(grad_dtype)
    if is_var_dtype_bf16_fp16:
        adamw_params = [
            var.to(torch.float32), grad.to(torch.float32), m.to(torch.float32), v.to(torch.float32),
            max_grad_norm.to(torch.float32), step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
        ]
    elif is_grad_dtype_bf16_fp16:
        adamw_params = [
            var, grad.to(torch.float32), m, v, max_grad_norm.to(torch.float32), step, lr, beta1, beta2,
            weight_decay, eps, amsgrad, maximize
        ]
    else:
        adamw_params = [
            var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
        ]
    if "int64" in str(step_dtype):
        step_fp32 = step.to(torch.float32)
        adamw_params[5] = step_fp32
    def single_tensor_adamw(*args):
        (param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, step_t,
         lr, beta1, beta2, weight_decay, eps, amsgrad, maximize) = args
        dtype1 = param.dtype
        dtype2 = grad.dtype
    
        lr = np.float32(lr)
        beta1 = np.float32(beta1)
        beta2 = np.float32(beta2)
        weight_decay = np.float32(weight_decay)
        eps = np.float32(eps)
    
        if dtype1 != dtype2:
            grad = grad.to(dtype1)
            max_exp_avg_sq = max_exp_avg_sq.to(dtype1)
        if maximize:
            grad = -grad
    
        step = step_t
        step = step.item()
    
        param = param * (1 - lr * weight_decay)
    
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
    
        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        if amsgrad:
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
    
        param.addcdiv_(exp_avg, denom, value=-step_size)
    
        if dtype1 != dtype2:
            max_exp_avg_sq = max_exp_avg_sq.to(dtype2)
        return param, exp_avg, exp_avg_sq, max_exp_avg_sq
    
    res_var, res_m, res_v, res_max_grad_norm = single_tensor_adamw(*adamw_params)

    if is_var_dtype_bf16_fp16:
        res_var, res_m, res_v, res_max_grad_norm = (
            res_var.to(var_dtype), res_m.to(var_dtype),
            res_v.to(var_dtype), res_max_grad_norm.to(max_grad_norm_dtype)
        )
    elif is_grad_dtype_bf16_fp16:
        res_max_grad_norm = res_max_grad_norm.to(max_grad_norm_dtype)
    var.copy_(res_var)
    m.copy_(res_m)
    v.copy_(res_v)
    max_grad_norm.copy_(res_max_grad_norm)
```

## 输入：
- `var`：必选输入，数据类型为tensor(float32)或tensor(float16)或tensor(float16)，表示模型参数。接受任意shape，但需保持`var, grad, m, v, max_grad_norm`shape相同。
- `grad`：必选输入，数据类型为tensor(float32)或tensor(float16)或tensor(float16)，表示模型参数的梯度。接受任意shape，但需保持`var, grad, m, v, max_grad_norm`shape相同。
- `m`：必选输入，数据类型必须与var完全一致，表示一阶矩动量。接受任意shape，但需保持`var, grad, m, v, max_grad_norm`shape相同。
- `v`：必选输入，数据类型必须与var完全一致，表示二阶矩动量。接受任意shape，但需保持`var, grad, m, v, max_grad_norm`shape相同。
- `max_grad_norm`：该参数在amsgrad为True时为必选输入，在amsgrad为False时为可选输入，数据类型为tensor(float32)或tensor(float16)或tensor(float16)，表示训练过程中最大的二阶矩动量。接受任意shape，但需保持`var, grad, m, v, max_grad_norm`shape相同。
- `step`：必选输入，数据类型为tensor(int64)，shape：(1,)，表示当前为第几步。
- `lr`：可选属性，数据类型为float32，默认值：1e-3。表示学习率。
- `beta1`：可选属性，数据类型为float32，默认值：0.9。表示一阶矩动量的衰减率。
- `beta2`：可选属性，数据类型为float32，默认值：0.999。表示二阶矩动量的衰减率。
- `weight_decay`：可选属性，数据类型为float32，默认值：0.0。表示模型参数的衰减率。
- `eps`：可选属性，数据类型为float32，默认值：1e-8。表示一个极小的数。
- `amsgrad`：可选属性，数据类型为bool，默认值：False。表示是否使用训练过程中最大的二阶矩动量。
- `maximize`：可选属性，数据类型为bool，默认值：False。表示是否最大化参数。


支持的输入数据类型组合：

| 参数名称          | 组合1             | 组合2             | 组合3              | 组合4             | 组合5             | 组合6              | 组合7              | 组合8              | 组合9              | 组合10            | 组合11            | 组合12             | 组合13            | 组合14            | 组合15             | 组合16             | 组合17             | 组合18             | 组合19             | 组合20             | 组合21             | 组合22             | 组合23             | 组合24             | 组合25             | 组合26             | 组合27             |
|---------------|-----------------|-----------------|------------------|-----------------|-----------------|------------------|------------------|------------------|------------------|-----------------|-----------------|------------------|-----------------|-----------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| var           | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| grad          | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| m             | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| v             | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float32) | tensor(float32) | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float32)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(float16) | tensor(float16) | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(float16)  | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| max_grad_norm | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32)  | tensor(float16)  | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32)  | tensor(float16)  | tensor(bfloat16) | tensor(float32)  | tensor(float16)  | tensor(bfloat16) | tensor(float32)  | tensor(float16)  | tensor(bfloat16) | tensor(float32)  | tensor(float16)  | tensor(bfloat16) |
| step          | tensor(int64)   | tensor(int64)   | tensor(int64)    | tensor(int64)   | tensor(int64)   | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)   | tensor(int64)   | tensor(int64)    | tensor(int64)   | tensor(int64)   | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | tensor(int64)    | 
| lr            | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          |
| beta1         | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          |
| beta2         | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          |
| weight_decay  | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          |
| eps           | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32         | float32         | float32          | float32         | float32         | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          | float32          |
| amsgrad       | bool            | bool            | bool             | bool            | bool            | bool             | bool             | bool             | bool             | bool            | bool            | bool             | bool            | bool            | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             |
| maximize      | bool            | bool            | bool             | bool            | bool            | bool             | bool             | bool             | bool             | bool            | bool            | bool             | bool            | bool            | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             | bool             |


## 输出：
该接口无输出，该接口调用后会inplace更新入参的 var, m, v, max_grad_norm

***
## 调用示例：

- 输入 var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
- 调用 npu_apply_fused_adamw_v2 实现 var, m, v 的 inplace 更新 

```python
import torch
import torch_npu
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2

var = torch.full((10, 10), 0.5).to(torch.float32).npu()
grad = torch.full((10, 10), 0.5).to(torch.float32).npu()
m = torch.full((10, 10), 0.9).to(torch.float32).npu()
v = torch.full((10, 10), 0.9).to(torch.float32).npu()
max_grad_norm = torch.full((10, 10), 0.9).to(torch.float32).npu()
step = torch.full((1, ), 1).to(torch.int64).npu()
lr, beta1, beta2, weight_decay, eps, amsgrad, maximize = 1e-3, 0.9999, 0.9999, 0.0, 1e-8, False, False
npu_apply_fused_adamw_v2(var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)

```