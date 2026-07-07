# npu_apply_fused_adamw_v2 External API

## Prototype

```python
npu_apply_fused_adamw_v2(var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
```

The npu_apply_fused_adamw_v2 interface is used to update four parameters in the AdamW optimizer: var (model parameter), m (first moment estimate), v (second moment estimate), and max_grad_norm (the maximum second moment estimate during training).

```python
import math
import torch
import torch_npu
import numpy as np
# An example of the internal calculation logic of the interface is as follows

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

## Input

- `var`: Required input, data type is tensor(float32) or tensor(float16) or tensor(bfloat16), representing the model parameters. Accepts any shape, but the shapes of `var, grad, m, v, max_grad_norm` must be the same.
- `grad`: Required input, data type is tensor(float32) or tensor(float16) or tensor(bfloat16), representing the gradient of the model parameters. Accepts any shape, but the shapes of `var, grad, m, v, max_grad_norm` must be the same.
- `m`: Required input, data type must be exactly the same as var, representing the first moment estimate. Accepts any shape, but the shapes of `var, grad, m, v, max_grad_norm` must be the same.
- `v`: Required input, data type must be exactly the same as var, representing the second moment estimate. Accepts any shape, but the shapes of `var, grad, m, v, max_grad_norm` must be the same.
- `max_grad_norm`: This parameter is a required input when amsgrad is True, and an optional input when amsgrad is False. The data type is tensor(float32) or tensor(float16) or tensor(bfloat16), representing the maximum second moment estimate during training. Accepts any shape, but the shapes of `var, grad, m, v, max_grad_norm` must be the same.
- `step`: Required input, data type is tensor(int64), shape: (1,), indicating the current step number.
- `lr`: Optional attribute, data type is float32, default value: 1e-3. Represents the learning rate.
- `beta1`: Optional attribute, data type is float32, default value: 0.9. Represents the decay rate of the first moment estimate.
- `beta2`: Optional attribute, data type is float32, default value: 0.999. Represents the decay rate of the second moment estimate.
- `weight_decay`: Optional attribute, data type is float32, default value: 0.0. Represents the decay rate of the model parameters.
- `eps`: Optional attribute, data type float32, default value: 1e-8. Represents a very small number.
- `amsgrad`: Optional attribute, data type bool, default value: False. Whether to use the largest second moment estimate during training.
- `maximize`: Optional attribute, data type bool, default value: False. Whether to maximize the parameter.

Supported input data type combinations:

| Parameter | Combination 1 | Combination 2 | Combination 3 | Combination 4 | Combination 5 | Combination 6 | Combination 7 | Combination 8 | Combination 9 | Combination 10 | Combination 11 | Combination 12 | Combination 13 | Combination 14 | Combination 15 | Combination 16 | Combination 17 | Combination 18 | Combination 19 | Combination 20 | Combination 21 | Combination 22 | Combination 23 | Combination 24 | Combination 25 | Combination 26 | Combination 27 |
|---------------|-----------------|-----------------|------------------|-----------------|-----------------|------------------|------------------|------------------|------------------|-----------------|-----------------|------------------|-----------------|-----------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| var | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| grad | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| m | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| v | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float32) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(float16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) | tensor(bfloat16) |
| max_grad_norm | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) | tensor(float32) | tensor(float16) | tensor(bfloat16) |
| step | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) | tensor(int64) |
| lr | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| beta1 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| beta2 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| weight_decay | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| eps | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| amsgrad | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| maximize | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool | bool |

## Output

This interface has no output. After the interface is called, the input parameters var, m, v, and max_grad_norm are updated in-place.

***

## Example Call

- Input var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
- Call npu_apply_fused_adamw_v2 to perform in-place updates of var, m, and v

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
