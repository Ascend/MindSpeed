# npu_apply_fused_ema_adamw External API

## Prototype

```python
npu_apply_fused_ema_adamw(grad, var, m, v, s, step, lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay)-> var, m, v, s
```

The npu_apply_fused_ema_adamw API is used to update four parameters in the fused_ema_adamw optimizer: var (Model Parameter), m (First-order Momentum), v (Second-order Momentum), and s (EMA Model Parameter).<br>

```python
# Example of the internal calculation logic of the API
def npu_apply_fused_ema_adamw(grad, var, m, v, s, step, lr, ema_decay,
                              beta1, beta2, eps, mode, bias_correction,
                              weight_decay):
    beta1_correction = 1 - torch.pow(beta1, step) * bias_correction
    beta2_correction = 1 - torch.pow(beta2, step) * bias_correction
    grad_ = grad + weight_decay * var * (1 - mode)
    m_ = beta1 * m + (1 - beta1) * grad_
    v_ = beta2 * v + (1 - beta2) * grad_ * grad_
    next_m = m_ / beta1_correction
    next_v = v_ / beta2_correction
    denom = torch.pow(next_v, 0.5) + eps
    update = next_m / denom + weight_decay * var * mode
    var_ = var - lr * update
    s_ = ema_decay * s + (1 - ema_decay) * var_
    return var_, m_, v_, s_
```

## Input

- `grad`: Required input, data type is tensor(float32), represents the gradient of the model parameter. Accepts any shape but must keep the shape consistent across the five input parameters `grad, var, m, v, s` during API call.
- `var`: Required input, data type is tensor(float32), represents the model parameter. Accepts any shape but must keep the shape consistent across the five input parameters `grad, var, m, v, s` during API call.
- `m`: Required input, data type is tensor(float32), represents the first-order momentum. Accepts any shape but must keep the shape consistent across the five input parameters `grad, var, m, v, s` during API call.
- `v`: Required input, data type is tensor(float32), represents the second-order momentum. Accepts any shape but must keep the shape consistent across the five input parameters `grad, var, m, v, s` during API call.
- `s`: Required input, data type is tensor(float32), represents the EMA model parameter. Accepts any shape but must keep the shape consistent across the five input parameters `grad, var, m, v, s` during API call.
- `step`: Required input, data type is tensor(int64), shape: (1,), represents the current step number.
- `lr`: Optional attribute, data type is float32, default value: 1e-3. Represents the learning rate.
- `ema_decay`: Optional attribute, data type is float32, default value: 0.9999. Represents the EMA decay hyperparameter.
- `beta1`: Optional attribute, data type is float32, default value: 0.9. Represents the decay rate of the first-order momentum.
- `beta2`: Optional attribute, data type is float32, default value: 0.999. Represents the decay rate of the second-order momentum.
- `eps`: Optional attribute, data type is float32, default value: 1e-8. Represents a very small number.
- `mode`: Optional attribute, data type is int, default value: 1. A value of 1 indicates calculation in adamw mode, and a value of 0 indicates calculation in adam mode.
- `bias_correction`: Optional attribute, data type is bool, default value: True. Whether to enable bias correction.
- `weight_decay`: Optional attribute, data type is float32, default value: 0.0. Represents the decay rate of the model parameter.

Supported input data type combinations:

| Name            | Type            |
|-----------------|-----------------|
| grad            | tensor(float32) |
| var             | tensor(float32) |
| m               | tensor(float32) |
| v               | tensor(float32) |
| s               | tensor(float32) |
| step            | tensor(int64)   |
| lr              | float32         |
| ema_decay       | float32         |
| beta1           | float32         |
| beta2           | float32         |
| eps             | float32         |
| mode            | int             |
| bias_correction | bool            |
| weight_decay    | float32         |

## Output

- var: Required output, data type is tensor(float32), shape is consistent with the input parameter var, represents the updated model parameter.
- m: Required output, data type is tensor(float32), shape is consistent with the input parameter m, represents the updated first-order momentum.
- v: Required output, data type is tensor(float32), shape is consistent with the input parameter v, represents the updated second-order momentum.
- s: Required output, data type is tensor(float32), shape is consistent with input parameter s, represents the updated EMA model parameter.

***

## Example Call

- Input grad, var, m, v, s, step, lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay
- Output var, m, v, s

```python
import torch
import torch_npu
from mindspeed.ops.npu_apply_fused_ema_adamw import npu_apply_fused_ema_adamw


grad = torch.full((10, 10), 0.5).to(torch.float32).npu()
var = torch.full((10, 10), 0.5).to(torch.float32).npu()
m = torch.full((10, 10), 0.9).to(torch.float32).npu()
v = torch.full((10, 10), 0.9).to(torch.float32).npu()
s = torch.full((10, 10), 0.5).to(torch.float32).npu()
step = torch.full((1, ), 1).to(torch.int64).npu()
lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay= 1e-8, 0.9999, 0.9999, 0.9999, 1e-8, 1, True, 0.001
var, m, v, s = npu_apply_fused_ema_adamw(grad, var, m, v, s, step, lr, ema_decay, beta1, beta2, eps, mode,
                                         bias_correction, weight_decay)

```
