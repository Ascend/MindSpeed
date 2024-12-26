# npu_apply_fused_ema_adamw 对外接口

## 接口原型
```
npu_apply_fused_ema_adamw(grad, var, m, v, s, step, lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay)-> var, m, v, s
```
npu_apply_fused_ema_adamw接口用于更新fused_ema_adamw优化器中的var(模型参数), m(一阶矩动量), v(二阶矩动量), s(ema模型参数)这四个参数。<br>

```python
# 接口内部计算逻辑示例如下
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
    demon = torch.pow(next_v, 0.5) + eps
    update = next_m / demon + weight_decay * var * mode
    var_ = var - lr * update
    s_ = ema_decay * s + (1 - ema_decay) * var_
    return var_, m_, v_, s_       
```

## 输入：
- `grad`：必选输入，数据类型为tensor(float32)，表示模型参数的梯度。接受任意shape但需保持接口调用时`grad, var, m, v, s`五个入参shape一致。
- `var`：必选输入，数据类型为tensor(float32)，表示模型参数。接受任意shape但需保持接口调用时`grad, var, m, v, s`五个入参shape一致。
- `m`：必选输入，数据类型为tensor(float32)，表示一阶矩动量。接受任意shape但需保持接口调用时`grad, var, m, v, s`五个入参shape一致。
- `v`：必选输入，数据类型为tensor(float32)，表示二阶矩动量。接受任意shape但需保持接口调用时`grad, var, m, v, s`五个入参shape一致。
- `s`：必选输入，数据类型为tensor(float32)，表示ema模型参数。接受任意shape但需保持接口调用时`grad, var, m, v, s`五个入参shape一致。
- `step`：必选输入，数据类型为tensor(int64)，shape：(1,)，表示当前为第几步。
- `lr`：可选属性，数据类型为float32，默认值：1e-3。表示学习率。
- `ema_decay`：可选属性，数据类型为float32，默认值：0.9999。表示ema衰减超参数。
- `beta1`：可选属性，数据类型为float32，默认值：0.9。表示一阶矩动量的衰减率。
- `beta2`：可选属性，数据类型为float32，默认值：0.999。表示二阶矩动量的衰减率。
- `eps`：可选属性，数据类型为float32，默认值：1e-8。表示一个极小的数。
- `mode`：可选属性，数据类型为int，默认值：1。取1表示以adamw模式计算，取0表示以adam模式计算。
- `bias_correction`：可选属性，数据类型为bool，默认值：True。表示是否开启偏置修正。
- `weight_decay`：可选属性，数据类型为float32，默认值：0.0。表示模型参数的衰减率。

支持的输入数据类型组合：

| 参数名称            | 数据类型            | 
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

## 输出：
- var：必选输出，数据类型为tensor(float32)，shape和入参var一致，表示更新后的模型参数。
- m：必选输出，数据类型为tensor(float32)，shape和入参m一致，表示更新后的一阶矩动量。
- v：必选输出，数据类型为tensor(float32)，shape和入参v一致，表示更新后的二阶矩动量。
- s：必选输出，数据类型为tensor(float32)，shape和入参s一致，表示更新后的ema模型参数。

***
## 调用示例：

- 输入 grad, var, m, v, s, step, lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay
- 输出 var, m, v, s

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