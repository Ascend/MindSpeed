# Ascend MC2

## 使用前提

**后端限制：** MC2 仅作用于原生 ColumnParallelLinear/RowParallelLinear，不覆盖 TE 路径（`--transformer-impl transformer_engine`，昇腾上由 TENPU 提供实现）下的线性层。

仅限于版本标识为`CANN 8.0.RC2`和`Ascend HDK 24.1.RC2`及其后续所有迭代版本的系统环境。

若在非指定版本中尝试采用该配置，可能触发系统级的异常行为，包括但不限于运行时错误。

## 背景与挑战

在开启了TP和SP的大模型训练场景下，存在matmul计算和all-reduce操作的强依赖关系（不开启SP），或存在matmul计算和all_gather/reduce_scatter操作的强依赖关系（开启SP）。当模型参数量较大时，此处通信量和计算量都较大，在串行执行时，会引入较长的等待闲置时间。

## 解决方案

针对存在强依赖关系的matmul计算和通信操作，昇腾自研MC2解决方案。

MC2通过融合算子的方式将matmul计算和集合通信操作进行融合，将较大的计算和通信任务切分成了较小的计算子任务和通信子任务，并通过流水的方式使得通信子任务和计算子任务可以互相掩盖，从而减少等待和闲置时间，提高利用率。

## 解决思路

MindSpeed在Python脚本侧将原本串行的matmul和all_gather/reduce_scatter操作通过MC2融合算子接口进行了融合。

具体实现参见[mc2_linears_seq_parallel](../../../mindspeed/core/tensor_parallel/ascend_turbo/mc2_linears_seq_parallel.py)。

MC2算子接口参见[mc2_opraters_api](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000449.html)。

## 使用场景

当开启了TP和SP时，建议开启MC2进一步优化。模型权重冻结和模型权重不冻结两个场景均支持。

### 说明

可以通过设置`requires_grad`属性为`False`来实现权重冻结。

```python
# 举例1:冻结所有参数
for param in model.parameters():
    param.requires_grad = False
```

```python
# 举例2:除了output_layer，冻结所有ColumnParallelLinear和RowParallelLinear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
for name, module in model.named_modules():
    if ('output_layer' not in name
            and (isinstance(module, ColumnParallelLinear) or isinstance(module, RowParallelLinear))):
        for param in module.parameters():
            param.requires_grad = False
```

> [!NOTE]
>
> **适用范围**：举例2只适用于`--transformer-impl local`路径。在`--transformer-impl transformer_engine`路径下，线性层是Megatron的TE封装类，不是`ColumnParallelLinear` / `RowParallelLinear`的子类，`isinstance`恒为`False` —— **冻结静默失效：不报错、不打印、不冻任何参数**。

两条路径下线性层的类：

| 模块 | `local` | `transformer_engine` |
| --- | --- | --- |
| `self_attention.linear_qkv` | `ColumnParallelLinear` | `TELayerNormColumnParallelLinear` |
| `mlp.linear_fc1` | `ColumnParallelLinear` | `TELayerNormColumnParallelLinear` |
| `self_attention.linear_proj` | `RowParallelLinear` | `TERowParallelLinear` |
| `mlp.linear_fc2` | `RowParallelLinear` | `TERowParallelLinear` |

- `local`：线性层为Megatron原生实现；开启`--use-ascend-mc2`后替换为`MindSpeedMC2Column/RowParallelLinear`，仍继承原生类，举例2 的`isinstance`照样命中。
- `transformer_engine`：昇腾上由TENPU（环境中的`transformer_engine`包）提供实现；模型线性层是Megatron的TE封装类，位于`megatron.core.extensions.transformer_engine`，继承TENPU的`te.pytorch.Linear` / `LayerNormLinear`。
- `TELayerNormColumnParallelLinear`融合了LayerNorm且不继承`TELinear`，需单独匹配，其`layer_norm_weight`保留可训练。

```python
# 举例3:冻结所有 column + row 线性层，两条路径通用
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

FREEZE_TYPES = (ColumnParallelLinear, RowParallelLinear,
                TEColumnParallelLinear, TERowParallelLinear, TELayerNormColumnParallelLinear)
for name, module in model.named_modules():
    if 'output_layer' in name or not isinstance(module, FREEZE_TYPES):
        continue
    for param_name, param in module.named_parameters(recurse=False):
        if 'norm' in param_name:  # 融合进线性层的 LayerNorm，保留可训练
            continue
        param.requires_grad = False
    print(f'rank:{torch.distributed.get_rank()}, frozen: {name}')
```

举例3必须从`megatron.core.extensions.transformer_engine`导入TE封装类（模型实例就是这一层的类）；`transformer_engine.pytorch.*`是TENPU的基类，模型里没有这种实例，TENPU改成不继承后会再次静默失效。

失效是静默的，建议保留打印并在CI断言条数：`rank数 × (num_layers / pipeline_model_parallel_size) × 4`（8卡/8层/PP=2时=128，只冻row=64）。

## 使用方法

原生线性层路径设置 `--transformer-impl local --use-ascend-mc2` 使能 MC2 算子。

TP 必须大于 1，并开启 `--sequence-parallel`。

## 使用效果

在开启TP和SP的训练场景下，使用MC2可以减少内存开销并提高计算效率。

## 注意事项

1. MoE模型暂不支持开启MC2。
2. FP8 场景仅支持 `mxfp8` recipe。
3. 该特性不支持在 Atlas 900 A3 硬件上使用。
