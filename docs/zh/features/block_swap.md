# block_swap

## 背景与挑战

大模型训练/推理中，部分张量（专家权重、常量表等）的显存只在少数几行代码里被用到，却全程常驻 HBM。对不经过 Megatron 线性层封装的自定义算子（裸 `matmul`/`bmm`、MoE 分组矩阵乘、自定义投影等），缺少按需换入换出显存的机制。

## 解决方案

`block_swap` 是一个代码级上下文管理器，把任意 NPU 张量包进 `with` 块：进入时从 CPU pinned 镜像物化到 NPU（H2D），退出时释放显存（`storage().resize_(0)`），块内张量与普通张量用法一致。与 swap-optimizer 同源机制，以每次使用一次 H2D 的 PCIe 流量换取该张量 100% 的常驻显存节省。

1. 首次使用时自动为张量建立 pinned 镜像并收编进管理器（惰性收编，之后复用）；
2. 块内注册 saved-tensor hooks：块内 forward 保存的张量在块外 backward 解包时自动重新物化，因此 **forward 在块内、backward 在块外**是训练安全的；
3. Megatron 训练循环自动集成：首次收编张量时自动接管 Megatron 优化器的 `step_with_ready_grads` / `reload_model_params`——step 时物化全部受管张量、更新后刷新镜像（D2H）并释放，用户无需任何额外代码；
4. 支持嵌套块（深度计数，最外层退出才释放）；块内 in-place 修改由版本计数自动检测并刷新镜像。

## 使用场景

* Megatron + MindSpeed 训练框架中自定义算子的大张量：裸 `matmul`/`bmm`、MoE 分组矩阵乘（GroupedMLP 的 `weight1`/`weight2`）、自定义投影等；
* 推理/常量表：使用频率低、显存占用大的张量。

## 使用方法

推理 / 自定义前向：

```python
from mindspeed.core.memory.block_swap import block_swap

# MoE 专家分组矩阵乘，weight1/weight2 形如 [num_local_experts, in, out]
with block_swap(self.weight1, self.weight2):
    hidden = torch.bmm(x, self.weight1.transpose(-1, -2))
    out = torch.bmm(hidden, self.weight2)
# 退出块：两份权重显存立即释放
```

Megatron 训练循环（框架自动处理 backward 与优化器更新，只需包住自定义算子）：

```python
from mindspeed.core.memory.block_swap import block_swap

# 例如 GroupedMLP forward 内：
with block_swap(self.weight1, self.weight2):
    hidden = torch.bmm(x, self.weight1.transpose(-1, -2))
    out = torch.bmm(hidden, self.weight2)
# 块外：backward 自动重物化、优化器 step 自动物化/更新/刷新镜像/释放
```

参数：`*tensors` 为任意个 NPU 张量（必填）；`copy_back` 默认 `False`（只读块退出零拷贝），块内通过 `.data` 修改值时传 `True`。

批量手动控制：`block_swap_all_in()` / `block_swap_all_out(copy_to_host=True)`。

> [!NOTE]
>
> - 张量必须在 NPU 上（CPU 张量报错；模型搬运到 NPU 后再包块）。
> - 视图张量（更大存储的切片）会告警：释放会殃及共享同一存储的兄弟张量。
> - **优化器 step 不得在块内执行**，检测到即报 RuntimeError。
> - `.data.copy_` 等不递增版本计数的写法无法自动检测，需显式 `copy_back=True`。
> - double backward（`create_graph=True`）不支持；块退出后不得再使用该张量。
> - 优化器 step 已由 Megatron 集成自动接管（物化→更新→刷新镜像→释放），无需手动包夹；`block_swap_all_in/out` 仅用于特殊场景的手动批量控制。
> - 自定义 `torch.autograd.Function` 场景：`with` 块包在 `apply` 调用**外**时（推荐），前反向全自动、零改动；若把块写在 `forward` **内部**，backward 不会自动重物化（saved-tensor hooks 在 `apply` 时刻捕获），需在 backward 中显式调用 `w.block_swap_state.swap_in()`。

## 使用效果

受管张量在块外常驻 HBM 占用降为 0，CPU 侧新增等量 pinned 内存；每进入一次块引入一次 H2D 流回，只读块退出零拷贝。
