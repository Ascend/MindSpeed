# block_swap

## Background

During large-model training and inference, some tensors (expert weights, constant tables) are only touched by a few lines of code yet occupy HBM the whole time. For custom ops that do not go through Megatron's linear-layer wrappers (raw `matmul`/`bmm`, MoE grouped matmuls, custom projections), there is no ready-made mechanism to swap this memory in and out on demand.

## Solution

`block_swap` is a code-level context manager: wrap any NPU tensors in a `with` block - on enter they are materialized from a CPU pinned mirror (H2D), on exit their device storage is released (`storage().resize_(0)`). Inside the block the tensors behave like ordinary device tensors. The mechanism shares its DNA with swap-optimizer: one H2D stream per use trades for a 100% resident-memory saving of that tensor.

1. A pinned mirror is created lazily on first use and the tensor is adopted into the manager (reused afterwards).
2. The block registers saved-tensor hooks: tensors saved by forward inside the block are re-materialized when a backward *outside* the block unpacks them, so "forward inside, backward outside" is training-safe.
3. Automatic Megatron training-loop integration: adopting a tensor takes over the Megatron optimizers' `step_with_ready_grads` / `reload_model_params` - the step materializes all managed tensors, and afterwards refreshes the mirrors (D2H) and releases, with no extra user code;
4. Nested blocks are supported (depth counting; only the outermost exit releases); in-place modification inside the block is detected via the version counter and refreshes the mirror automatically.

## Usage Scenarios

* Large tensors in custom ops inside the Megatron + MindSpeed training framework: raw `matmul`/`bmm`, MoE grouped matmuls (GroupedMLP `weight1`/`weight2`), custom projections;
* inference / constant tables: low access frequency, large memory footprint.

## Usage

Inference / custom forward:

```python
from mindspeed.core.memory.block_swap import block_swap

# MoE grouped expert matmul, weight1/weight2 of shape [num_local_experts, in, out]
with block_swap(self.weight1, self.weight2):
    hidden = torch.bmm(x, self.weight1.transpose(-1, -2))
    out = torch.bmm(hidden, self.weight2)
# exiting the block releases both weights' device storage
```

Inside the Megatron training loop (backward and optimizer updates are handled by the framework automatically - just wrap the custom op):

```python
from mindspeed.core.memory.block_swap import block_swap

# e.g. inside a GroupedMLP forward:
with block_swap(self.weight1, self.weight2):
    hidden = torch.bmm(x, self.weight1.transpose(-1, -2))
    out = torch.bmm(hidden, self.weight2)
# outside the block: backward re-materializes automatically; the optimizer
# step materializes / updates / refreshes mirrors / releases automatically
```

Parameters: `*tensors` - any number of NPU tensors (required); `copy_back` - defaults to `False` (copy-free exit for read-only blocks), pass `True` when the block modifies values through `.data` writes.

Manual bulk control: `block_swap_all_in()` / `block_swap_all_out(copy_to_host=True)`.

> [!NOTE]
>
> - Tensors must be on the NPU (CPU tensors are rejected; wrap model weights only after the model has been moved to the device).
> - Views into larger storages warn: releasing them also releases sibling tensors sharing the storage.
> - **Optimizer steps must not run inside a block**; this is rejected with a RuntimeError.
> - Writes that do not bump the version counter (`.data.copy_`) cannot be detected; pass `copy_back=True` explicitly.
> - Double backward (`create_graph=True`) is unsupported; a tensor must not be used after its block exits.
> - Optimizer steps are taken over by the Megatron integration automatically (materialize -> update -> refresh mirror -> release); use `block_swap_all_in/out` only for special manual bulk control.
> - With a custom `torch.autograd.Function`: wrapping the block **around** the `apply` call (recommended) needs zero changes and covers forward/backward automatically; putting the block **inside** `forward` does NOT auto-rematerialize in backward (saved-tensor hooks are captured at `apply` time) - call `w.block_swap_state.swap_in()` explicitly in the backward.

## Effects

Outside their blocks, managed tensors' resident HBM usage drops to zero; an equal amount of pinned host memory is added. Each block entry introduces one H2D stream; read-only blocks exit copy-free.
