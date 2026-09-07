# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from megatron.core import mpu
from torch.utils.checkpoint import detach_variable

from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state


def _resolve_pipeline_model_parallel_layout(config):
    """Return a parsed pipeline layout for either args or TransformerConfig."""
    layout = getattr(config, "pipeline_model_parallel_layout", None)
    if not isinstance(layout, (str, list)):
        return layout

    from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import (
        PipelineParallelLayerLayout,
    )

    pp_size = int(getattr(config, "pipeline_model_parallel_size", 1))
    if isinstance(layout, str):
        return PipelineParallelLayerLayout.from_str(layout, pp_size)
    return PipelineParallelLayerLayout(layout, pp_size)


def _get_recompute_layer_position(config, layer_number, vp_stage=None):
    """Return the local layer index, VP stage and VP size for a global layer number."""
    configured_vpp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
    vpp_size = configured_vpp_size or 1
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    layout = _resolve_pipeline_model_parallel_layout(config)

    if configured_vpp_size is None:
        vp_stage = 0
    elif vp_stage is None:
        vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()

    # Megatron 0.18 does not set the global virtual rank while constructing model chunks.
    # Infer it from the global layer number for callers that cannot carry module metadata
    # through an autograd Function.
    if vp_stage is None:
        if layout is not None:
            for candidate in range(vpp_size):
                offset = layout.get_layer_offset(vp_stage=candidate, pp_rank=pp_rank)
                count = layout.get_num_layers_to_build(vp_stage=candidate, pp_rank=pp_rank)
                if offset < layer_number <= offset + count:
                    vp_stage = candidate
                    break
        else:
            from megatron.core.transformer import transformer_block, transformer_layer

            for candidate in range(vpp_size):
                offset = transformer_layer.get_transformer_layer_offset(config, candidate, pp_rank)
                count = transformer_block.get_num_layers_to_build(config, candidate, pp_rank)
                if offset < layer_number <= offset + count:
                    vp_stage = candidate
                    break

    if vp_stage is None:
        vp_stage = 0

    if layout is not None:
        layer_offset = layout.get_layer_offset(vp_stage=vp_stage, pp_rank=pp_rank)
        local_layer_index = layer_number - layer_offset - 1
    else:
        from megatron.core.transformer import transformer_layer

        native_vp_stage = vp_stage if configured_vpp_size is not None else None
        layer_offset = transformer_layer.get_transformer_layer_offset(
            config,
            native_vp_stage,
            pp_rank,
        )
        local_layer_index = layer_number - layer_offset - 1

    return local_layer_index, vp_stage, vpp_size


def get_recompute_priority(config, layer_number, vp_stage=None):
    """Return a PP/VPP-aware recompute order for a global transformer layer number."""
    local_layer_index, vp_stage, vpp_size = _get_recompute_layer_position(
        config,
        layer_number,
        vp_stage=vp_stage,
    )

    return local_layer_index * vpp_size + vp_stage


class CheckpointFunctionWithoutOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, checkpoint, *args):
        with torch.no_grad():
            outputs = run_function(*args)

        # Store everything
        ctx.save_for_backward(*detach_variable(args))
        checkpoint.ctx = ctx

        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        outputs = ctx.outputs
        torch.autograd.backward(outputs, args)
        ctx.outputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput:
    def __init__(self, get_cuda_rng_tracker_func):
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None
        self.outputs = None
        self.ctx = None
        self.get_cuda_rng_tracker = get_cuda_rng_tracker_func

    def checkpoint(self, run_function, distribute_saved_activations, *args):
        self.run_function = run_function

        if distribute_saved_activations:
            raise RuntimeError("CheckpointFunctionWithoutOutput does not support distribute_saved_activations")

        # Copy the rng states.
        self.fwd_cpu_rng_state = torch.get_rng_state()
        self.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        self.fwd_cuda_rng_state_tracker = self.get_cuda_rng_tracker().get_states()

        outputs = CheckpointFunctionWithoutOutput.apply(run_function, self, *args)
        self.outputs = outputs
        if isinstance(self.outputs, torch.Tensor):
            self.outputs = (self.outputs,)

        return outputs

    def discard_output(self):
        for output in self.outputs:
            output.untyped_storage().resize_(0)

    def recompute(self, _):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")

        # Store the current states.
        cur_cpu_rng_state = torch.get_rng_state()
        cur_cuda_rng_state = torch.cuda.get_rng_state()
        cur_cuda_rng_state_tracker = self.get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(self.fwd_cpu_rng_state)
        _set_cuda_rng_state(self.fwd_cuda_rng_state)
        self.get_cuda_rng_tracker().set_states(self.fwd_cuda_rng_state_tracker)

        with torch.enable_grad():
            outputs = self.run_function(*self.ctx.saved_tensors)
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(cur_cpu_rng_state)
        _set_cuda_rng_state(cur_cuda_rng_state)
        self.get_cuda_rng_tracker().set_states(cur_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for output, recomputation_output in zip(self.outputs, outputs):
            output_size = recomputation_output.untyped_storage().size()
            output.untyped_storage().resize_(output_size)
            with torch.no_grad():
                output.untyped_storage().copy_(recomputation_output.untyped_storage())

        self.ctx.outputs = outputs
        self.outputs = None
        self.ctx = None


def should_recompute(config, layer_number, num_recompute, vp_stage=None):
    enable_per_pp_rank = getattr(config, 'enable_recompute_layers_per_pp_rank', False)
    if enable_per_pp_rank:
        recompute_priority = get_recompute_priority(config, layer_number, vp_stage=vp_stage)
    else:
        recompute_priority, _, _ = _get_recompute_layer_position(
            config,
            layer_number,
            vp_stage=vp_stage,
        )
    full_recompute_layers = config.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True

        return False

    if num_recompute is None:
        return True

    return recompute_priority < num_recompute
