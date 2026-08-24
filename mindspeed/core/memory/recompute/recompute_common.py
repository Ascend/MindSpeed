# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from megatron.core import mpu
from torch.utils.checkpoint import detach_variable

from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state


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


def _get_pipeline_model_parallel_layout(config):
    layout = getattr(config, 'pipeline_model_parallel_layout', None)
    if layout is None:
        return None

    from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import (
        PipelineParallelLayerLayout,
    )

    if isinstance(layout, PipelineParallelLayerLayout):
        return layout

    pp_size = getattr(config, 'pipeline_model_parallel_size', None)
    if pp_size is None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    if isinstance(layout, str):
        return PipelineParallelLayerLayout.from_str(layout, pp_size)
    if isinstance(layout, list):
        return PipelineParallelLayerLayout(layout, pp_size)
    raise TypeError(
        f'pipeline_model_parallel_layout must be a str, list, or PipelineParallelLayerLayout, but got {type(layout)}'
    )


def _get_layout_recompute_priority(layout, layer_number, enable_per_pp_rank):
    from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import LayerType

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank() or 0
    vpp_size = layout.virtual_pipeline_model_parallel_size
    if not 0 <= vpp_rank < vpp_size:
        raise ValueError(f'Invalid virtual pipeline rank {vpp_rank} for layout with {vpp_size} virtual stages.')

    decoder_layer_id = layer_number - 1
    layer_ids_per_vpp_rank = [[] for _ in range(vpp_size)]
    decoder_layer_offset = 0
    for current_vpp_rank in range(vpp_size):
        for current_pp_rank in range(layout.pipeline_model_parallel_size):
            num_decoder_layers = layout.layout[current_pp_rank][current_vpp_rank].count(LayerType.decoder)
            layer_ids = list(range(decoder_layer_offset, decoder_layer_offset + num_decoder_layers))
            if current_pp_rank == pp_rank:
                layer_ids_per_vpp_rank[current_vpp_rank] = layer_ids
            decoder_layer_offset += num_decoder_layers

    current_chunk_layer_ids = layer_ids_per_vpp_rank[vpp_rank]
    if decoder_layer_id not in current_chunk_layer_ids:
        raise ValueError(
            f'Decoder layer {layer_number} is not present in pipeline layout for '
            f'pp_rank={pp_rank}, vpp_rank={vpp_rank}. layout={layout}'
        )

    local_layer_index = current_chunk_layer_ids.index(decoder_layer_id)
    if not enable_per_pp_rank:
        return local_layer_index

    recompute_priority = 0
    max_chunk_size = max((len(layer_ids) for layer_ids in layer_ids_per_vpp_rank), default=0)
    for layer_index in range(max_chunk_size):
        for layer_ids in layer_ids_per_vpp_rank:
            if layer_index >= len(layer_ids):
                continue
            if layer_ids[layer_index] == decoder_layer_id:
                return recompute_priority
            recompute_priority += 1

    raise RuntimeError(f'Failed to calculate recompute priority for decoder layer {layer_number}.')


def get_recompute_priority(config, layer_number, enable_per_pp_rank=None):
    """Return the layer recompute priority for uniform or custom pipeline layouts."""
    if layer_number is None:
        raise ValueError('layer_number must not be None when calculating recompute priority.')

    if enable_per_pp_rank is None:
        enable_per_pp_rank = getattr(config, 'enable_recompute_layers_per_pp_rank', False)

    layout = _get_pipeline_model_parallel_layout(config)
    if layout is not None:
        return _get_layout_recompute_priority(layout, layer_number, enable_per_pp_rank)

    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = getattr(config, 'virtual_pipeline_model_parallel_size', None)
    pp_size = getattr(config, 'pipeline_model_parallel_size', None)

    if vpp_size is not None:
        layer_per_chunk = getattr(config, 'num_layers_per_virtual_pipeline_stage', None)
        if layer_per_chunk is None:
            layer_per_chunk = config.num_layers // pp_size // vpp_size
    elif pp_size is not None:
        layer_per_chunk = config.num_layers // pp_size
    else:
        layer_per_chunk = config.num_layers

    if vpp_rank is None or not enable_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not enable_per_pp_rank:
        vpp_size = 1
    return ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank


def should_recompute(config, layer_number, num_recompute):
    full_recompute_layers = config.recompute_num_layers
    if not full_recompute_layers and num_recompute is None:
        return True

    recompute_priority = get_recompute_priority(config, layer_number)

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
