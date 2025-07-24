#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#  Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
import torch
from torch import Tensor
from mindspeed.core.pipeline_parallel.noop_layers.adaptor import NoopTransformerLayer
from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state, InferenceParams,
    tensor_parallel,
    PackedSeqParams,
    make_viewless_tensor,
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region
)
from .modules.utils import (
    NoopLayerGraph, LayerGraph, is_p2p_comm_needed,
    p2p_comm_helper, P2PCommOutput, P2PCommParams
)
from .overlap_funcs import (
    transformer_layer_forward_moe,
    transformer_layer_forward_dense,
    transformer_layer_forward_noop,
    transformer_layer_backward_moe,
    transformer_layer_backward_dense,
    transformer_layer_backward_noop,
    transformer_layer_forward_moe_backward_moe_overlaping,
)


class bwd_synchronize_check(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor
    
    @staticmethod
    def backward(ctx, grad):
        torch.cuda.synchronize()
        return grad


def transformer_layer_forward(*args, **kwargs):
    self = args[0]
    if kwargs['checkpoint']:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()

    with checkpoint_context:
        layer_forward_func = None
        if isinstance(self, NoopTransformerLayer):
            layer_forward_func = transformer_layer_forward_noop
        elif hasattr(self.mlp, 'experts'):
            layer_forward_func = transformer_layer_forward_moe
        else:
            layer_forward_func = transformer_layer_forward_dense

        return layer_forward_func(*args, **kwargs)


def transformer_layer_backward(
    layer_output_grad,
    layer_graph
):
    if layer_graph.checkpointed:
        with torch.enable_grad():
            _, _, restored_layer_graph = transformer_layer_forward(
                layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs, checkpoint=False
            )
            restored_layer_graph.unperm2_graph = (restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1])
            layer_graph = restored_layer_graph
    if isinstance(layer_graph, NoopLayerGraph):
        return transformer_layer_backward_noop(layer_output_grad, layer_graph)
    elif layer_graph.is_moe_layer:
        return transformer_layer_backward_moe(layer_output_grad, layer_graph)
    else:
        return transformer_layer_backward_dense(layer_output_grad, layer_graph)


def should_run_fb_overlap(fwd_layer, bwd_layer_graph: LayerGraph = None):
    flag = True
    fwd_is_moe = (not isinstance(fwd_layer, NoopTransformerLayer)) and hasattr(fwd_layer.mlp, 'experts')
    bwd_is_moe = (bwd_layer_graph is not None) and bwd_layer_graph.is_moe_layer

    return fwd_is_moe and bwd_is_moe


def transformer_layer_forward_backward_overlaping(
    fwd_layer,
    hidden_states,
    attention_mask,
    bwd_layer_output_grad=None,
    bwd_layer_graph: LayerGraph = None,
    bwd_unperm_a2a_handle=None,
    next_bwd_layer_graph: LayerGraph = None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    if not should_run_fb_overlap(fwd_layer, bwd_layer_graph):
        # no f&w overlaping
        if bwd_layer_graph is None:
            out = transformer_layer_forward(
                fwd_layer, hidden_states, attention_mask, context, context_mask, rotary_pos_emb, rotary_pos_cos,
                rotary_pos_sin, attention_bias, inference_params, packed_seq_params, checkpoint=checkpoint
            )
            if len(out) > 2 and checkpoint:
                out[2].record_layer_inputs(
                    attention_mask, context, context_mask, rotary_pos_emb, rotary_pos_cos,
                    rotary_pos_sin, attention_bias, inference_params, packed_seq_params
                )
            return out
        else:
            output, context, graph = transformer_layer_forward(
                fwd_layer, hidden_states, attention_mask, context, context_mask, rotary_pos_emb, rotary_pos_cos,
                rotary_pos_sin, attention_bias, inference_params, packed_seq_params, checkpoint=checkpoint
            )
            # handle fwd p2p communication
            next_iter_input_tensor, fwd_p2p_handles = None, None
            fwd_pp_comm_params = pp_comm_params
            if is_p2p_comm_needed(fwd_pp_comm_params):
                next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

            bwd_input_grad = transformer_layer_backward(bwd_layer_output_grad, bwd_layer_graph)
            next_iter_output_tensor_grad, bwd_p2p_handles = None, None
            if bwd_input_grad is not None:
                # handle bwd p2p communication
                if is_p2p_comm_needed(bwd_pp_comm_params):
                    next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_input_grad)

            if checkpoint:
                graph.record_layer_inputs(
                    attention_mask, context, context_mask, rotary_pos_emb, rotary_pos_cos,
                    rotary_pos_sin, attention_bias, inference_params, packed_seq_params
                )
            return (output, context, graph,
                    (bwd_input_grad, None),
                    P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_input_grad))

    else:
        fb_overlap_func = None
        if hasattr(fwd_layer.mlp, 'experts') and bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_moe_backward_moe_overlaping
        else:
            raise AssertionError('Check Layer Spec, f&b overlap func is not supported!')


        if bwd_layer_graph.checkpointed:
            _, _, bwd_layer_graph = transformer_layer_forward(
                bwd_layer_graph.layer, bwd_layer_graph.layer_input, *bwd_layer_graph.layer_inputs, checkpoint=False
            )

        out = fb_overlap_func(
            fwd_layer, hidden_states, attention_mask, bwd_layer_output_grad, bwd_layer_graph, bwd_unperm_a2a_handle,
            next_bwd_layer_graph, context, context_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, attention_bias,
            inference_params, packed_seq_params, pp_comm_params, bwd_pp_comm_params, checkpoint=checkpoint
        )

        if checkpoint:
            out[2].record_layer_inputs(
                attention_mask, context, context_mask, rotary_pos_emb, rotary_pos_cos,
                rotary_pos_sin, attention_bias, inference_params, packed_seq_params
            )

        return out


def dualpipev_fb_overlap_mtp_layer_forward(
    self,
    decoder_input: Tensor,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    sequence_len_offset: Tensor = None,
):
    """
    Perform the forward pass through the MTP layer with moe_fb_overlap.
    MTPTransformerLayer is used to overlap compute.

    Args:
        hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the
            sequence length, b is the batch size, and h is the hidden size.
        decoder_input (Tensor): Input tensor of shape [s, b, h] where s is the
            sequence length, b is the batch size, and h is the hidden size.
            At the (k - 1)-th MTP module, the i-th element of decoder input is
            the embedding of (i + K)-th tocken.
        attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
            self-attention.
        context (Tensor, optional): Context tensor for cross-attention.
        context_mask (Tensor, optional): Mask for cross-attention context
        rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
        attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
            to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
            Used as an alternative to apply attention mask for TE cuDNN attention.
        inference_params (InferenceParams, optional): Parameters for inference-time
            optimizations.
        packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
            processing.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
        [s, b, h], and optionally the updated context tensor if cross-attention is used.
    """
    assert context is None, f"multi token prediction + cross attention is not yet supported."
    assert (
        packed_seq_params is None
    ), f"multi token prediction + sequence packing is not yet supported."

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        import transformer_engine  # To keep out TE dependency when not training in fp8

        if self.config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif self.config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        fp8_recipe = TEDelayedScaling(
            config=self.config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, not self.config.fp8_wgrad),
        )
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
            )
        fp8_context = transformer_engine.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
        )
    else:
        fp8_context = nullcontext()

    with rng_context, fp8_context:
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(
            inp=decoder_input, requires_grad=True, keep_graph=True
        )
        hidden_states = self.hnorm(hidden_states)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )
        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel, all gather after linear_fc.
        hidden_states = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        hidden_states, _, _ = MTPTransformerLayer.apply(
            self.transformer_layer,
            self.config,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            inference_params,
            packed_seq_params,
        )

    # Layer norm before shared head layer.
    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)
    # TENorm produces a "viewed" tensor. This will result in schedule.py's
    # deallocate_output_tensor() throwing an error, so a viewless tensor is
    # created to prevent this.
    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    return hidden_states


class MTPTransformerLayer(torch.autograd.Function):
    '''
    A transformer layer with MindSpeedFbOverlapMoELayer in MTP block.
    '''
    @staticmethod
    def forward(ctx,
            layer,
            mtp_config,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            inference_params,
            packed_seq_params,
            checkpoint=False
    ):
        hidden_states = bwd_synchronize_check.apply(hidden_states)

        if checkpoint:
            checkpoint_context = torch.no_grad()
        else:
            checkpoint_context = nullcontext()

        with checkpoint_context:
            layer_forward_func = None
            if isinstance(layer, NoopTransformerLayer):
                layer_forward_func = transformer_layer_forward_noop
            elif hasattr(layer.mlp, 'experts'):
                layer_forward_func = transformer_layer_forward_moe
            else:
                layer_forward_func = transformer_layer_forward_dense

        with torch.enable_grad():
            hidden_states.requires_grad = True
            output, context, graph = layer_forward_func(layer,
                                                        hidden_states,
                                                        attention_mask,
                                                        context,
                                                        context_mask,
                                                        rotary_pos_emb,
                                                        rotary_pos_cos,
                                                        rotary_pos_sin,
                                                        attention_bias,
                                                        inference_params,
                                                        packed_seq_params,
                                                        checkpoint)

        output = bwd_synchronize_check.apply(output)
        ctx.graph = graph
        return output, context, graph
    
    @staticmethod
    def backward(ctx, layer_output_grad, context, layer_graph):
        layer_graph = ctx.graph
        layer_output_grad = transformer_layer_backward(layer_output_grad, layer_graph)
        return None, None, layer_graph.layer_input.grad, None, None, None, None, None, None, None, None, None, None
