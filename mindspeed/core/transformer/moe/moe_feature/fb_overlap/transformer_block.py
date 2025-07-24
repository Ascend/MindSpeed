# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import types
from functools import wraps
from typing import List, Union, Optional
from contextlib import nullcontext
import torch
from torch import Tensor
from mindspeed.core.transformer.moe.moe_feature import (
    get_args, InferenceParams,
    PackedSeqParams,
    make_viewless_tensor,
    BaseInferenceContext
)
from mindspeed.core.pipeline_parallel.noop_layers.adaptor import NoopTransformerLayer
from .modules.utils import (
    detach_tensor, LayerGraph, P2PCommParams
)
from .transformer_layer import transformer_layer_backward, transformer_layer_forward_backward_overlaping


def mtp_block_fb_overlap_forward_wrapper(fwd):
    @wraps(fwd)
    def wrapper(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        embedding=None,
        output_layer=None,
        output_weight: Optional[torch.Tensor] = None,
        compute_language_model_loss=None,
        bwd_block_output_grad=None,
        bwd_block_graphs=None,
        pp_comm_params: P2PCommParams = None,
        bwd_pp_comm_params: P2PCommParams = None
        ):
    
        hidden_states_main_model = fwd(
            self, 
            input_ids,
            position_ids,
            hidden_states,
            attention_mask,
            labels,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            inference_params,
            packed_seq_params,
            sequence_len_offset,
            extra_block_kwargs,
            runtime_gather_output,
            loss_mask,
            embedding,
            output_layer,
            output_weight,
            compute_language_model_loss,
        )
        
        return hidden_states_main_model

    return wrapper


def transformer_block_fb_overlap_init_wrapper(init_fn):
    @wraps(init_fn)
    def wrapper(*args, **kwargs):
        init_fn(*args, **kwargs)
        self = args[0]
        if get_args().moe_fb_overlap:
            def set_fwd_layer_graphs(self, layer_graphs: List[LayerGraph]):
                assert self.fwd_layer_graphs is None
                self.fwd_layer_graphs = layer_graphs

            def get_fwd_layer_graphs(self):
                assert self.fwd_layer_graphs is not None
                out = self.fwd_layer_graphs
                self.fwd_layer_graphs = None
                return out

            def set_pp_comm_output(self, pp_comm_output):
                assert self.pp_comm_output is None

                self.pp_comm_output = pp_comm_output

            def get_pp_comm_output(self):
                assert self.pp_comm_output is not None

                out = self.pp_comm_output
                self.pp_comm_output = None

                return out
            self.fwd_layer_graphs = None
            self.pp_comm_output = None
            self.forward = types.MethodType(transformer_block_forward_backward_overlaping, self)
            self.set_fwd_layer_graphs = types.MethodType(set_fwd_layer_graphs, self)
            self.get_fwd_layer_graphs = types.MethodType(get_fwd_layer_graphs, self)
            self.set_pp_comm_output = types.MethodType(set_pp_comm_output, self)
            self.get_pp_comm_output = types.MethodType(get_pp_comm_output, self)

            for layer in self.layers:
                layer.forward = types.MethodType(transformer_layer_forward_backward_overlaping, layer)

    return wrapper



def transformer_block_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_context: Optional[BaseInferenceContext] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[Tensor] = None,
    inference_params: Optional[BaseInferenceContext] = None,
):
    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(
        inp=hidden_states,
        requires_grad=True,
        keep_graph=True,
    )

    rng_context = nullcontext()
    fp8_context = nullcontext()

    assert not self.config.enable_cuda_graph
    layer_graphs = []

    with rng_context and fp8_context:
        for l_no, layer in enumerate(self.layers):
            checkpoint = False
            if self.config.recompute_granularity == 'full' and self.training:
                if self.config.recompute_method == 'block':
                    recompute_skip_num_layers = 0
                    if self.config.fp8 and not hidden_states.requires_grad:
                        recompute_skip_num_layers += 1
                    if (l_no >= recompute_skip_num_layers and l_no < self.config.recompute_num_layers + recompute_skip_num_layers):
                        checkpoint = True
                if self.config.recompute_method == 'uniform':
                    assert self.config.recompute_num_layers == 1
                    checkpoint = True
            if not (self.pre_process and l_no == 0):
                hidden_states = detach_tensor(hidden_states, checkpoint_forward=checkpoint)
            hidden_states, context, saved_graphs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                checkpoint=checkpoint
            )
            layer_graphs.append(saved_graphs)

    # Final layer norm.
    if self.post_process and self.post_layer_norm and self.final_layernorm is not None:
        detached_hidden_states = detach_tensor(hidden_states)
        layer_graphs[-1].unperm2_graph = (layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        hidden_states = self.final_layernorm(detached_hidden_states)

    if torch.is_grad_enabled() or get_args().schedules_method == 'dualpipev':
        self.set_fwd_layer_graphs(layer_graphs)

    return hidden_states


def transformer_block_forward_backward_overlaping(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_context: Optional[BaseInferenceContext] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[Tensor] = None,
    inference_params: Optional[BaseInferenceContext] = None,
    bwd_block_output_grad: Tensor = None,
    bwd_block_graphs: List[LayerGraph] = None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
):
    if bwd_block_graphs is None:
        return transformer_block_forward(
            self, hidden_states, attention_mask, context, context_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
            attention_bias, inference_context, packed_seq_params, sequence_len_offset, inference_params
        )

    fwd_block = self
    if not fwd_block.pre_process:
        # See set_input_tensor()
        hidden_states = fwd_block.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(
        inp=hidden_states,
        requires_grad=True,
        keep_graph=True,
    )

    rng_context = nullcontext()
    fp8_context = nullcontext()

    assert not fwd_block.config.enable_cuda_graph
    fwd_layer_graphs = []

    bwd_layer_output_grad = bwd_block_output_grad
    bwd_unperm_a2a_handle = None

    fwd_hidden_states, fwd_context = hidden_states, context
    with (((rng_context and fp8_context))):
        for l_no, fwd_layer in enumerate(fwd_block.layers):
            checkpoint = False
            if fwd_block.config.recompute_granularity == 'full' and fwd_block.training:
                if fwd_block.config.recompute_method == 'block':
                    recompute_skip_num_layers = 0
                    if fwd_block.config.fp8 and not hidden_states.requires_grad:
                        recompute_skip_num_layers += 1
                    if (l_no >= recompute_skip_num_layers and l_no < fwd_block.config.recompute_num_layers + recompute_skip_num_layers):
                        checkpoint = True
                if fwd_block.config.recompute_method == 'uniform':
                    assert fwd_block.config.recompute_num_layers == 1
                    checkpoint = True
            bwd_layer_graph = bwd_block_graphs.pop(-1)
            cur_p2p_params = pp_comm_params
            cur_bwd_p2p_params = bwd_pp_comm_params
            if l_no != len(fwd_block.layers) - 1 or len(bwd_block_graphs) > 0:
                # no need to excute pp communication in the intermediate layers
                cur_p2p_params = P2PCommParams()
                cur_bwd_p2p_params = P2PCommParams()
            next_bwd_layer_graph = None
            if (len(bwd_block_graphs) > 0 and
                not bwd_block_graphs[-1].checkpointed and
                l_no != len(fwd_block.layers) - 1 and
                not isinstance(fwd_block.layers[l_no + 1], NoopTransformerLayer)
            ):
                next_bwd_layer_graph = bwd_block_graphs[-1]
            # block with pre_process and first layer input do not detach for runing preprocess backward
            if not (self.pre_process and l_no == 0):
                fwd_hidden_states = detach_tensor(fwd_hidden_states, checkpoint_forward=checkpoint)
            fwd_hidden_states, fwd_context, fwd_layer_graph, \
            (bwd_layer_output_grad, bwd_unperm_a2a_handle), \
            pp_comm_output = \
                fwd_layer(
                    fwd_hidden_states,
                    attention_mask,
                    bwd_layer_output_grad,
                    bwd_layer_graph=bwd_layer_graph,
                    bwd_unperm_a2a_handle=bwd_unperm_a2a_handle,
                    next_bwd_layer_graph=next_bwd_layer_graph,
                    context=fwd_context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    pp_comm_params=cur_p2p_params,
                    bwd_pp_comm_params=cur_bwd_p2p_params,
                    checkpoint=checkpoint
                )
            fwd_layer_graphs.append(fwd_layer_graph)

    # Final layer norm.
    if fwd_block.post_process and fwd_block.post_layer_norm and fwd_block.final_layernorm is not None:
        detached_hidden_states = detach_tensor(fwd_hidden_states)
        fwd_layer_graphs[-1].unperm2_graph = (fwd_layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        fwd_hidden_states = fwd_block.final_layernorm(detached_hidden_states)

    if torch.is_grad_enabled() or get_args().schedules_method == 'dualpipev':
        self.set_fwd_layer_graphs(fwd_layer_graphs)
        self.set_pp_comm_output(pp_comm_output)

    return fwd_hidden_states


def transformer_block_backward(
    block_output_grad,
    layer_graphs: List[LayerGraph],
):
    # should call backward fisrt for final_layernorm and postprocess grad
    layer_output_grad = block_output_grad
    while len(layer_graphs) > 0:
        layer_graph = layer_graphs.pop(-1)
        layer_output_grad = transformer_layer_backward(layer_output_grad, layer_graph)

    return layer_output_grad

