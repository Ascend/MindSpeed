import enum
import os
from functools import wraps

from contextlib import nullcontext
import torch
import torch_npu
import torch.nn.functional as F

from megatron import core
from megatron.training import get_args, get_num_microbatches
from megatron.core import tensor_parallel, parallel_state, mpu
from megatron.legacy.model.transformer import bias_dropout_add_fused_train, get_bias_dropout_add, bias_dropout_add_fused_inference
from megatron.legacy.model.enums import AttnMaskType, LayerType, AttnType


def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        fn(self, *args, **kwargs)
        if self.mlp.__class__ is MoELayer:
            self.mlp.experts.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper


def parallel_transformer_layer_forward_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.optimize_recomp_communication_level == 0:
            output = forward_func(*args, **kwargs)
        else:
            output = parallel_transformer_layer_forward(*args, **kwargs)
        return output
    return row_parallel_forward


class TransformerLayerStage(enum.Enum):
    attn = 1
    ffn = 2


def parallel_transformer_layer_forward(self, hidden_states, attention_mask=None,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None,
                transformer_stage=None):
    def _ckpt_comm_process_sp(residual):
        args = get_args()
        if args.optimize_recomp_communication_level:
            if args.optimize_recomp_communication_status > 2:
                if args.sequence_parallel:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    residual_empty = torch.empty(residual.shape, dtype=residual.dtype,
                                                 device=torch.cuda.current_device(),
                                                 requires_grad=False)
                    residual = torch.concat([residual_empty] * tp_rank + [residual] + [residual_empty] * (
                            parallel_state.get_tensor_model_parallel_world_size() - tp_rank - 1), 0)
                    return residual
        return None
    # hidden_states: [s, b, h]
    if self.bias_dropout_fusion:
        if self.training:
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = bias_dropout_add_fused_inference
    else:
        bias_dropout_add_func = get_bias_dropout_add(self.training)

    global_args = get_args()
    if transformer_stage is None or transformer_stage == TransformerLayerStage.attn:
        norm_output = self.input_norm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                norm_output,
                attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb)

        # Residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states

        residual_comm = _ckpt_comm_process_sp(residual) if not global_args.use_ascend_mc2 else residual
        residual = residual if residual_comm is None else residual_comm
        if self.drop_path is None:
            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                norm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            norm_input = residual + self.drop_path(out)
        if transformer_stage == TransformerLayerStage.attn:
            return norm_input
        
    if transformer_stage is None or transformer_stage == TransformerLayerStage.ffn:
        if transformer_stage == TransformerLayerStage.ffn:
            norm_input = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        # Layer norm post the self attention.
        norm_output = self.post_attention_norm(norm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(norm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        residual_comm = _ckpt_comm_process_sp(residual) if not global_args.use_ascend_mc2 else residual
        residual = residual if residual_comm is None else residual_comm

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)

            output = core.utils.make_viewless_tensor(inp=output,
                                                     requires_grad=output.requires_grad,
                                                     keep_graph=True)
        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(mlp_output,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        if self.layer_type == LayerType.retro_decoder_with_retriever:
            return output, retriever_output
        else:
            return output


def parallel_transformer_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.optimize_recomp_communication_level == 0:
            if global_args.recompute_method != 'block':
                output = forward_func(*args, **kwargs)
            else:
                output = parallel_transformer_checkpointed_forward(*args, **kwargs)
        else:
            output = parallel_transformer_checkpointed_forward_tp_optimized(*args, **kwargs)
        return output
    return row_parallel_forward


def parallel_transformer_checkpointed_forward(self, hidden_states, attention_mask,
                                              encoder_output, enc_dec_attn_mask,
                                              rotary_pos_emb, is_first_microbatch):
    """Forward method with activation checkpointing."""

    def custom(start, end):
        def custom_forward(*args, **kwargs):
            x_, *args = args
            for index in range(start, end):
                layer = self._get_layer(index)
                x_ = layer(x_, *args, **kwargs)
            return x_

        return custom_forward
    
    args = get_args()
    if self.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        global_args = get_args()
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        vpp_size = global_args.virtual_pipeline_model_parallel_size
        if vpp_rank is None or not global_args.enable_recompute_layers_per_pp_rank:
            vpp_rank = 0
        if vpp_size is None or not global_args.enable_recompute_layers_per_pp_rank:
            vpp_size = 1
        for l in range(self.num_layers):
            # The number of layers each pipeline rank recomputes is self.recompute_num_layers.
            # If self.recompute_num_layers cannot divide exactly  the number of layers in each pp rank,
            # we try to balance the number of recomputed layers in each model chunk.
            # e.g. with 8 layers, 2 stages, and 2 virtual stages, the assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]   [4, 5]
            # Stage 1: [2, 3]   [6, 7]
            # With self.recompute_num_layers = 2, we will recompute layers 0,4 for stage 0, and 2,6 for stage 1.
            # With self.recompute_num_layers = 3, we will recompute layers 0,1,4 for stage 0, and 2,3,6 for stage 1.
            def should_recompute():
                if global_args.reduce_recompute_for_last_chunk:
                    def is_last_layer():
                        return (l == self.num_layers - 1) and mpu.is_pipeline_last_stage()

                    return ((l * vpp_size + vpp_rank) < self.recompute_num_layers) and not is_last_layer()
                else:
                    return (l * vpp_size + vpp_rank) < self.recompute_num_layers

            if should_recompute() and not args.swap_attention:
                hidden_states = tensor_parallel.checkpoint(
                    custom(l, l + 1),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask,
                    encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
            else:
                hidden_states = custom(l, l + 1)(
                    hidden_states, attention_mask,
                    encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states


def parallel_transformer_checkpointed_forward_tp_optimized(self, hidden_states, attention_mask,
                          encoder_output, enc_dec_attn_mask,
                          rotary_pos_emb, is_first_microbatch):
    """Forward method with activation checkpointing."""
    def custom(start, end):
        def custom_forward(*args, **kwargs):
            x_, *args = args
            for index in range(start, end):
                layer = self._get_layer(index)
                x_ = layer(x_, *args, **kwargs)
            return x_
        return custom_forward
    args = get_args()
    if args.optimize_recomp_communication_level > 1:
        def custom_nocomm(start, end):
            def custom_attn(*args, **kwargs):
                kwargs['transformer_stage'] = TransformerLayerStage.attn
                layer = self._get_layer(start)
                output = layer(*args, **kwargs)
                return output

            def custom_ffn(*args, **kwargs):
                kwargs['transformer_stage'] = TransformerLayerStage.ffn
                layer = self._get_layer(start)
                output = layer(*args, **kwargs)
                return output
            return custom_attn, custom_ffn

        def custom_checkpoint(function, distribute_saved_activations, *args):
            attn, ffn = function
            attn_output = checkpoint_func(attn, distribute_saved_activations, *args)
            args = tuple([attn_output]) + args[1:]
            return checkpoint_func(ffn, distribute_saved_activations, *args)

        custom = custom_nocomm
        if not hasattr(self, "replace_checkpoint_flag"):
            self.replace_checkpoint_flag = False
        if not self.replace_checkpoint_flag:
            checkpoint_func = tensor_parallel.checkpoint
            tensor_parallel.checkpoint = custom_checkpoint
            self.replace_checkpoint_flag = True

    if self.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and
        # checkpoint the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        l = 0
        while l < self.num_layers:
            hidden_states = tensor_parallel.checkpoint(
                custom(l, l + self.recompute_num_layers),
                self.distribute_saved_activations,
                hidden_states, attention_mask,
                encoder_output, enc_dec_attn_mask,
                None, None, None, None, rotary_pos_emb)
            l += self.recompute_num_layers

    elif self.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        vpp_size = args.virtual_pipeline_model_parallel_size
        if vpp_rank is None or not args.enable_recompute_layers_per_pp_rank:
            vpp_rank = 0
        if vpp_size is None or not args.enable_recompute_layers_per_pp_rank:
            vpp_size = 1
        for l in range(self.num_layers):
            # The number of layers each pipeline rank recomputes is self.recompute_num_layers.
            # If self.recompute_num_layers cannot divide exactly  the number of layers in each pp rank,
            # we try to balance the number of recomputed layers in each model chunk.
            # e.g. with 8 layers, 2 stages, and 2 virtual stages, the assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]   [4, 5]
            # Stage 1: [2, 3]   [6, 7]
            # With self.recompute_num_layers = 2, we will recompute layers 0,4 for stage 0, and 2,6 for stage 1.
            # With self.recompute_num_layers = 3, we will recompute layers 0,1,4 for stage 0, and 2,3,6 for stage 1.
            if l * vpp_size + vpp_rank < self.recompute_num_layers:
                hidden_states = tensor_parallel.checkpoint(
                    custom(l, l + 1),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask,
                    encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
            else:
                hidden_states = custom(l, l + 1)(
                    hidden_states, attention_mask,
                    encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states
