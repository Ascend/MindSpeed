import enum
import os
from functools import wraps

from contextlib import nullcontext
import torch
import torch_npu
import torch.nn.functional as F

from megatron import core
from megatron.training import get_args
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core import tensor_parallel, parallel_state, mpu
from megatron.core.utils import make_viewless_tensor
from megatron.legacy.model.transformer import bias_dropout_add_fused_train, get_bias_dropout_add, bias_dropout_add_fused_inference
from megatron.legacy.model.enums import AttnMaskType, LayerType, AttnType
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.transformer.moe.moe_utils import only_recompute_activation


def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
        fn(self, *args, **kwargs)
        if self.mlp.__class__ is MoELayer:
            if self.mlp.experts.__class__ is GroupedMLP:
                self.mlp.experts.layer_number = self.layer_number
            if self.mlp.experts.__class__ is SequentialMLP:
                for expert in self.mlp.experts.local_experts:
                    expert.layer_number = self.layer_number
            global_args = get_args()
            if global_args.n_shared_experts:
                self.mlp.shared_experts.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper


def parallel_transformer_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.recompute_method != 'block' and not global_args.swap_attention:
            output = forward_func(*args, **kwargs)
        else:
            output = parallel_transformer_checkpointed_forward(*args, **kwargs)
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

    global_args = get_args()
    num_layers_per_pipeline_rank = global_args.num_layers // global_args.pipeline_model_parallel_size
    if self.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and
        # checkpoint the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        if not global_args.swap_attention:
            l = 0
            while l < num_layers_per_pipeline_rank:
                hidden_states = tensor_parallel.checkpoint(
                    custom(l, l + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask,
                    encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)

                l += self.recompute_num_layers
        else:
            for l in range(num_layers_per_pipeline_rank):
                hidden_states = custom(l, l + 1)(
                    hidden_states, attention_mask,
                    encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
    elif self.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
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

            if should_recompute() and not global_args.swap_attention:
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


def core_mlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.layer_number = getattr(self, "layer_number", None)
        is_recompute_activation = should_recompute_activation(self.layer_number)
        if get_args().moe_alltoall_overlap_comm and not isinstance(args[-1], torch.Tensor):
            moe_ctx = args[-1]
            args = args[:-1]

        def activation_function(*function_args):
            intermediate, bias = function_args
            if bias is not None:
                intermediate = intermediate + bias
            if self.config.gated_linear_unit:
                assert (self.config.activation_func == F.silu), 'Activation function must be silu when using fused_swiglu'
                if not hasattr(self, 'origin_activation_func'):
                    self.origin_activation_func = self.activation_func
                self.activation_func = fused_swiglu
                intermediate = self.activation_func(intermediate)
            else:
                intermediate = self.activation_func(intermediate)

            return intermediate

        moe_zero_memory = get_args().moe_zero_memory
        if not (is_recompute_activation or moe_zero_memory != "disable"):
            if hasattr(self, 'origin_activation_func'):
                self.activation_func = self.origin_activation_func
            output, output_bias = fn(self, *args, **kwargs)
        elif moe_zero_memory == "level1" and not only_recompute_activation(self.layer_number):
            if self.shared_expert:
                self.activation_function = activation_function
                hidden_states = args[0]
                fc1_out_parallel, bias_parallel = self.linear_fc1(hidden_states)
                act_out_parallel = activation_function(fc1_out_parallel, bias_parallel)
                output, output_bias = self.linear_fc2(act_out_parallel)
                fc1_out_parallel.untyped_storage().resize_(0)
                act_out_parallel.untyped_storage().resize_(0)
                moe_ctx.shared_fc1_out = fc1_out_parallel
                moe_ctx.shared_act_out = act_out_parallel
            else:
                output, output_bias = fn(self, *args, **kwargs)
        else:
            hidden_states = args[0]
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(activation_function,
                                                                                  False,
                                                                                  intermediate_parallel,
                                                                                  bias_parallel)
            # [s, b, h]
            output, output_bias = self.linear_fc2(intermediate_parallel)

            # discard the output of the activation function,
            # which will be restored by recomputation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            if output.requires_grad:
                output.register_hook(self.activation_checkpoint_manager.recompute)
        return output, output_bias
    return wrapper


def norm_recompute_forward(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
):
    # hidden_states: [s, b, h]

    # Residual connection.
    residual = hidden_states

    # Optional Input Layer norm
    self.norm_ckpt1 = CheckpointWithoutOutput()
    input_layernorm_output = self.norm_ckpt1.checkpoint(self.input_layernorm, False, hidden_states)

    # Self attention.
    attention_output_with_bias = self.self_attention(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
    )

    self.norm_ckpt1.discard_output()
    if self.training:
        attention_output_with_bias[0].register_hook(self.norm_ckpt1.recompute)

    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm after self-attention
    pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

    # Cross attention.
    attention_output_with_bias = self.cross_attention(
        pre_cross_attn_layernorm_output,
        attention_mask=context_mask,
        key_value_states=context,
        inference_params=inference_params,
    )

    if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
        context = attention_output_with_bias["context"]

    with self.bias_dropout_add_exec_handler():
        hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm post the cross-attention.
    self.norm_ckpt2 = CheckpointWithoutOutput()
    pre_mlp_layernorm_output = self.norm_ckpt2.checkpoint(self.pre_mlp_layernorm, False, hidden_states)

    # MLP.
    mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

    if self.training:
        self.norm_ckpt2.discard_output()
        mlp_output_with_bias[0].register_hook(self.norm_ckpt2.recompute)

    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, self.hidden_dropout
        )

    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    return output, context
