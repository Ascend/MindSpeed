# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch

from megatron.training import get_args
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput


AsyncAll2All_INPUT = []
AsyncAll2All_OUTPUT = []


def set_async_alltoall_inputs(comm_fn, *args, **kwargs):
    AsyncAll2All_INPUT.append((comm_fn, args, kwargs))


def get_async_alltoall_outputs():
    return AsyncAll2All_OUTPUT.pop(0)


def launch_async_all2all():
    if len(AsyncAll2All_INPUT) > 0:
        comm_fn, args, kwargs = AsyncAll2All_INPUT.pop(0)
        output, a2a_handle = comm_fn(*args, **kwargs)
        AsyncAll2All_OUTPUT.append((output, a2a_handle))


def launch_async_all2all_hook(_):
    launch_async_all2all()


def _discard_mhc_pre_recompute_output(mhc_module, hook_tensor):
    if not hasattr(mhc_module, 'discard_mhc_pre_ascend_output'):
        return
    if isinstance(hook_tensor, torch.Tensor) and hook_tensor.requires_grad:
        mhc_module.discard_mhc_pre_ascend_output(hook_tensor)


def discard_mlp_mhc_pre_recompute_output(layer, hook_tensor):
    _discard_mhc_pre_recompute_output(getattr(layer, 'mlp_mhc', None), hook_tensor)


def _should_defer_attention_recompute_for_mhc_post(layer):
    args = get_args()
    if not getattr(args, 'recompute_csa_attention', False):
        return False
    attn_mhc = getattr(layer, 'attn_mhc', None)
    should_recompute_post = getattr(attn_mhc, '_should_recompute_mhc_post_ascend', None)
    if callable(should_recompute_post):
        return should_recompute_post()
    return (
        getattr(args, 'enable_mhc', False)
        and getattr(args, 'mhc_recompute', False)
        and getattr(args, 'use_fused_mhc', False)
        and getattr(layer, 'training', False)
        and not getattr(attn_mhc, 'is_mtp_layer', False)
    )


def discard_attention_recompute_outputs_for_mhc_post(layer, hook_tensor):
    hook_tensor_has_grad = isinstance(hook_tensor, torch.Tensor) and hook_tensor.requires_grad
    attn_mhc = getattr(layer, 'attn_mhc', None)
    _discard_mhc_pre_recompute_output(attn_mhc, hook_tensor)

    defer_attention_recompute = getattr(layer, 'defer_attention_recompute_for_mhc_post', False)
    if defer_attention_recompute:
        norm_checkpoint = getattr(layer, 'norm_ckpt1', None)
        if getattr(layer, 'defer_attention_recompute_norm', False) and norm_checkpoint is not None:
            if hook_tensor_has_grad:
                norm_checkpoint.discard_output()
                hook_tensor.register_hook(norm_checkpoint.recompute)
            layer.norm_ckpt1 = None

        self_attention = getattr(layer, 'self_attention', None)
        if hook_tensor_has_grad and hasattr(self_attention, 'discard_csa_attention_output'):
            self_attention.discard_csa_attention_output(hook_tensor)

        bda_checkpoint = getattr(layer, 'self_attn_bda_checkpoint', None)
        if bda_checkpoint is not None:
            if hook_tensor_has_grad:
                bda_checkpoint.discard_output()
                hook_tensor.register_hook(bda_checkpoint.recompute)
            layer.self_attn_bda_checkpoint = None

    if hook_tensor_has_grad and hasattr(attn_mhc, 'discard_mhc_post_ascend_output'):
        attn_mhc.discard_mhc_post_ascend_output(hook_tensor)

    if defer_attention_recompute:
        layer.defer_attention_recompute_for_mhc_post = False
        layer.defer_attention_recompute_norm = False


def _self_attn_bda_is_output_only(bda_module, attention_bias):
    return attention_bias is None and bda_module.__class__.__name__ == 'AddOpWithBias'


def attention_forward(
    self,
    hidden_states,
    residual,
    attention_mask=None,
    inference_params=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    recompute_norm=False,
):
    args = get_args()
    defer_attention_recompute = _should_defer_attention_recompute_for_mhc_post(self)
    self.self_attn_bda_checkpoint = None
    self.defer_attention_recompute_for_mhc_post = defer_attention_recompute
    self.defer_attention_recompute_norm = recompute_norm if defer_attention_recompute else False
    if getattr(args, 'enable_mhc', False):
        # attn mHC pre
        post, comb = None, None
        hidden_states = self.attn_mhc(hidden_states, mhc_stage='pre')
        if isinstance(hidden_states, tuple):
            hidden_states, post, comb = hidden_states[0], hidden_states[1], hidden_states[2]

    # Optional Input Layer norm
    def pre_norm(hidden_states):
        args = get_args()
        input_layernorm_output = self.input_layernorm(hidden_states)
        if getattr(args, 'input_layernorm_in_fp32', False):
            input_layernorm_output = input_layernorm_output.float()
        return input_layernorm_output

    if recompute_norm:
        self.norm_ckpt1 = CheckpointWithoutOutput()
        input_layernorm_output = self.norm_ckpt1.checkpoint(pre_norm, False, hidden_states)
    else:
        input_layernorm_output = pre_norm(hidden_states)

    # Self attention.
    attention_output_with_bias = self.self_attention(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_context=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
    )

    # NOTE: `bias_dropout_add_exec_handler` could potentially be moved
    # inside the module provided in the `bias_dropout_add_spec` module?
    bda_module = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)
    if (
        defer_attention_recompute
        and isinstance(attention_output_with_bias, tuple)
        and len(attention_output_with_bias) == 2
        and isinstance(attention_output_with_bias[0], torch.Tensor)
        and not _self_attn_bda_is_output_only(bda_module, attention_output_with_bias[1])
    ):
        attention_output, attention_bias = attention_output_with_bias

        def self_attn_bda_without_bias(attention_output, residual):
            with self.bias_dropout_add_exec_handler():
                return bda_module((attention_output, None), residual, self.hidden_dropout)

        def self_attn_bda_with_bias(attention_output, attention_bias, residual):
            with self.bias_dropout_add_exec_handler():
                return bda_module((attention_output, attention_bias), residual, self.hidden_dropout)

        self.self_attn_bda_checkpoint = CheckpointWithoutOutput()
        if attention_bias is None:
            hidden_states = self.self_attn_bda_checkpoint.checkpoint(
                self_attn_bda_without_bias, False, attention_output, residual
            )
        else:
            hidden_states = self.self_attn_bda_checkpoint.checkpoint(
                self_attn_bda_with_bias, False, attention_output, attention_bias, residual
            )
    else:
        with self.bias_dropout_add_exec_handler():
            hidden_states = bda_module(attention_output_with_bias, residual, self.hidden_dropout)

    if recompute_norm and not defer_attention_recompute:
        self.norm_ckpt1.discard_output()
        hidden_states.register_hook(self.norm_ckpt1.recompute)

    if getattr(args, 'enable_mhc', False):
        # attn mHC post
        hidden_states = self.attn_mhc(hidden_states, mhc_stage='post', residual=residual, post=post, comb=comb)

    return hidden_states
