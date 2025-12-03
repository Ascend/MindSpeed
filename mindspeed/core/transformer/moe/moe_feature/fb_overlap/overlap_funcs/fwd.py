# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core.utils import make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from ..modules.attention import attention_forward
from ..modules.utils import (
    detach_tensor,
    NoopLayerGraph, LayerGraph,
)


def router_forward(
    self,
    hidden_states
):
    probs, routing_map = self.mlp.router(hidden_states)

    return probs, routing_map


def transformer_layer_forward_moe(
    self,
    hidden_states,
    attention_mask=None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    checkpoint=False
):
    # hidden_states: [s, b, h]
    args = get_args()
    use_shared_experts = hasattr(self.mlp, 'shared_experts') and self.mlp.shared_experts is not None
    recomp_norm = getattr(args, 'recompute_norm', False)
    self.mlp.experts.layer_number = self.layer_number

    detached_layer_input = hidden_states

    # Residual connection.
    residual1 = detached_layer_input

    # input_layernorm + AttentionForward
    hidden_states = attention_forward(
        self, detached_layer_input, residual1,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
        recompute_norm=recomp_norm
    )

    attention_out, detached_attention_out = hidden_states, detach_tensor(hidden_states, checkpoint_forward=checkpoint)

    # Residual connection.
    residual2 = detached_attention_out

    # Layer Norm after attention
    if recomp_norm:
        self.norm_ckpt2 = CheckpointWithoutOutput()
        pre_mlp_layernorm_output = self.norm_ckpt2.checkpoint(self.pre_mlp_layernorm, False, detached_attention_out)
    else:
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(detached_attention_out)

    # MLP.
    dispatcher = self.mlp.token_dispatcher
    detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)
    if use_shared_experts:
        dispatcher.overlap_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(dispatcher.overlap_stream):
            # Shared Experts PreComm.
            self.mlp.shared_experts.pre_forward_comm(detached_mlp_input)
            shared_fc1_input = self.mlp.shared_experts.cached_fc1_input
    else:
        shared_fc1_input = None

    # Router forward.
    if hasattr(self.mlp.token_dispatcher, "num_tokens_per_expert") \
            and (getattr(get_args(), "enable_expert_placement", False) or getattr(get_args(), "print_expert_load", False)):
        self.mlp.predict_expert_load(self.mlp.token_dispatcher.num_tokens_per_expert)
    probs, routing_map = router_forward(self, detached_mlp_input)
    shared_expert_output = None

    # Token Perm1 Forward
    probs_detached = detach_tensor(probs, checkpoint_forward=checkpoint)
    perm1_out, perm1_probs, tokens_per_expert = dispatcher.token_permute1(detached_mlp_input, probs_detached, routing_map)

    if use_shared_experts:
        # Shared Experts Forward.
        self.mlp.shared_experts.linear_fc1_forward_and_act()
        self.mlp.shared_experts.linear_fc2_forward()
    if dispatcher.num_local_experts > 1:
        # launch synchronization here to wait for non-blocking mem copy in preprocess func.
        dispatcher.cuda_sync_point = "no_sync"
        torch.npu.current_stream().synchronize()

    # Async Perm A2A.
    from ..modules.token_dispatcher import PREMUTE_FINISH_EVENT
    if PREMUTE_FINISH_EVENT is not None:
        #Wait for permute1 finish.
        torch.npu.current_stream().wait_event(PREMUTE_FINISH_EVENT)

    (perm_a2a_out, perm_a2a_handle), (perm_prob_a2a_out, perm_prob_a2a_handle) = dispatcher.async_dispatch_comm(perm1_out, perm1_probs)

    if recomp_norm:
        self.norm_ckpt2.discard_output()
    # overlap perm a2a by shared experts computation.
    perm_a2a_handle.wait()
    # perm1_out tensor storage is not need by backward,
    # but backward func of perm1_out is needed, so resize the storage but keep tensor.
    perm1_out.untyped_storage().resize_(0)

    if use_shared_experts:
        with torch.npu.stream(dispatcher.overlap_stream):
            self.mlp.shared_experts.post_forward_comm(wait_event=perm_prob_a2a_handle)

    detached_perm_a2a_out = detach_tensor(perm_a2a_out, checkpoint_forward=checkpoint)
    detached_perm_prob_a2a_out = detach_tensor(perm_prob_a2a_out, checkpoint_forward=checkpoint)
    # Token Perm2 Forward.
    perm_prob_a2a_handle.wait()
    dispached_input, dispached_input_probs = dispatcher.token_permute2(detached_perm_a2a_out, detached_perm_prob_a2a_out)
    perm_a2a_out.untyped_storage().resize_(0)

    # Grouped MLP Forward
    detached_dispached_input = detach_tensor(dispached_input, checkpoint_forward=checkpoint)
    detached_dispached_input_probs = detach_tensor(dispached_input_probs, checkpoint_forward=checkpoint)
    (expert_output, act_ckpt_manager), _ = self.mlp.experts(
        detached_dispached_input, tokens_per_expert, permuted_probs=detached_dispached_input_probs
    )

    if args.moe_zero_memory == 'level0':
        dispached_input.untyped_storage().resize_(0)
        recompute_needed_tensors = [dispached_input, probs, routing_map, dispatcher.num_global_tokens_per_local_expert_cpu]
    else:
        recompute_needed_tensors = [None, None, None, None]

    detached_expert_output = detach_tensor(expert_output, checkpoint_forward=checkpoint)

    # Token Unperm1 Forward
    unperm1_out = dispatcher.token_unpermute1(detached_expert_output, None)
    expert_output.untyped_storage().resize_(0)


    # Launch Token Unperm2 A2A
    unperm_a2a_out, unperm_a2a_handle = dispatcher.async_combine_comm(unperm1_out)
    unperm_a2a_handle.wait()
    # unperm1_out tensor storage is not need by backward,
    # but backward func of unperm1_out is needed, so resize the storage but keep tensor.
    unperm1_out.untyped_storage().resize_(0)
    detached_unperm_a2a_out = detach_tensor(unperm_a2a_out, checkpoint_forward=checkpoint)
    route_expert_output, unperm2_swap_manager = dispatcher.token_unpermute2(detached_unperm_a2a_out)
    unperm_a2a_out.untyped_storage().resize_(0)

    if use_shared_experts:
        with torch.npu.stream(dispatcher.overlap_stream):
            shared_expert_output, share_experts_graph = self.mlp.shared_experts.get_output()
        detached_shared_expert_output = detach_tensor(shared_expert_output, checkpoint_forward=checkpoint)
        torch.npu.current_stream().wait_stream(dispatcher.overlap_stream)
        mlp_output = route_expert_output + detached_shared_expert_output
        shared_expert_output.untyped_storage().resize_(0)
    else:
        detached_shared_expert_output = None
        share_experts_graph = None
        mlp_output = route_expert_output

    if recomp_norm and mlp_output.requires_grad:
        mlp_output.register_hook(self.norm_ckpt2.recompute)


    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            (mlp_output, None), residual2, self.hidden_dropout
        )

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    saved_tensors = (
        (attention_out, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (probs, probs_detached),
        ((perm1_out, perm1_probs), (None, None)),  # perm1 graph
        (None, (detached_perm_a2a_out, detached_perm_prob_a2a_out)),
        ((dispached_input, dispached_input_probs), (detached_dispached_input, detached_dispached_input_probs)),
        # perm2 graph
        (expert_output, detached_expert_output),  # grouped mlp graph
        (unperm1_out, None),  # unperm1 graph
        (None, detached_unperm_a2a_out),
        (output, None),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors, self, checkpointed=checkpoint
    )
    graph.act_ckpt_manager = act_ckpt_manager
    graph.unperm2_swap_manager = unperm2_swap_manager
    if hasattr(self.self_attention, 'swap_managers'):
        graph.attn_swap_managers = self.self_attention.swap_managers

    return output, context, graph


def transformer_layer_forward_dense(
    self,
    hidden_states,
    attention_mask=None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    checkpoint=False
):
    # hidden_states: [s, b, h]
    args = get_args()
    recomp_norm = getattr(args, 'recompute_norm', False)

    detached_layer_input = hidden_states

    # Residual connection.
    residual1 = detached_layer_input

    # input_layernorm + AttentionForward
    hidden_states = attention_forward(
        self, detached_layer_input, residual1,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
        recompute_norm=recomp_norm
    )

    attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states, checkpoint_forward=checkpoint)

    # Residual connection.
    residual2 = detached_attention_out

    if recomp_norm:
        self.norm_ckpt2 = CheckpointWithoutOutput()
        pre_mlp_layernorm_output = self.norm_ckpt2.checkpoint(self.pre_mlp_layernorm, False, detached_attention_out)
    else:
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(detached_attention_out)

    # MLP.
    detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)
    mlp_output_with_bias = self.mlp(detached_mlp_input)

    if recomp_norm:
        self.norm_ckpt2.discard_output()
        if mlp_output_with_bias[0].requires_grad:
            mlp_output_with_bias[0].register_hook(self.norm_ckpt2.recompute)


    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual2, self.hidden_dropout
        )

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    saved_tensors = (
        (attention_graph, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (output, None),
        (None, None),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, [], self, checkpointed=checkpoint
    )
    if hasattr(self.self_attention, 'swap_managers'):
        graph.attn_swap_managers = self.self_attention.swap_managers

    return output, context, graph


class IdentityOPFunc(torch.autograd.Function):
    @staticmethod
    def forward(cls, x):
        return x
    
    @staticmethod
    def backward(cls, grad):
        return grad
    

def transformer_layer_forward_noop(
    self,
    hidden_states,
    attention_mask=None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    checkpoint=False
):
    detached_layer_input = hidden_states
    output = IdentityOPFunc.apply(detached_layer_input)

    return output, context, NoopLayerGraph(detached_layer_input, output, self, checkpointed=checkpoint)
