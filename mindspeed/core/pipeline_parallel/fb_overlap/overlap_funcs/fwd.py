#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather, async_reduce_scatter
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS
from mindspeed.model.transformer import should_recompute_activation
from ..modules.token_dispatcher import (
    alltoall_token_perm1, alltoall_token_perm2,
    alltoall_token_unperm1, alltoall_token_unperm2
)
from ..modules.attention import attention_forward
from ..modules.utils import (
    detach_tensor,
    NoopLayerGraph, LayerGraph,
)


def router_forward(
    self,
    hidden_states
):
    probs, indices = self.mlp.router(hidden_states)

    return probs, indices


def transformer_layer_forward_moe(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
    checkpoint=False
):
    # hidden_states: [s, b, h]
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    if args.moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    use_shared_experts = hasattr(self.mlp, 'shared_experts') and self.mlp.shared_experts is not None
    recomp_norm = getattr(args, 'recompute_norm', False)

    detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

    # Residual connection.
    residual1 = detached_layer_input

    # input_layernorm + AttentionForward
    hidden_states = attention_forward(
        self, detached_layer_input, residual1,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
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
    detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)
    if tp_size > 1 and use_shared_experts:
        # shared experts tp communication
        _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
            detached_mlp_input, tp_group, is_use_get_global_memory_buffer=True
        )
        AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
    else:
        shared_experts_input, shared_experts_allgather_handle = detached_mlp_input, None

    # Router forward.
    probs, indices = router_forward(self, detached_mlp_input)
    shared_expert_output = None

    # Token Perm1 Forward
    probs_detached = detach_tensor(probs, checkpoint_forward=checkpoint)
    perm1_out, tokens_per_expert = alltoall_token_perm1(self.mlp.token_dispatcher, detached_mlp_input, probs_detached, indices)
    if shared_experts_allgather_handle is not None:
        # overlap shared experts tp comm by token perm1.
        shared_experts_allgather_handle.wait()
    # Async Perm A2A.
    _, perm_a2a_out, perm_a2a_handle = async_all_to_all(
        perm1_out,
        self.mlp.token_dispatcher.output_splits,
        self.mlp.token_dispatcher.input_splits,
        ep_group
    )
    # Shared Experts Forward.
    if use_shared_experts:
        shared_expert_output, _ = self.mlp.shared_experts(detached_mlp_input)
    if recomp_norm:
        self.norm_ckpt2.discard_output()
    # overlap perm a2a by shared experts computation.
    perm_a2a_handle.wait()
    # perm1_out tensor storage is not need by backward,
    # but backward func of perm1_out is needed, so resize the storage but keep tensor.
    perm1_out.untyped_storage().resize_(0)
    if tp_size > 1 and use_shared_experts:
        # tp comm for shared experts
        share_experts_graph, shared_expert_output, rs_shared_experts_handle = async_reduce_scatter(
            shared_expert_output, tp_group
        )
    else:
        share_experts_graph = shared_expert_output
        rs_shared_experts_handle = None

    detached_perm_a2a_out = detach_tensor(perm_a2a_out, checkpoint_forward=checkpoint)
    # Token Perm2 Forward.
    dispached_input = alltoall_token_perm2(self.mlp.token_dispatcher, detached_perm_a2a_out)
    perm_a2a_out.untyped_storage().resize_(0)

    # Grouped MLP Forward
    detached_dispached_input = detach_tensor(dispached_input, checkpoint_forward=checkpoint)
    (expert_output, fc1_output, act_out), _ = self.mlp.experts(detached_dispached_input, tokens_per_expert)
    if args.moe_zero_memory == 'level0':
        dispached_input.untyped_storage().resize_(0)
        recompute_needed_tensors = [dispached_input, fc1_output, act_out, probs, indices,
                                    self.mlp.token_dispatcher.global_input_tokens_local_experts_indices]
    else:
        if should_recompute_activation(self.layer_number):
            recompute_needed_tensors = [None, fc1_output, act_out, None, None, None]
        else:
            recompute_needed_tensors = [None, None, None, None, None, None]

    detached_expert_output = detach_tensor(expert_output, checkpoint_forward=checkpoint)

    # Token Unperm1 Forward
    unperm1_out = alltoall_token_unperm1(self.mlp.token_dispatcher, detached_expert_output, None)
    expert_output.untyped_storage().resize_(0)
    if rs_shared_experts_handle is not None:
        # overlap shared experts tp comm by token perm2 + gmm
        rs_shared_experts_handle.wait()
        # share_experts_graph tensor storage is not need by backward,
        # but backward func of share_experts_graph is needed, so resize the storage but keep tensor.
        share_experts_graph.untyped_storage().resize_(0)

    # Launch Token Unperm2 A2A
    _, unperm_a2a_out, unperm_a2a_handle = async_all_to_all(
        unperm1_out,
        self.mlp.token_dispatcher.input_splits,
        self.mlp.token_dispatcher.output_splits,
        ep_group
    )
    unperm_a2a_handle.wait()
    # unperm1_out tensor storage is not need by backward,
    # but backward func of unperm1_out is needed, so resize the storage but keep tensor.
    unperm1_out.untyped_storage().resize_(0)
    detached_unperm_a2a_out = detach_tensor(unperm_a2a_out, checkpoint_forward=checkpoint)
    route_expert_output, _ = alltoall_token_unperm2(self.mlp.token_dispatcher, detached_unperm_a2a_out)

    if use_shared_experts:
        detached_shared_expert_output = detach_tensor(shared_expert_output, checkpoint_forward=checkpoint)
        mlp_output = route_expert_output + detached_shared_expert_output
        shared_expert_output.untyped_storage().resize_(0)
    else:
        detached_shared_expert_output = None
        share_experts_graph = None
        mlp_output = route_expert_output

    if recomp_norm:
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
        (perm1_out, None),  # perm1 graph
        (None, detached_perm_a2a_out),
        (dispached_input, detached_dispached_input),  # perm2 graph
        (expert_output, detached_expert_output),  # grouped mlp graph
        (unperm1_out, None),  # unperm1 graph
        (None, detached_unperm_a2a_out),
        (output, None),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors,
        self.mlp.token_dispatcher.input_splits, self.mlp.token_dispatcher.output_splits, self,
        checkpointed=checkpoint
    )

    return output, context, graph


def transformer_layer_forward_dense(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
    checkpoint=False
):
    # hidden_states: [s, b, h]
    args = get_args()
    recomp_norm = getattr(args, 'recompute_norm', False)

    detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

    # Residual connection.
    residual1 = detached_layer_input

    # input_layernorm + AttentionForward
    hidden_states = attention_forward(
        self, detached_layer_input, residual1,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
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
        saved_tensors, [], None, None, self,
        checkpointed=checkpoint
    )

    return output, context, graph


def transformer_layer_forward_noop(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
    checkpoint=False
):
    detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)
    output = detached_layer_input.clone()

    return output, context, NoopLayerGraph(detached_layer_input, output, self, checkpointed=checkpoint)