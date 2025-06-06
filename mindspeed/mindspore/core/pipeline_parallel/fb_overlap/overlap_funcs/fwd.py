#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather, async_reduce_scatter
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS
from mindspeed.model.transformer import should_recompute_activation

from mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher import (
    alltoall_token_perm1, alltoall_token_perm2,
    alltoall_token_unperm1, alltoall_token_unperm2, overlap_stream
)
from mindspeed.core.pipeline_parallel.fb_overlap.modules.attention import attention_forward

from ..modules.utils import (
    detach_tensor, run_graph_forward,
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
    def attention_forward_func(detached_layer_input, residual1):
        hidden_states = attention_forward(
            self, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )
        return hidden_states
    hidden_states, attention_forward_vjp = run_graph_forward(attention_forward_func, detached_layer_input, residual1)


    attention_out, detached_attention_out = hidden_states, detach_tensor(hidden_states, checkpoint_forward=checkpoint)

    # Residual connection.
    residual2 = detached_attention_out

    # Layer Norm after attention
    if recomp_norm:
        self.norm_ckpt2 = CheckpointWithoutOutput()

        def pre_mlp_forward(detached_attention_out):
            pre_mlp_layernorm_output = self.norm_ckpt2.checkpoint(self.pre_mlp_layernorm, False, detached_attention_out)
            return pre_mlp_layernorm_output
        pre_mlp_layernorm_output, pre_mlp_vjp = run_graph_forward(pre_mlp_forward, detached_attention_out)
    else:
        pre_mlp_layernorm_output, pre_mlp_vjp = run_graph_forward(self.pre_mlp_layernorm, detached_attention_out)

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
    indices = None

    def router_function(detached_mlp_input):
        nonlocal indices
        probs, indices = router_forward(self, detached_mlp_input)
        return probs
    probs, router_forward_vjp = run_graph_forward(router_function, detached_mlp_input) # @check indices grad(self, detached_mlp_input)
    shared_expert_output = None

    # Token Perm1 Forward
    probs_detached = detach_tensor(probs, checkpoint_forward=checkpoint)
    tokens_per_expert = None

    def alltoall_token_perm1_func(detached_mlp_input, probs_detached):
        nonlocal tokens_per_expert
        if args.moe_zerc:
            perm1_out, perm1_probs, tokens_per_expert, global_map_info = alltoall_token_perm1(self.mlp.token_dispatcher,
                                                                                              detached_mlp_input,
                                                                                              probs_detached, indices)
            return perm1_out, perm1_probs, global_map_info
        else:
            perm1_out, perm1_probs, tokens_per_expert = alltoall_token_perm1(self.mlp.token_dispatcher,
                                                                             detached_mlp_input,
                                                                             probs_detached, indices)
            return perm1_out, perm1_probs
    if args.moe_zerc:
        (perm1_out, perm1_probs, global_map_info), perm1_vjp = run_graph_forward(alltoall_token_perm1_func,
                                                                                 detached_mlp_input,
                                                                                 probs_detached)
    else:
        (perm1_out, perm1_probs), perm1_vjp = run_graph_forward(alltoall_token_perm1_func, detached_mlp_input,
                                                                probs_detached)

    shared_experts_vjp = None
    if not args.moe_zerc:
        if use_shared_experts:
            if shared_experts_allgather_handle is not None:
                shared_experts_allgather_handle.wait()
                shared_experts_allgather_handle = None
            # Shared Experts Forward.
            (shared_expert_output, _), shared_experts_vjp = run_graph_forward(self.mlp.shared_experts,
                                                                                detached_mlp_input)  # @check bias; cell as arg
        disp = self.mlp.token_dispatcher
        if disp.num_local_experts > 1:
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            disp.cuda_sync_point = "no_sync"
            disp.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                disp.expert_ids_per_ep_rank, disp.num_global_tokens_per_local_expert.ravel()
            )
        torch.npu.current_stream().wait_stream(overlap_stream.stream)

    # Async Perm A2A.
    _, perm_a2a_out, perm_a2a_handle = async_all_to_all(
        perm1_out,
        self.mlp.token_dispatcher.output_splits,
        self.mlp.token_dispatcher.input_splits,
        ep_group
    )
    perm_prob_a2a_out, perm_prob_a2a_handle = None, None
    if args.moe_unperm2_mem_optim and perm1_probs is not None:
        _, perm_prob_a2a_out, perm_prob_a2a_handle = async_all_to_all(
            perm1_probs,
            self.mlp.token_dispatcher.output_splits,
            self.mlp.token_dispatcher.input_splits,
            ep_group,
            event=perm_a2a_handle,
            stream=torch.npu.current_stream()
        )

    if args.moe_zerc:
        if shared_experts_allgather_handle is not None:
            shared_experts_allgather_handle.wait()
            shared_experts_allgather_handle = None
            # Shared Experts Forward.
            (shared_expert_output, _), shared_experts_vjp = run_graph_forward(self.mlp.shared_experts,
                                                                              detached_mlp_input)  # @check bias; cell as arg

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
            shared_expert_output, tp_group, 
            event=perm_prob_a2a_handle if perm_prob_a2a_handle else None
        )
    else:
        share_experts_graph = shared_expert_output
        rs_shared_experts_handle = None

    detached_perm_a2a_out = detach_tensor(perm_a2a_out, checkpoint_forward=checkpoint)
    detached_perm_prob_a2a_out = detach_tensor(perm_prob_a2a_out, checkpoint_forward=checkpoint)
    # Token Perm2 Forward.
    if args.moe_unperm2_mem_optim and perm_prob_a2a_handle:
        perm_prob_a2a_handle.wait()

    def alltoall_token_perm2_func(detached_perm_a2a_out, detached_perm_prob_a2a_out):
        return alltoall_token_perm2(self.mlp.token_dispatcher, detached_perm_a2a_out, detached_perm_prob_a2a_out)
    (dispached_input, dispached_input_probs), perm2_vjp = run_graph_forward(alltoall_token_perm2_func, detached_perm_a2a_out, detached_perm_prob_a2a_out)
    perm_a2a_out.untyped_storage().resize_(0)

    # Grouped MLP Forward
    detached_dispached_input = detach_tensor(dispached_input, checkpoint_forward=checkpoint)
    detached_dispached_input_probs = detach_tensor(dispached_input_probs, checkpoint_forward=checkpoint)
    recompute_needed_tensors = []
    act_ckpt_manager = None

    def mlp_experts(detached_dispached_input, detached_dispached_input_probs):
        nonlocal act_ckpt_manager
        (expert_output, act_ckpt_manager), _ = self.mlp.experts(detached_dispached_input, tokens_per_expert, permuted_probs=detached_dispached_input_probs)
        return expert_output

    expert_output, grouped_mlp_vjp = run_graph_forward(mlp_experts, detached_dispached_input, detached_dispached_input_probs)

    if args.moe_zero_memory == 'level0':
        dispached_input.untyped_storage().resize_(0)
        if args.moe_zerc:
            recompute_needed_tensors = [dispached_input, probs, indices,
                                        self.mlp.token_dispatcher.select_index, self.mlp.token_dispatcher.nr_token_id_recover]
        else:
            recompute_needed_tensors = [dispached_input, probs, indices, self.mlp.token_dispatcher.global_input_tokens_local_experts_indices]
    else:
        if args.moe_zerc:
            recompute_needed_tensors = [None, None, None, None, None]
        else:
            recompute_needed_tensors = [None, None, None, None]



    detached_expert_output = detach_tensor(expert_output, checkpoint_forward=checkpoint)

    # Token Unperm1 Forward
    def alltoall_token_unperm1_func(detached_expert_output, global_map_info=None):
        if args.moe_zerc:
            unperm1_out = alltoall_token_unperm1(self.mlp.token_dispatcher, detached_expert_output, None,
                                                 global_map_info)
        else:
            unperm1_out = alltoall_token_unperm1(self.mlp.token_dispatcher, detached_expert_output, None)
        return unperm1_out

    if not args.moe_zerc:
        global_map_info = None
    unperm1_out, unperm1_vjp = run_graph_forward(alltoall_token_unperm1_func, detached_expert_output, global_map_info)
    if not args.moe_zerc or args.moe_unperm2_mem_optim:
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

    if use_shared_experts:
        detached_shared_expert_output = detach_tensor(shared_expert_output, checkpoint_forward=checkpoint)
    else:
        detached_shared_expert_output = None
        share_experts_graph = None

    unperm2_swap_manager = None

    def alltoall_token_unperm2_func(detached_unperm_a2a_out, detached_shared_expert_output, residual2, probs): # @check input shared_expert might be none
        nonlocal unperm2_swap_manager
        if args.moe_unperm2_mem_optim:
            probs = None
        route_expert_output, unperm2_swap_manager = alltoall_token_unperm2(self.mlp.token_dispatcher, detached_unperm_a2a_out, probs)

        if use_shared_experts:
            mlp_output = route_expert_output + detached_shared_expert_output
            shared_expert_output.untyped_storage().resize_(0)
        else:
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

        return output



    output, unperm2_vjp = run_graph_forward(alltoall_token_unperm2_func,
        detached_unperm_a2a_out, detached_shared_expert_output, residual2, probs_detached)


    saved_tensors = (
        (attention_out, detached_attention_out, attention_forward_vjp),
        (pre_mlp_layernorm_output, detached_mlp_input, pre_mlp_vjp),
        (probs, probs_detached, router_forward_vjp),
        ((perm1_out, perm1_probs), (None, None), perm1_vjp),  # perm1 graph
        (None, (detached_perm_a2a_out, detached_perm_prob_a2a_out), None),
        ((dispached_input, dispached_input_probs), (detached_dispached_input, detached_dispached_input_probs), perm2_vjp),
        # perm2 graph
        (expert_output, detached_expert_output, grouped_mlp_vjp),  # grouped mlp graph
        (unperm1_out, None, unperm1_vjp),  # unperm1 graph
        (None, detached_unperm_a2a_out, None), # unperm a2a graph
        (output, None, unperm2_vjp),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output, shared_experts_vjp),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors,
        self.mlp.token_dispatcher.input_splits, self.mlp.token_dispatcher.output_splits, self,
        checkpointed=checkpoint
    )
    graph.act_ckpt_manager = act_ckpt_manager
    graph.unperm2_swap_manager = unperm2_swap_manager
    if hasattr(self.self_attention, 'swap_managers'):
        graph.attn_swap_managers = self.self_attention.swap_managers

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

    # ms forward
    def attention_func(detached_layer_input, residual1):
        hidden_states = attention_forward(
            self, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )
        return hidden_states

    hidden_states, attention_forward_vjp = run_graph_forward(attention_func, detached_layer_input, residual1)


    attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states, checkpoint_forward=checkpoint)

    # Residual connection.
    residual2 = detached_attention_out
    
    if recomp_norm:
        self.norm_ckpt2 = CheckpointWithoutOutput()

        def pre_mlp_layernorm_func(detached_attention_out):
            pre_mlp_layernorm_output = self.norm_ckpt2.checkpoint(self.pre_mlp_layernorm, False, detached_attention_out)
            return pre_mlp_layernorm_output
    else:
        def pre_mlp_layernorm_func(detached_attention_out):

            pre_mlp_layernorm_output = self.pre_mlp_layernorm(detached_attention_out)
            return pre_mlp_layernorm_output

    pre_mlp_layernorm_output, pre_mlp_layernorm_forward_vjp = run_graph_forward(pre_mlp_layernorm_func, detached_attention_out)


    # MLP.
    detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)

    def func_mlp(detached_mlp_input, residual2):
        mlp_output_with_bias = self.mlp(detached_mlp_input)

        if recomp_norm:
            self.norm_ckpt2.discard_output()
            # if mlp_output_with_bias[0].requires_grad:
            mlp_output_with_bias[0].register_hook(self.norm_ckpt2.recompute)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual2, self.hidden_dropout
            )
        return hidden_states

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.

    output, output_vjp = run_graph_forward(func_mlp, detached_mlp_input, residual2)


    saved_tensors = (
        (attention_graph, detached_attention_out, attention_forward_vjp),
        (pre_mlp_layernorm_output, detached_mlp_input, pre_mlp_layernorm_forward_vjp),
        (None, None, None),
        (None, None, None),
        (None, None, None),
        (None, None, None),
        (None, None, None),
        (None, None, None),
        (None, None, None),
        (output, None, output_vjp),
        (None, None, None),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, [], None, None, self,
        checkpointed=checkpoint
    )
    if hasattr(self.self_attention, 'swap_managers'):
        graph.attn_swap_managers = self.self_attention.swap_managers

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