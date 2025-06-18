#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from contextlib import nullcontext
import torch
import acl
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import permute
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather, async_reduce_scatter
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS

from mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher import (
    alltoall_token_perm1, alltoall_token_perm2,
    alltoall_token_unperm1, alltoall_token_unperm2, overlap_stream
)

from mindspeed.core.pipeline_parallel.fb_overlap.modules.weight_grad_store import WeightGradStore
from mindspeed.core.pipeline_parallel.fb_overlap.modules.attention import (
    attention_forward, set_async_alltoall_inputs, get_async_alltoall_outputs
)
from mindspeed.core.pipeline_parallel.fb_overlap.modules.utils import (LayerGraph, is_p2p_comm_needed,
    p2p_comm_helper, P2PCommOutput, P2PCommParams
)
from ..modules.utils import (
    detach_tensor, run_graph_forward, run_graph_backward, LayerGraph
)


def router_forward(
    self,
    hidden_states
):
    probs, indices = self.mlp.router(hidden_states)

    return probs, indices


def transformer_layer_forward_dense_backward_moe_overlaping(
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
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    if args.moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    recomp_norm = getattr(args, 'recompute_norm', False)
    if args.moe_zerc:
        bwd_dispached_input, bwd_probs, bwd_indices, bwd_select_index, bwd_nr_token_id_recover = bwd_layer_graph.recompute_needed_tensors
    else:
        bwd_dispached_input, bwd_probs, bwd_indices, global_input_tokens_local_experts_indices = bwd_layer_graph.recompute_needed_tensors
        bwd_select_index = None

    # Launch swap-in at the beginning of the backward pass.
    if bwd_layer_graph.unperm2_swap_manager:
        bwd_layer_graph.unperm2_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    # Unperm2 Bwd
    # check if backward unpermutation alltoall is launched at bwd layer before
    if bwd_unperm_a2a_handle is None:
        unperm_a2a_graph_grad, detached_shared_expert_output_grad, residual2_grad, probs_grad = \
        run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)
        # Async Unperm A2A
        _, unperm1_out_grad, bwd_unperm_a2a_handle = async_all_to_all(
            unperm_a2a_graph_grad,
            bwd_layer_graph.output_splits,
            bwd_layer_graph.input_splits,
            ep_group
        )
    else:
        unperm1_out_grad, detached_shared_expert_output_grad, residual2_grad, probs_grad = bwd_layer_output_grad

    if args.moe_zero_memory == 'level0':
        with torch.no_grad():
            bwd_input_before_perm1 = bwd_layer_graph.pre_mlp_layernorm_graph[0]

            def recomp_token_permutation1(hidden_states, indices, _select_index):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                if args.moe_zerc:
                    permutated_local_input_tokens = hidden_states.index_select(0, _select_index)
                else:
                    permutated_local_input_tokens, _ = permute(
                        hidden_states, indices
                    )
                return permutated_local_input_tokens
            bwd_perm1_out = recomp_token_permutation1(bwd_input_before_perm1, bwd_indices, bwd_select_index)

    with checkpoint_context:
        # Atten Fwd
        detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        def attention_forward_func(detached_layer_input, residual1):
            hidden_states = attention_forward(
                fwd_layer, detached_layer_input, residual1,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                recompute_norm=recomp_norm
            )
            return hidden_states
        hidden_states, attention_forward_vjp = run_graph_forward(attention_forward_func, detached_layer_input, residual1)


        attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states,
                                                                               checkpoint_forward=checkpoint)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            fwd_layer.norm_ckpt2 = CheckpointWithoutOutput()

            def pre_mlp_forward(detached_attention_out):
                pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False, detached_attention_out)
                return pre_mlp_layernorm_output
            pre_mlp_layernorm_output, pre_mlp_vjp = run_graph_forward(pre_mlp_forward, detached_attention_out)
        else:
            pre_mlp_layernorm_output, pre_mlp_vjp = run_graph_forward(fwd_layer.pre_mlp_layernorm, detached_attention_out)


    if args.moe_zero_memory == 'level0':
        _, bwd_perm_a2a_out, bwd_recomp_perm_a2a_handle = async_all_to_all(
            bwd_perm1_out,
            bwd_layer_graph.output_splits,
            bwd_layer_graph.input_splits,
            ep_group,
            event=bwd_unperm_a2a_handle,
            stream=torch.npu.current_stream()
        )

    if args.moe_zero_memory == 'level0' or should_recompute_activation(bwd_layer_graph.layer.layer_number):
        bwd_layer_graph.act_ckpt_manager.recompute(True)


    bwd_unperm_a2a_handle.wait()
    bwd_unperm_a2a_handle = None
    # when open moe-unperm2-mem-optim, the global_map_info_grad here will be zero
    (detached_expert_output_grad, global_map_info_grad) = run_graph_backward(bwd_layer_graph.unperm1_graph,
                                                                             unperm1_out_grad)
    unperm1_out_grad.untyped_storage().resize_(0)
    WeightGradStore.start_decouple()
    detached_dispached_input_grad, detached_dispached_input_probs_grad = run_graph_backward(bwd_layer_graph.grouped_mlp_graph, detached_expert_output_grad,
                                                        keep_grad=True)  # keep for dw
    WeightGradStore.end_decouple()
    detached_perm_a2a_out_grad, perm_prob_a2a_out_grad, perm2_global_map_info_grad = run_graph_backward(bwd_layer_graph.perm2_graph, (detached_dispached_input_grad, detached_dispached_input_probs_grad),
                                                       keep_graph=True)  # keep for dw
    if perm2_global_map_info_grad is not None:
        global_map_info_grad = global_map_info_grad + perm2_global_map_info_grad

    if args.moe_zero_memory == 'level0':
        with torch.no_grad():
            bwd_recomp_perm_a2a_handle.wait()
            bwd_recomp_perm_a2a_handle = None
            if args.moe_zerc:
                recompute_fc1_input = bwd_perm_a2a_out.index_select(0, bwd_nr_token_id_recover)
            else:
                recompute_fc1_input, _ = permute(bwd_perm_a2a_out, global_input_tokens_local_experts_indices)
            bwd_perm_a2a_out.untyped_storage().resize_(0)

    if tp_size > 1:
        shared_expert_grad = detached_shared_expert_output_grad
        _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
            shared_expert_grad, parallel_state.get_tensor_model_parallel_group()
        )
    else:
        backward_ag_shared = detached_shared_expert_output_grad
        backward_ag_shared_handle = None

    _, perm1_out_grad, bwd_perm_a2a_handle = async_all_to_all(
        detached_perm_a2a_out_grad,
        bwd_layer_graph.input_splits,
        bwd_layer_graph.output_splits,
        ep_group,
        event=backward_ag_shared_handle
    )
    perm1_prob_out_grad, bwd_prob_handle = None, None
    if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name() and not args.moe_zerc:
        _, perm1_prob_out_grad, bwd_prob_handle = async_all_to_all(
            perm_prob_a2a_out_grad,
            bwd_layer_graph.input_splits,
            bwd_layer_graph.output_splits,
            ep_group,
            event=bwd_perm_a2a_handle,
            stream=torch.npu.current_stream()
        )

    # Grouped MLP dw computation

    with checkpoint_context:
        # MLP Forward
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)

        def mlp_function(detached_mlp_input, residual2):
            mlp_output_with_bias = fwd_layer.mlp(detached_mlp_input)
            if recomp_norm:
                fwd_layer.norm_ckpt2.discard_output()
                mlp_output_with_bias[0].register_hook(fwd_layer.norm_ckpt2.recompute)

            with fwd_layer.bias_dropout_add_exec_handler():
                hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual2, fwd_layer.hidden_dropout)
            return hidden_states

        hidden_states, mlp_vjp = run_graph_forward(mlp_function, detached_mlp_input, residual2)

    bwd_perm_a2a_handle.wait()
    bwd_perm_a2a_handle = None
    if bwd_prob_handle:
        bwd_prob_handle.wait()
    if args.moe_zerc:
        detached_mlp_input_grad, probs_detached_grad = run_graph_backward(bwd_layer_graph.perm1_graph, (
        perm1_out_grad, perm1_prob_out_grad, global_map_info_grad))
    else:
        detached_mlp_input_grad, probs_detached_grad = run_graph_backward(bwd_layer_graph.perm1_graph,
                                                        (perm1_out_grad, perm1_prob_out_grad))

    WeightGradStore.start_decouple()
    if backward_ag_shared_handle is not None:
        backward_ag_shared_handle.wait()
        backward_ag_shared_handle = None
    (detached_mlp_input_shared_grad, ) = run_graph_backward(bwd_layer_graph.shared_experts_graph, backward_ag_shared, keep_grad=True)  # dw computation
    WeightGradStore.end_decouple()


    # swap-in unperm2 input for probs_grad computation
    if bwd_layer_graph.unperm2_swap_manager:
        bwd_layer_graph.unperm2_swap_manager.wait_swap_in()

    if args.moe_unperm2_mem_optim_swap:
        # dprobs computation
        output_grad = bwd_layer_output_grad
        if hasattr(bwd_layer_graph, 'last_layer_input_grad'):
            output_grad = bwd_layer_graph.last_layer_input_grad
        H = bwd_layer_graph.unperm2_swap_manager.npu_tensor.shape[-1]
        K = args.moe_router_topk
        probs_dtype = bwd_probs.dtype
        probs_grad = bwd_layer_graph.unperm2_swap_manager.npu_tensor.reshape(-1, K, H).to(probs_dtype) * output_grad.to(probs_dtype)
        output_grad.untyped_storage().resize_(0)
        bwd_layer_graph.unperm2_swap_manager.npu_tensor.untyped_storage().resize_(0)
        probs_grad = probs_grad.sum(dim=-1)

    if args.moe_unperm2_mem_optim:
        probs_grad = probs_detached_grad

    if args.moe_zerc:
        probs_grad = probs_detached_grad
    (detached_mlp_input_router_grad, ) = run_graph_backward(bwd_layer_graph.router_graph, probs_grad)
    detached_mlp_input_grad = detached_mlp_input_grad + detached_mlp_input_shared_grad + detached_mlp_input_router_grad

    (detached_attention_out_grad, ) = run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, detached_mlp_input_grad, keep_graph=True)
    WeightGradStore.start_decouple()
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.wait_swap_in()
    (layer_input_grad, _) = run_graph_backward(bwd_layer_graph.attn_graph, detached_attention_out_grad + residual2_grad, keep_grad=True)
    WeightGradStore.end_decouple()


    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        (unperm_a2a_graph_grad, detached_shared_expert_output_grad, residual2_grad, probs_grad) = run_graph_backward(next_bwd_layer_graph.unperm2_graph, layer_input_grad, keep_graph=True, keep_grad=True)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = layer_input_grad, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            unperm_a2a_graph_grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )
        next_layer_output_grad = (next_layer_output_grad, detached_shared_expert_output_grad, residual2_grad, probs_grad)


    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, layer_input_grad)

    if args.moe_zero_memory == 'level0':
        # restore fc1 input for dw computation
        bwd_dispached_input.untyped_storage().resize_(recompute_fc1_input.untyped_storage().size())
        bwd_dispached_input.untyped_storage().copy_(recompute_fc1_input.untyped_storage(), False)
        recompute_fc1_input.untyped_storage().resize_(0)
    WeightGradStore.pop()

    saved_tensors = (
        (attention_graph, detached_attention_out, attention_forward_vjp),
        (pre_mlp_layernorm_output, detached_mlp_input, pre_mlp_vjp),
        (None, None),
        (None, None),
        (None, None),
        (None, None),  # perm2 graph
        (None, None),  # grouped mlp graph
        (None, None),  # unperm1 graph
        (None, None),
        (output, None, mlp_vjp),  # unperm2 graph
        (None, None),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, [], None, None, fwd_layer,
        checkpointed=checkpoint
    )
    if hasattr(fwd_layer.self_attention, 'swap_managers'):
        graph.attn_swap_managers = fwd_layer.self_attention.swap_managers

    # save original layer output for probs_grad computation
    if args.moe_unperm2_mem_optim_swap \
        and next_bwd_layer_graph is not None \
        and getattr(next_bwd_layer_graph, 'is_moe_layer', False):

        next_bwd_layer_graph.last_layer_input_grad = layer_input_grad

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)

    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, layer_input_grad))



def transformer_layer_forward_moe_backward_dense_overlaping(
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
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    args = get_args()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    use_shared_experts = hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    if args.moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    recomp_norm = getattr(args, 'recompute_norm', False)
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    with checkpoint_context:
        # Atten Fwd
        detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        def attention_forward_func(detached_layer_input, residual1):
            hidden_states = attention_forward(
                fwd_layer, detached_layer_input, residual1,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                recompute_norm=recomp_norm
            )
            return hidden_states
        hidden_states, attention_forward_vjp = run_graph_forward(attention_forward_func, detached_layer_input, residual1)


        attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            fwd_layer.norm_ckpt2 = CheckpointWithoutOutput()

            def pre_mlp_forward(detached_attention_out):
                pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False, detached_attention_out)
                return pre_mlp_layernorm_output
            pre_mlp_layernorm_output, pre_mlp_vjp = run_graph_forward(pre_mlp_forward, detached_attention_out)
        else:
            pre_mlp_layernorm_output, pre_mlp_vjp = run_graph_forward(fwd_layer.pre_mlp_layernorm, detached_attention_out)

        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)

        indices = None

        def router_func(detached_mlp_input):
            nonlocal indices
            probs, indices = router_forward(fwd_layer, detached_mlp_input)
            return probs

        probs, router_forward_vjp = run_graph_forward(router_func, detached_mlp_input)

        if tp_size > 1 and use_shared_experts:
            _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                detached_mlp_input, tp_group, is_use_get_global_memory_buffer=True
            )
            AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            shared_experts_input, shared_experts_allgather_handle = detached_mlp_input, None

        # Token Permutation Forward
        probs_detached = detach_tensor(probs, checkpoint_forward=checkpoint)
        perm1_out, perm1_probs, tokens_per_expert = alltoall_token_perm1(fwd_layer.mlp.token_dispatcher, detached_mlp_input, probs_detached, indices)
        tokens_per_expert = None

        def alltoall_token_perm1_func(detached_mlp_input, probs_detached):
            nonlocal tokens_per_expert
            perm1_out, perm1_probs, tokens_per_expert = alltoall_token_perm1(fwd_layer.mlp.token_dispatcher, detached_mlp_input, probs_detached, indices)
            return perm1_out, perm1_probs
        (perm1_out, perm1_probs), perm1_vjp = run_graph_forward(alltoall_token_perm1_func, detached_mlp_input, probs_detached) # @check tokens_per_expert grad

        shared_experts_vjp = None
        if use_shared_experts:
            if shared_experts_allgather_handle is not None:
                shared_experts_allgather_handle.wait()
                shared_experts_allgather_handle = None

            def mlp_shared_expert_func(detached_mlp_input):
                shared_expert_output, _ = fwd_layer.mlp.shared_experts(detached_mlp_input)
                return shared_expert_output
            shared_expert_output, shared_experts_vjp = run_graph_forward(mlp_shared_expert_func, detached_mlp_input)
        
        disp = fwd_layer.mlp.token_dispatcher
        if disp.num_local_experts > 1:
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            disp.cuda_sync_point = "no_sync"
            disp.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                disp.expert_ids_per_ep_rank, disp.num_global_tokens_per_local_expert.ravel()
            )

        torch.npu.current_stream().wait_stream(overlap_stream.stream)

        _, perm_a2a_out, perm_a2a_handle = async_all_to_all(
            perm1_out,
            fwd_layer.mlp.token_dispatcher.output_splits,
            fwd_layer.mlp.token_dispatcher.input_splits,
            ep_group
        )
        perm_prob_a2a_out, perm_prob_a2a_handle = None, None
        if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name():
            _, perm_prob_a2a_out, perm_prob_a2a_handle = async_all_to_all(
                perm1_probs,
                fwd_layer.mlp.token_dispatcher.output_splits,
                fwd_layer.mlp.token_dispatcher.input_splits,
                ep_group,
                event=perm_a2a_handle,
                stream=torch.npu.current_stream()
            )

    WeightGradStore.start_decouple()
    # bwd for mlp

    detached_mlp_input_grad, residual2_grad = run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)  # keep for dw
    (detached_attention_out_grad, ) = run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, detached_mlp_input_grad, keep_graph=True)
    detached_attention_out_grad = detached_attention_out_grad + residual2_grad
    WeightGradStore.end_decouple()

    perm_a2a_handle.wait()
    perm_a2a_handle = None
    perm1_out.untyped_storage().resize_(0)

    # Grouped MLP dw computation

    with checkpoint_context:
        # Token Perm2 forward
        detached_perm_a2a_out = detach_tensor(perm_a2a_out, checkpoint_forward=checkpoint)
        detached_perm_prob_a2a_out = detach_tensor(perm_prob_a2a_out, checkpoint_forward=checkpoint)
        if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name():
            perm_prob_a2a_handle.wait()

        # global_map_info is not used here, it's used in moe_zerc's forward_moe_backward_dense function
        global_map_info = None

        def alltoall_token_perm2_func(detached_perm_a2a_out, detached_perm_prob_a2a_out, global_map_info=None):
            return alltoall_token_perm2(fwd_layer.mlp.token_dispatcher, detached_perm_a2a_out, detached_perm_prob_a2a_out)

        (dispached_input, dispached_input_probs), perm2_vjp = run_graph_forward(alltoall_token_perm2_func, detached_perm_a2a_out, detached_perm_prob_a2a_out, global_map_info)
        perm_a2a_out.untyped_storage().resize_(0)


        # Grouped MLP Forward
        # to check
        recompute_needed_tensors = []
        detached_dispached_input = detach_tensor(dispached_input, checkpoint_forward=checkpoint)
        detached_dispached_input_probs = detach_tensor(dispached_input_probs, checkpoint_forward=checkpoint)
        act_ckpt_manager = None

        def mlp_experts(detached_dispached_input, detached_dispached_input_probs):
            nonlocal act_ckpt_manager
            (expert_output, act_ckpt_manager), _ = fwd_layer.mlp.experts(detached_dispached_input, tokens_per_expert, permuted_probs=detached_dispached_input_probs)
            return expert_output
        expert_output, mlp_experts_vjp = run_graph_forward(mlp_experts, detached_dispached_input, detached_dispached_input_probs)

        if args.moe_zero_memory == 'level0':
            dispached_input.untyped_storage().resize_(0)
            recompute_needed_tensors = [dispached_input, probs, indices,
                                        fwd_layer.mlp.token_dispatcher.global_input_tokens_local_experts_indices]
        else:
            recompute_needed_tensors = [None, None, None, None]

        # Token Unpermutaion Forward
        def alltoall_token_unperm1_func(detached_expert_output, global_map_info=None):
            unperm1_out = alltoall_token_unperm1(fwd_layer.mlp.token_dispatcher, detached_expert_output, None)
            return unperm1_out

        detached_expert_output = detach_tensor(expert_output, checkpoint_forward=checkpoint)
        unperm1_out, alltoall_token_unperm1_vjp = run_graph_forward(alltoall_token_unperm1_func, detached_expert_output, global_map_info)


        expert_output.untyped_storage().resize_(0)
        _, unperm_a2a_out, unperm_a2a_handle = async_all_to_all(
            unperm1_out,
            fwd_layer.mlp.token_dispatcher.input_splits,
            fwd_layer.mlp.token_dispatcher.output_splits,
            ep_group
        )

        share_experts_graph = None
        if use_shared_experts:
            if tp_size > 1:
                share_experts_graph, shared_expert_output, rs_shared_experts_handle = async_reduce_scatter(
                    shared_expert_output, tp_group
                )
                rs_shared_experts_handle.wait()
                rs_shared_experts_handle = None
                share_experts_graph.untyped_storage().resize_(0)
            else:
                share_experts_graph = shared_expert_output

        if recomp_norm:
            fwd_layer.norm_ckpt2.discard_output()

    WeightGradStore.start_decouple()
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.wait_swap_in()
    detached_layer_input_grad, _ = run_graph_backward(bwd_layer_graph.attn_graph, detached_attention_out_grad, keep_grad=True)
    WeightGradStore.end_decouple()

    # to_check
    layer_input_grad = detached_layer_input_grad
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        (unperm_a2a_graph_grad, detached_shared_expert_output_grad, residual2_grad, probs_grad) = run_graph_backward(next_bwd_layer_graph.unperm2_graph, layer_input_grad, keep_graph=True, keep_grad=True)

    unperm_a2a_handle.wait()
    unperm_a2a_handle = None
    unperm1_out.untyped_storage().resize_(0)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = layer_input_grad, None
    # to_check
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            unperm_a2a_graph_grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )
        next_layer_output_grad = (next_layer_output_grad, detached_shared_expert_output_grad, residual2_grad, probs_grad)


    with checkpoint_context:
        detached_unperm_a2a_out = detach_tensor(unperm_a2a_out, checkpoint_forward=checkpoint)
        if hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None:
            detached_shared_expert_output = detach_tensor(shared_expert_output, checkpoint_forward=checkpoint)
        else:
            detached_shared_expert_output = None

    unperm2_swap_manager = None

    def alltoall_token_unperm2_func(detached_unperm_a2a_out, detached_shared_expert_output, residual2, probs):
        nonlocal unperm2_swap_manager
        if args.moe_unperm2_mem_optim:
            probs = None
        route_expert_output, unperm2_swap_manager = alltoall_token_unperm2(fwd_layer.mlp.token_dispatcher, detached_unperm_a2a_out, probs)
        if hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None:
            mlp_output = route_expert_output + detached_shared_expert_output
            if args.moe_unperm2_mem_optim:
                shared_expert_output.untyped_storage().resize_(0)
        else:
            mlp_output = route_expert_output

        if recomp_norm:
            mlp_output.register_hook(fwd_layer.norm_ckpt2.recompute)

        with fwd_layer.bias_dropout_add_exec_handler():
            hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                (mlp_output, None), residual2, fwd_layer.hidden_dropout
            )
        return hidden_states


    hidden_states, alltoall_token_unperm2_vjp = run_graph_forward(alltoall_token_unperm2_func, detached_unperm_a2a_out,
                                                                  detached_shared_expert_output, residual2,
                                                                  probs_detached)


    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, layer_input_grad)
    WeightGradStore.pop()

    saved_tensors = (
        (attention_graph, detached_attention_out, attention_forward_vjp),
        (pre_mlp_layernorm_output, detached_mlp_input, pre_mlp_vjp),
        (probs, probs_detached, router_forward_vjp),
        ((perm1_out, perm1_probs), (None, None), perm1_vjp),  # perm1 graph
        (None, (detached_perm_a2a_out, (detached_perm_prob_a2a_out, detached_dispached_input_probs))),
        ((dispached_input, dispached_input_probs), (detached_dispached_input, detached_dispached_input_probs), perm2_vjp), # perm2 graph
        (expert_output, detached_expert_output, mlp_experts_vjp),  # grouped mlp graph
        (unperm1_out, None, alltoall_token_unperm1_vjp),  # unperm1 graph
        (None, detached_unperm_a2a_out),
        (output, None, alltoall_token_unperm2_vjp),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output, shared_experts_vjp),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors,
        fwd_layer.mlp.token_dispatcher.input_splits, fwd_layer.mlp.token_dispatcher.output_splits, fwd_layer,
        checkpointed=checkpoint
    )
    graph.act_ckpt_manager = act_ckpt_manager
    graph.unperm2_swap_manager = unperm2_swap_manager
    if hasattr(fwd_layer.self_attention, 'swap_managers'):
        graph.attn_swap_managers = fwd_layer.self_attention.swap_managers

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)

    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, layer_input_grad))



def transformer_layer_forward_dense_backward_dense_overlaping(
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
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    if args.moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    recomp_norm = getattr(args, 'recompute_norm', False)
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    with checkpoint_context:
        # Atten Fwd
        detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        def attention_func(detached_layer_input, residual1):
            hidden_states = attention_forward(
                fwd_layer, detached_layer_input, residual1,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                recompute_norm=recomp_norm
            )
            return hidden_states

        hidden_states, attention_forward_vjp = run_graph_forward(attention_func, detached_layer_input, residual1)

        attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states,
                                                                               checkpoint_forward=checkpoint)
        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            fwd_layer.norm_ckpt2 = CheckpointWithoutOutput()

            def pre_mlp_layernorm_func(detached_attention_out):
                pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False,
                                                                      detached_attention_out)
                return pre_mlp_layernorm_output
        else:

            def pre_mlp_layernorm_func(detached_attention_out):
                pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)
                return pre_mlp_layernorm_output

        pre_mlp_layernorm_output, pre_mlp_layernorm_forward_vjp = run_graph_forward(pre_mlp_layernorm_func,
                                                                                    detached_attention_out)

        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)

        def func_mlp(detached_mlp_input, residual2):
            mlp_output_with_bias = fwd_layer.mlp(detached_mlp_input)
            if recomp_norm:
                fwd_layer.norm_ckpt2.discard_output()
                mlp_output_with_bias[0].register_hook(fwd_layer.norm_ckpt2.recompute)

            # inside the module provided in the `bias_dropout_add_spec` module?
            with fwd_layer.bias_dropout_add_exec_handler():
                hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual2, fwd_layer.hidden_dropout
                )
            return hidden_states

        output, output_vjp = run_graph_forward(func_mlp, detached_mlp_input, residual2)


    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # Detach backward into dx/dw
    WeightGradStore.start_decouple()
    detached_mlp_input_grad, residual2_grad = run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)  # keep for dw
    detached_attention_out_grad, = run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, detached_mlp_input_grad, keep_graph=True)
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.wait_swap_in()
    detached_layer_input_grad, _ = run_graph_backward(bwd_layer_graph.attn_graph, detached_attention_out_grad + residual2_grad, keep_grad=True)
    bwd_layer_input_grad = detached_layer_input_grad

    WeightGradStore.end_decouple()

    unperm_a2a_grad = None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        (next_bwd_unperm_a2a_out_grad, next_bwd_shared_expert_output_grad, next_bwd_residual2_grad, next_bwd_probs_grad) = run_graph_backward(next_bwd_layer_graph.unperm2_graph, bwd_layer_input_grad, keep_graph=True, keep_grad=True)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = bwd_layer_input_grad, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            next_bwd_unperm_a2a_out_grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )
        next_layer_output_grad = (next_layer_output_grad, next_bwd_shared_expert_output_grad, next_bwd_residual2_grad, next_bwd_probs_grad)


    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_layer_input_grad)

    WeightGradStore.pop()

    saved_tensors = (
        (attention_graph, detached_attention_out, attention_forward_vjp),
        (pre_mlp_layernorm_output, detached_mlp_input, pre_mlp_layernorm_forward_vjp),
        (None, None, None),
        (None, None, None),  # perm1 graph
        (None, None, None),
        (None, None, None),  # perm2 graph
        (None, None, None),  # grouped mlp graph
        (None, None, None),  # unperm1 graph
        (None, None, None),
        (output, None, output_vjp),  # unperm2 graph
        (None, None, None),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, [], None, None, fwd_layer,
        checkpointed=checkpoint
    )

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            del tensor

    if hasattr(fwd_layer.self_attention, 'swap_managers'):
        graph.attn_swap_managers = fwd_layer.self_attention.swap_managers

    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_layer_input_grad))


def transformer_layer_forward_moe_backward_moe_overlaping(
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
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    if args.moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    use_shared_experts = hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None
    recomp_norm = getattr(args, 'recompute_norm', False)
    swap_unperm2 = getattr(args, 'moe_unperm2_mem_optim_swap', False)
    bwd_dispached_input, bwd_probs, bwd_indices, global_input_tokens_local_experts_indices = bwd_layer_graph.recompute_needed_tensors
    a2a_hooked_on_attention = getattr(fwd_layer.self_attention, 'a2a_hooked_on_attention', False)

    # Launch swap-in
    if bwd_layer_graph.unperm2_swap_manager:
        bwd_layer_graph.unperm2_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    # shard experts backward grad Allgather
    last_comm_handle = None
    shared_experts_grad = bwd_layer_output_grad if bwd_unperm_a2a_handle is None else bwd_layer_output_grad[1]
    if tp_size > 1:
        _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
            shared_experts_grad, tp_group,
            stream=torch.npu.current_stream() if last_comm_handle else None
        )
        last_comm_handle = backward_ag_shared_handle
    else:
        backward_ag_shared = shared_experts_grad
        backward_ag_shared_handle = None

    # Unperm2 Bwd
    # check if backward unpermutation alltoall is launched at bwd layer before
    if bwd_unperm_a2a_handle is None:
        unperm_a2a_graph_grad, shared_expert_output_grad, residual2_grad, probs_grad = run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)
        # Async Unperm A2A
        if tp_size > 1 and a2a_hooked_on_attention:
            set_async_alltoall_inputs(
                unperm_a2a_graph_grad,
                bwd_layer_graph.output_splits,
                bwd_layer_graph.input_splits,
                ep_group,
                last_comm_handle,
                torch.npu.current_stream() if last_comm_handle else None
            )
        else:
            _, unperm1_out_grad, bwd_unperm_a2a_handle = async_all_to_all(
                unperm_a2a_graph_grad,
                bwd_layer_graph.output_splits,
                bwd_layer_graph.input_splits,
                ep_group,
                last_comm_handle,
                torch.npu.current_stream() if last_comm_handle else None
            )
            last_comm_handle = bwd_unperm_a2a_handle
    else:
        unperm1_out_grad, shared_expert_output_grad, residual2_grad, probs_grad = bwd_layer_output_grad

    if args.moe_zero_memory == 'level0':
        with torch.no_grad():
            bwd_input_before_perm1 = bwd_layer_graph.pre_mlp_layernorm_graph[0]

            def recomp_token_permutation1(hidden_states, indices):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, _ = permute(
                    hidden_states, indices
                )
                return permutated_local_input_tokens

            bwd_perm1_out = recomp_token_permutation1(bwd_input_before_perm1, bwd_indices)

    with checkpoint_context:

        # Residual connection.
        detached_layer_input = detach_tensor(hidden_states)
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward

        def attention_func(detached_layer_input, residual1):
            hidden_states = attention_forward(
                fwd_layer, detached_layer_input, residual1,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                recompute_norm=recomp_norm
            )
            return hidden_states

        hidden_states, attention_forward_vjp = run_graph_forward(attention_func, detached_layer_input, residual1)


        if bwd_unperm_a2a_handle is None and tp_size > 1 and a2a_hooked_on_attention:
            unperm1_out_grad, bwd_unperm_a2a_handle = get_async_alltoall_outputs()

        attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states)

        # Residual connection.
        residual2 = detached_attention_out


        if recomp_norm:
            fwd_layer.norm_ckpt2 = CheckpointWithoutOutput()

            def pre_mlp_layernorm_func(detached_attention_out):
                pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False,
                                                                           detached_attention_out)
                return pre_mlp_layernorm_output
        else:

            def pre_mlp_layernorm_func(detached_attention_out):
                pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)
                return pre_mlp_layernorm_output

        pre_mlp_layernorm_output, pre_mlp_layernorm_forward_vjp = run_graph_forward(pre_mlp_layernorm_func,
                                                                                    detached_attention_out)

        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output)
        indices = None

        def router_func(detached_mlp_input):
            nonlocal indices
            probs, indices = router_forward(fwd_layer, detached_mlp_input)
            return probs

        probs, router_forward_vjp = run_graph_forward(router_func, detached_mlp_input)

        if tp_size > 1 and use_shared_experts:
            # launch tp comm here and wait last aync comm finish
            _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                detached_mlp_input, tp_group, event=bwd_unperm_a2a_handle,
                stream=torch.npu.current_stream() if bwd_unperm_a2a_handle else None,
                is_use_get_global_memory_buffer=True
            )
            AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            shared_experts_input, shared_experts_allgather_handle = detached_mlp_input, None

        # Token Permutation1 Forward
        probs_detached = detach_tensor(probs)
        tokens_per_expert = None

        def alltoall_token_perm1_func(detached_mlp_input, probs_detached):
            nonlocal tokens_per_expert
            perm1_out, perm1_probs, tokens_per_expert = alltoall_token_perm1(fwd_layer.mlp.token_dispatcher, detached_mlp_input, probs_detached, indices)
            return perm1_out, perm1_probs
        (perm1_out, perm1_probs), perm1_vjp = run_graph_forward(alltoall_token_perm1_func, detached_mlp_input, probs_detached) # @check tokens_per_expert grad



        # backward
        shared_experts_vjp = None
        if use_shared_experts:
            if shared_experts_allgather_handle is not None:
                shared_experts_allgather_handle.wait()
                shared_experts_allgather_handle = None

            def mlp_shared_expert_func(detached_mlp_input):
                shared_expert_output, _ = fwd_layer.mlp.shared_experts(detached_mlp_input)
                return shared_expert_output

            shared_expert_output, shared_experts_vjp = run_graph_forward(mlp_shared_expert_func,
                                                                            detached_mlp_input)

        if args.moe_zero_memory != 'disable':
            _, bwd_perm_a2a_out, bwd_recomp_perm_a2a_handle = async_all_to_all(
                bwd_perm1_out,
                bwd_layer_graph.output_splits,
                bwd_layer_graph.input_splits,
                ep_group,
                stream=overlap_stream.stream
            )
            last_comm_handle = bwd_recomp_perm_a2a_handle

        disp = fwd_layer.mlp.token_dispatcher
        if disp.num_local_experts > 1:
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            disp.cuda_sync_point = "no_sync"
            disp.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                disp.expert_ids_per_ep_rank, disp.num_global_tokens_per_local_expert.ravel()
            )

        torch.npu.current_stream().wait_stream(overlap_stream.stream)

    bwd_unperm_a2a_handle.wait()
    bwd_unperm_a2a_handle = None
    mlp_graph_grad, _ = run_graph_backward(bwd_layer_graph.unperm1_graph, unperm1_out_grad)
    if args.moe_zero_memory == 'level0' or should_recompute_activation(bwd_layer_graph.layer.layer_number):
        bwd_layer_graph.act_ckpt_manager.recompute(True)
    unperm1_out_grad.untyped_storage().resize_(0)

    # Shared Experts Backward
    if backward_ag_shared_handle is not None:
        # ensure tp comm is not overlaped with alltoall comm
        backward_ag_shared_handle.wait()
        backward_ag_shared_handle = None

    WeightGradStore.start_decouple()
    mlp_input_grad_sharedep, = run_graph_backward(bwd_layer_graph.shared_experts_graph, backward_ag_shared, keep_grad=True)  # dw computation
    WeightGradStore.end_decouple()

    with checkpoint_context:
        _, perm_a2a_out, perm_a2a_handle = async_all_to_all(
            perm1_out,
            fwd_layer.mlp.token_dispatcher.output_splits,
            fwd_layer.mlp.token_dispatcher.input_splits,
            ep_group,
            event=last_comm_handle,
            stream=torch.npu.current_stream() if last_comm_handle else None
        )
        last_comm_handle = perm_a2a_handle
        perm_prob_a2a_out, perm_prob_a2a_handle = None, None
        if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name():
            _, perm_prob_a2a_out, perm_prob_a2a_handle = async_all_to_all(
                perm1_probs,
                fwd_layer.mlp.token_dispatcher.output_splits,
                fwd_layer.mlp.token_dispatcher.input_splits,
                ep_group,
                event=last_comm_handle,
                stream=torch.npu.current_stream() if last_comm_handle else None
            )
            last_comm_handle = perm_prob_a2a_handle

    WeightGradStore.start_decouple()
    perm2_graph_grad, detached_dispached_input_probs_grad = run_graph_backward(bwd_layer_graph.grouped_mlp_graph, mlp_graph_grad, keep_grad=True)  # keep for dw
    WeightGradStore.end_decouple()

    with checkpoint_context:
        if use_shared_experts:
            if tp_size > 1:
                # launch tp comm after permf a2a and wait until shared experts computation finish.
                share_experts_graph, shared_expert_output, rs_shared_experts_handle = async_reduce_scatter(
                    shared_expert_output, tp_group, event=last_comm_handle,
                    stream=torch.npu.current_stream() if last_comm_handle else None
                )
                last_comm_handle = rs_shared_experts_handle
            else:
                share_experts_graph = shared_expert_output
                rs_shared_experts_handle = None
    if recomp_norm:
        fwd_layer.norm_ckpt2.discard_output()

    # in forward, the 3rd input tensor of perm2 function is global_map_info, the grad of global_map_info will not be used here, it is used in moe_zerc's forward_moe_backward_moe function
    perm_a2a_graph_grad, perm_prob_a2a_out_grad, _ = run_graph_backward(bwd_layer_graph.perm2_graph, (perm2_graph_grad, detached_dispached_input_probs_grad), keep_graph=True)

    _, perm1_out_grad, bwd_perm_a2a_handle = async_all_to_all(
        perm_a2a_graph_grad,
        bwd_layer_graph.input_splits,
        bwd_layer_graph.output_splits,
        ep_group,
        event=last_comm_handle,
        stream=torch.npu.current_stream() if last_comm_handle else None
    )
    last_comm_handle = bwd_perm_a2a_handle
    perm1_prob_out_grad, bwd_prob_handle = None, None
    if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name():
        _, perm1_prob_out_grad, bwd_prob_handle = async_all_to_all(
            perm_prob_a2a_out_grad,
            bwd_layer_graph.input_splits,
            bwd_layer_graph.output_splits,
            ep_group,
            event=last_comm_handle,
            stream=torch.npu.current_stream()
        )
        last_comm_handle = bwd_prob_handle

    # Grouped MLP dw computation
    if args.moe_zero_memory == 'level0':
        # restore fc1 input for dw computation
        with torch.no_grad():
            bwd_recomp_perm_a2a_handle.wait()
            bwd_recomp_perm_a2a_handle = None
            recompute_fc1_input, _ = permute(bwd_perm_a2a_out, global_input_tokens_local_experts_indices)
            bwd_perm_a2a_out.untyped_storage().resize_(0)
        bwd_dispached_input.untyped_storage().resize_(recompute_fc1_input.untyped_storage().size())
        bwd_dispached_input.untyped_storage().copy_(recompute_fc1_input.untyped_storage(), False)
        recompute_fc1_input.untyped_storage().resize_(0)

    WeightGradStore.pop()

    with checkpoint_context:
        # Token Perm2 Forward
        perm_a2a_handle.wait()
        perm_a2a_handle = None
        perm1_out.untyped_storage().resize_(0)
        detached_perm_a2a_out = detach_tensor(perm_a2a_out)
        detached_perm_prob_a2a_out = detach_tensor(perm_prob_a2a_out, checkpoint_forward=checkpoint)
        if args.moe_unperm2_mem_optim and '910B' not in acl.get_soc_name():
            perm_prob_a2a_handle.wait()

        # global_map_info is not used here, it's used in moe_zerc's forward_moe_backward_moe function
        global_map_info = None

        def alltoall_token_perm2_func(detached_perm_a2a_out, detached_perm_prob_a2a_out, global_map_info=None):
            return alltoall_token_perm2(fwd_layer.mlp.token_dispatcher, detached_perm_a2a_out, detached_perm_prob_a2a_out)

        (dispached_input, dispached_input_probs), perm2_vjp = run_graph_forward(alltoall_token_perm2_func, detached_perm_a2a_out, detached_perm_prob_a2a_out, global_map_info)
        perm_a2a_out.untyped_storage().resize_(0)

        # Grouped MLP Forward
        detached_dispached_input = detach_tensor(dispached_input)
        detached_dispached_input_probs = detach_tensor(dispached_input_probs, checkpoint_forward=checkpoint)
        act_ckpt_manager = None

        def mlp_experts_func(detached_dispached_input, detached_dispached_input_probs):
            nonlocal act_ckpt_manager
            (expert_output, act_ckpt_manager), _ = fwd_layer.mlp.experts(
                detached_dispached_input, tokens_per_expert, permuted_probs=detached_dispached_input_probs
            )
            return expert_output

        expert_output, mlp_experts_vjp = run_graph_forward(mlp_experts_func, detached_dispached_input, detached_dispached_input_probs)
        if args.moe_zero_memory == 'level0':
            dispached_input.untyped_storage().resize_(0)
            recompute_needed_tensors = [dispached_input, probs, indices,
                                        fwd_layer.mlp.token_dispatcher.global_input_tokens_local_experts_indices]
        else:
            recompute_needed_tensors = [None, None, None, None]
        detached_expert_output = detach_tensor(expert_output)

        # Token Unpermutaion Forward
        def alltoall_token_unperm1_func(detached_expert_output, global_map_info=None):
            unperm1_out = alltoall_token_unperm1(fwd_layer.mlp.token_dispatcher, detached_expert_output, None)
            return unperm1_out

        unperm1_out, unperm1_vjp = run_graph_forward(alltoall_token_unperm1_func, detached_expert_output, global_map_info)

        expert_output.untyped_storage().resize_(0)
        if rs_shared_experts_handle is not None:
            rs_shared_experts_handle.wait()
            rs_shared_experts_handle = None
            share_experts_graph.untyped_storage().resize_(0)
        bwd_perm_a2a_handle.wait()
        bwd_perm_a2a_handle = None

    with checkpoint_context:
        # launch async all2all in the middle of attention graph backward
        if tp_size > 1 and a2a_hooked_on_attention:
            set_async_alltoall_inputs(
                unperm1_out, fwd_layer.mlp.token_dispatcher.input_splits, fwd_layer.mlp.token_dispatcher.output_splits, ep_group
            )
        else:
            _, unperm_a2a_out, unperm_a2a_handle = async_all_to_all(
                unperm1_out,
                fwd_layer.mlp.token_dispatcher.input_splits,
                fwd_layer.mlp.token_dispatcher.output_splits,
                ep_group
            )

    if bwd_prob_handle:
        bwd_prob_handle.wait()
    (detached_mlp_input_grad, probs_detached_grad) = run_graph_backward(bwd_layer_graph.perm1_graph, [perm1_out_grad, perm1_prob_out_grad])
    perm1_out_grad.untyped_storage().resize_(0)

    # router backward
    if bwd_layer_graph.unperm2_swap_manager:
        bwd_layer_graph.unperm2_swap_manager.wait_swap_in()

    if swap_unperm2:
        # dprobs computation
        output_grad = bwd_layer_output_grad
        if hasattr(bwd_layer_graph, 'last_layer_input_grad'):
            output_grad = bwd_layer_graph.last_layer_input_grad
        H = bwd_layer_graph.unperm2_swap_manager.npu_tensor.shape[-1]
        K = args.moe_router_topk
        probs_dtype = bwd_probs.dtype
        probs_grad = bwd_layer_graph.unperm2_swap_manager.npu_tensor.reshape(-1, K, H).to(probs_dtype) * output_grad.to(probs_dtype)
        output_grad.untyped_storage().resize_(0)
        bwd_layer_graph.unperm2_swap_manager.npu_tensor.untyped_storage().resize_(0)
        probs_grad = probs_grad.sum(dim=-1)

    if args.moe_unperm2_mem_optim:
        probs_grad = probs_detached_grad
    (detached_mlp_input_router_grad, ) = run_graph_backward(bwd_layer_graph.router_graph, probs_grad)
    detached_mlp_input_grad = detached_mlp_input_grad + mlp_input_grad_sharedep + detached_mlp_input_router_grad
    (detached_attention_out_grad, ) = run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, detached_mlp_input_grad, keep_graph=True)

    WeightGradStore.start_decouple()
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.wait_swap_in()
    detached_layer_input_grad, _ = run_graph_backward(bwd_layer_graph.attn_graph, detached_attention_out_grad + residual2_grad, keep_grad=True)
    layer_input_grad_sum = detached_layer_input_grad
    
    WeightGradStore.end_decouple()
    if tp_size > 1 and a2a_hooked_on_attention:
        unperm_a2a_out, unperm_a2a_handle = get_async_alltoall_outputs()

    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        (next_bwd_unperm_a2a_out_grad, next_bwd_shared_expert_output_grad, next_bwd_residual2_grad, next_bwd_probs_grad) = run_graph_backward(next_bwd_layer_graph.unperm2_graph, layer_input_grad_sum, keep_graph=True, keep_grad=True)

    unperm_a2a_handle.wait()
    unperm_a2a_handle = None
    unperm1_out.untyped_storage().resize_(0)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = layer_input_grad_sum, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            next_bwd_unperm_a2a_out_grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )
        next_layer_output_grad = (
            next_layer_output_grad, next_bwd_shared_expert_output_grad, next_bwd_residual2_grad, next_bwd_probs_grad)
    # forward
    with checkpoint_context:
        detached_unperm_a2a_out = detach_tensor(unperm_a2a_out)
        if hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None:
            detached_shared_expert_output = detach_tensor(shared_expert_output)
        else:
            detached_shared_expert_output = None
            share_experts_graph = None

        unperm2_swap_manager = None

        def alltoall_token_unperm2_func(detached_unperm_a2a_out, detached_shared_expert_output, residual2, probs):
            nonlocal unperm2_swap_manager
            if args.moe_unperm2_mem_optim:
                probs = None
            route_expert_output, unperm2_swap_manager = alltoall_token_unperm2(fwd_layer.mlp.token_dispatcher, detached_unperm_a2a_out, probs)
            if args.moe_unperm2_mem_optim:
                unperm_a2a_out.untyped_storage().resize_(0)

            if hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None:
                mlp_output = route_expert_output + detached_shared_expert_output
                shared_expert_output.untyped_storage().resize_(0)
            else:
                mlp_output = route_expert_output

            if recomp_norm:
                mlp_output.register_hook(fwd_layer.norm_ckpt2.recompute)


            with fwd_layer.bias_dropout_add_exec_handler():
                hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                    (mlp_output, None), residual2, fwd_layer.hidden_dropout
                )
            return hidden_states

        output, unperm2_vjp = run_graph_forward(alltoall_token_unperm2_func,
                                                detached_unperm_a2a_out, detached_shared_expert_output, residual2, probs_detached)


    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, layer_input_grad_sum)

    WeightGradStore.pop()

    saved_tensors = (
        (attention_graph, detached_attention_out, attention_forward_vjp),
        (pre_mlp_layernorm_output, detached_mlp_input, pre_mlp_layernorm_forward_vjp),
        (probs, probs_detached, router_forward_vjp),
        ((perm1_out, perm1_probs), (None, None), perm1_vjp),  # perm1 graph
        (None, (detached_perm_a2a_out, detached_perm_prob_a2a_out), None),
        ((dispached_input, dispached_input_probs), (detached_dispached_input, detached_dispached_input_probs), perm2_vjp), # perm2 graph
        (expert_output, detached_expert_output, mlp_experts_vjp),  # grouped mlp graph
        (unperm1_out, None, unperm1_vjp),  # unperm1 graph
        (None, detached_unperm_a2a_out, None),
        (output, None, unperm2_vjp),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output, shared_experts_vjp),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors,
        fwd_layer.mlp.token_dispatcher.input_splits, fwd_layer.mlp.token_dispatcher.output_splits, fwd_layer,
        checkpointed=checkpoint
    )
    graph.act_ckpt_manager = act_ckpt_manager
    graph.unperm2_swap_manager = unperm2_swap_manager
    if hasattr(fwd_layer.self_attention, 'swap_managers'):
        graph.attn_swap_managers = fwd_layer.self_attention.swap_managers

    # save original layer output for probs_grad computation
    if swap_unperm2 \
        and next_bwd_layer_graph is not None \
        and getattr(next_bwd_layer_graph, 'is_moe_layer', False):

        next_bwd_layer_graph.last_layer_input_grad = layer_input_grad
    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)

    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, layer_input_grad_sum))
