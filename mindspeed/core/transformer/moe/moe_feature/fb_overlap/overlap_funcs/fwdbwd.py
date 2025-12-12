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
from ..modules.weight_grad_store import WeightGradStore
from ..modules.attention import (
    attention_forward, set_async_alltoall_inputs, get_async_alltoall_outputs
)
from ..modules.utils import (
    detach_tensor, run_graph_backward, LayerGraph, is_p2p_comm_needed,
    p2p_comm_helper, P2PCommOutput, P2PCommParams
)


def router_forward(
    self,
    hidden_states
):
    probs, routing_map = self.mlp.router(hidden_states)

    return probs, routing_map


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
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    raise NotImplementedError('forward dense & backward moe is not yet implemented.')


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
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    checkpoint=False
):
    raise NotImplementedError('forward moe & backward dense is not yet implemented.')


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
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
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
    use_shared_experts = hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None
    fwd_shared_experts = fwd_layer.mlp.shared_experts if use_shared_experts else None
    bwd_shared_experts = bwd_layer_graph.layer.mlp.shared_experts if use_shared_experts else None
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    recomp_norm = getattr(args, 'recompute_norm', False)
    swap_unperm2 = getattr(args, 'moe_unperm2_mem_optim_swap', False)
    bwd_dispached_input, bwd_probs, bwd_routing_map, bwd_num_global_tokens_per_local_expert_cpu = bwd_layer_graph.recompute_needed_tensors
    a2a_hooked_on_attention = getattr(fwd_layer.self_attention, 'a2a_hooked_on_attention', False)
    fwd_dispatcher = fwd_layer.mlp.token_dispatcher
    bwd_dispatcher = bwd_layer_graph.layer.mlp.token_dispatcher
    fwd_layer.mlp.experts.layer_number = fwd_layer.layer_number

    # Launch swap-in
    if bwd_layer_graph.unperm2_swap_manager:
        bwd_layer_graph.unperm2_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.async_swap_in(wait_stream=torch.npu.current_stream())

    # shard experts backward grad Allgather
    last_comm_handle = None
    if use_shared_experts:
        shared_experts_grad = bwd_layer_output_grad if bwd_unperm_a2a_handle is None else bwd_layer_graph.shared_experts_graph[1].grad
        bwd_shared_experts.pre_backward_comm(shared_experts_grad)
        last_comm_handle = bwd_shared_experts.pre_backward_handle

    if bwd_layer_graph.fc1_swap_manager:
        bwd_layer_graph.fc1_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    if bwd_layer_graph.probs_swap_manager:
        bwd_layer_graph.probs_swap_manager.async_swap_in(wait_stream=torch.npu.current_stream())
    # Unperm2 Bwd
    # check if backward unpermutation alltoall is launched at bwd layer before
    if bwd_unperm_a2a_handle is None:
        run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)
        # Async Unperm A2A
        if tp_size > 1 and a2a_hooked_on_attention:
            set_async_alltoall_inputs(
                bwd_dispatcher.backward_async_combine_comm,
                bwd_layer_graph.unperm_a2a_graph[1].grad,
                input_splits=bwd_layer_graph.input_splits,
                output_splits=bwd_layer_graph.output_splits,
                output_splits_tp=bwd_layer_graph.output_splits_tp,
                wait_event=last_comm_handle
            )
        else:
            unperm1_out_grad, bwd_unperm_a2a_handle = bwd_dispatcher.backward_async_combine_comm(
                bwd_layer_graph.unperm_a2a_graph[1].grad,
                input_splits=bwd_layer_graph.input_splits,
                output_splits=bwd_layer_graph.output_splits,
                output_splits_tp=bwd_layer_graph.output_splits_tp,
                wait_event=last_comm_handle
            )
            last_comm_handle = bwd_unperm_a2a_handle
    else:
        unperm1_out_grad = bwd_layer_output_grad

    if args.moe_zero_memory != 'disable':
        with torch.no_grad():
            bwd_input_before_perm1 = bwd_layer_graph.pre_mlp_layernorm_graph[0]

            def recomp_token_permutation1(hidden_states, routing_map):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, _, _ = permute(
                    hidden_states, routing_map, num_out_tokens=bwd_dispatcher.num_out_tokens, fused=args.moe_permute_fusion
                )
                return permutated_local_input_tokens

            bwd_perm1_out = recomp_token_permutation1(bwd_input_before_perm1, bwd_routing_map)

    with checkpoint_context:

        # Residual connection.
        detached_layer_input = hidden_states

        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        hidden_states = attention_forward(
            fwd_layer, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )

        if bwd_unperm_a2a_handle is None and tp_size > 1 and a2a_hooked_on_attention:
            unperm1_out_grad, bwd_unperm_a2a_handle = get_async_alltoall_outputs()

        attention_graph, detached_attention_out = hidden_states, detach_tensor(hidden_states)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            fwd_layer.norm_ckpt2 = CheckpointWithoutOutput()
            pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False, detached_attention_out)
        else:
            pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)
        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output)
        if hasattr(fwd_layer.mlp.token_dispatcher, "num_tokens_per_expert") and (getattr(args, "enable_expert_placement", 
                                                            False) or getattr(args, "print_expert_load", False)):
            fwd_layer.mlp.predict_expert_load(fwd_layer.mlp.token_dispatcher.num_tokens_per_expert)
        
        probs, routing_map = router_forward(fwd_layer, detached_mlp_input)
        if use_shared_experts:
            fwd_shared_experts.pre_forward_comm(detached_mlp_input, wait_event=bwd_unperm_a2a_handle)
            shared_fc1_input = fwd_shared_experts.cached_fc1_input
        else:
            shared_fc1_input = None

        # Token Permutation1 Forward
        probs_detached = detach_tensor(probs)
        perm1_out, perm1_probs, tokens_per_expert = fwd_dispatcher.token_permute1(detached_mlp_input, probs_detached, routing_map)

        if use_shared_experts:
            # Shared Experts Forward.
            fwd_shared_experts.linear_fc1_forward_and_act()
            fwd_shared_experts.linear_fc2_forward()

        if args.moe_zero_memory != 'disable':
            (bwd_perm_a2a_out, bwd_recomp_perm_a2a_handle), _ = bwd_dispatcher.async_dispatch_comm(
                bwd_perm1_out,
                output_splits=bwd_layer_graph.output_splits,
                input_splits=bwd_layer_graph.input_splits,
                output_splits_tp=bwd_layer_graph.output_splits_tp
            )
            last_comm_handle = bwd_recomp_perm_a2a_handle

        if fwd_dispatcher.num_local_experts > 1:
            # launch synchronization here to wait for non-blocking mem copy in preprocess func.
            fwd_dispatcher.cuda_sync_point = "no_sync"
            torch.npu.current_stream().synchronize()

        torch.npu.current_stream().wait_stream(fwd_dispatcher.overlap_stream)

    bwd_unperm_a2a_handle.wait()
    bwd_unperm_a2a_handle = None
    run_graph_backward(bwd_layer_graph.unperm1_graph, unperm1_out_grad)
    unperm1_out_grad.untyped_storage().resize_(0)

    # Shared Experts Backward
    if use_shared_experts:
        with torch.npu.stream(bwd_dispatcher.overlap_stream):
            WeightGradStore.start_decouple()
            bwd_shared_experts.linear_fc2_act_fc1_backward(bwd_layer_graph.shared_experts_graph, keep_grad=True)
            WeightGradStore.end_decouple()

    with checkpoint_context:
        # Async Perm A2A.
        from ..modules.token_dispatcher import PREMUTE_FINISH_EVENT
        if PREMUTE_FINISH_EVENT is not None:
            #Wait for permute1 finish.
            torch.npu.current_stream().wait_event(PREMUTE_FINISH_EVENT)
        (perm_a2a_out, perm_a2a_handle), (perm_prob_a2a_out, perm_prob_a2a_handle) = fwd_dispatcher.async_dispatch_comm(
            perm1_out, perm1_probs, wait_event=last_comm_handle
        )
        last_comm_handle = perm_prob_a2a_handle if perm_prob_a2a_handle else perm_a2a_handle

    if bwd_layer_graph.fc1_swap_manager:
        bwd_layer_graph.fc1_swap_manager.wait_swap_in()
    if bwd_layer_graph.probs_swap_manager:
        bwd_layer_graph.probs_swap_manager.wait_swap_in()
    if bwd_layer_graph.act_ckpt_manager is not None:
        bwd_layer_graph.act_ckpt_manager.recompute(True)

    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.grouped_mlp_graph, keep_grad=True)  # keep for dw
    WeightGradStore.end_decouple()

    with checkpoint_context:
        if use_shared_experts:
            with torch.npu.stream(fwd_dispatcher.overlap_stream):
                fwd_shared_experts.post_forward_comm(wait_event=last_comm_handle)
                last_comm_handle = fwd_shared_experts.fc2_output_comm_handle

    if recomp_norm:
        fwd_layer.norm_ckpt2.discard_output()

    run_graph_backward(bwd_layer_graph.perm2_graph, keep_graph=True)

    (perm1_out_grad, bwd_perm_a2a_handle), (perm1_prob_out_grad, bwd_prob_handle) = bwd_dispatcher.backward_async_dispatch_comm(
        bwd_layer_graph.perm_a2a_graph[1][0].grad,
        bwd_layer_graph.perm_a2a_graph[1][1].grad,
        input_splits=bwd_layer_graph.output_splits,
        output_splits=bwd_layer_graph.input_splits,
        input_splits_tp=bwd_layer_graph.output_splits_tp,
        wait_event=last_comm_handle
    )
    last_comm_handle = bwd_prob_handle if bwd_prob_handle else bwd_perm_a2a_handle

    # launch shared experts post backward comm
    if use_shared_experts:
        with torch.npu.stream(bwd_dispatcher.overlap_stream):
            bwd_shared_experts.post_backward_comm(wait_event=last_comm_handle)

    # Grouped MLP dw computation
    if args.moe_zero_memory != 'disable':
        # restore fc1 input for dw computation
        with torch.no_grad():
            bwd_recomp_perm_a2a_handle.wait()
            bwd_recomp_perm_a2a_handle = None
            recompute_fc1_input, _ = bwd_dispatcher.token_permute2(bwd_perm_a2a_out, None, bwd_num_global_tokens_per_local_expert_cpu)
            bwd_perm_a2a_out.untyped_storage().resize_(0)
        bwd_dispached_input.untyped_storage().resize_(recompute_fc1_input.untyped_storage().size())
        bwd_dispached_input.untyped_storage().copy_(recompute_fc1_input.untyped_storage())
        recompute_fc1_input.untyped_storage().resize_(0)

    WeightGradStore.pop(experts_only=True)

    with checkpoint_context:
        # Token Perm2 Forward
        perm_a2a_handle.wait()
        perm_prob_a2a_handle.wait()
        perm1_out.untyped_storage().resize_(0)
        detached_perm_a2a_out = detach_tensor(perm_a2a_out)
        detached_perm_prob_a2a_out = detach_tensor(perm_prob_a2a_out, checkpoint_forward=checkpoint)
        dispached_input, dispached_input_probs = fwd_dispatcher.token_permute2(detached_perm_a2a_out, detached_perm_prob_a2a_out)
        perm_a2a_out.untyped_storage().resize_(0)

        # Grouped MLP Forward
        detached_dispached_input = detach_tensor(dispached_input)
        detached_dispached_input_probs = detach_tensor(dispached_input_probs, checkpoint_forward=checkpoint)
        (expert_output, act_ckpt_manager, fc1_swap_manager, probs_swap_manager), _ = fwd_layer.mlp.experts(
            detached_dispached_input, tokens_per_expert, permuted_probs=detached_dispached_input_probs
        )
        if args.moe_zero_memory != 'disable':
            dispached_input.untyped_storage().resize_(0)
            recompute_needed_tensors = [dispached_input, probs, routing_map,
                                        fwd_dispatcher.num_global_tokens_per_local_expert_cpu]
        else:
            recompute_needed_tensors = [None, None, None, None]
        detached_expert_output = detach_tensor(expert_output)

        # Token Unpermutaion Forward
        unperm1_out = fwd_dispatcher.token_unpermute1(detached_expert_output, None)
        expert_output.untyped_storage().resize_(0)
        if use_shared_experts:
            with torch.npu.stream(fwd_dispatcher.overlap_stream):
                shared_expert_output, share_experts_graph = fwd_shared_experts.get_output()
            torch.npu.current_stream().wait_stream(fwd_dispatcher.overlap_stream)
        bwd_perm_a2a_handle.wait()
        bwd_perm_a2a_handle = None

    with checkpoint_context:
        # launch async all2all in the middle of attention graph backward
        if tp_size > 1 and a2a_hooked_on_attention:
            set_async_alltoall_inputs(fwd_dispatcher.async_combine_comm, unperm1_out)
        else:
            unperm_a2a_out, unperm_a2a_handle = fwd_dispatcher.async_combine_comm(unperm1_out)

    if bwd_prob_handle:
        bwd_prob_handle.wait()
    if use_shared_experts:
        with torch.npu.stream(bwd_dispatcher.overlap_stream):
            shared_experts_grad = bwd_shared_experts.get_backward_grad()
            if shared_experts_grad is not None:
                bwd_layer_graph.pre_mlp_layernorm_graph[1].grad = shared_experts_grad

    torch.npu.current_stream().wait_stream(bwd_dispatcher.overlap_stream)
    run_graph_backward(bwd_layer_graph.perm1_graph, [perm1_out_grad, perm1_prob_out_grad])
    perm1_out_grad.untyped_storage().resize_(0)

    # router backward
    if bwd_layer_graph.unperm2_swap_manager:
        bwd_layer_graph.unperm2_swap_manager.wait_swap_in()
    probs_grad = None
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
    run_graph_backward(bwd_layer_graph.router_graph, probs_grad)

    run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, keep_graph=True)
    WeightGradStore.start_decouple()
    if bwd_layer_graph.attn_swap_managers:
        for manager in bwd_layer_graph.attn_swap_managers:
            manager.wait_swap_in()
    run_graph_backward(bwd_layer_graph.attn_graph, keep_grad=True)
    WeightGradStore.end_decouple()
    if tp_size > 1 and a2a_hooked_on_attention:
        unperm_a2a_out, unperm_a2a_handle = get_async_alltoall_outputs()

    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        run_graph_backward(next_bwd_layer_graph.unperm2_graph, bwd_layer_graph.layer_input.grad, keep_graph=True, keep_grad=swap_unperm2)

    unperm_a2a_handle.wait()
    unperm_a2a_handle = None
    unperm1_out.untyped_storage().resize_(0)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = getattr(bwd_layer_graph.layer_input, 'grad', None), None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        next_layer_output_grad, next_bwd_unperm_a2a_handle = next_bwd_layer_graph.layer.mlp.token_dispatcher.backward_async_combine_comm(
            next_bwd_layer_graph.unperm_a2a_graph[1].grad,
            output_splits=next_bwd_layer_graph.output_splits,
            input_splits=next_bwd_layer_graph.input_splits,
            output_splits_tp=next_bwd_layer_graph.output_splits_tp
        )

    with checkpoint_context:
        detached_unperm_a2a_out = detach_tensor(unperm_a2a_out)
        route_expert_output, unperm2_swap_manager = fwd_dispatcher.token_unpermute2(detached_unperm_a2a_out)
        unperm_a2a_out.untyped_storage().resize_(0)

        if use_shared_experts:
            detached_shared_expert_output = detach_tensor(shared_expert_output)
            mlp_output = route_expert_output + detached_shared_expert_output
            shared_expert_output.untyped_storage().resize_(0)
        else:
            detached_shared_expert_output = None
            share_experts_graph = None
            mlp_output = route_expert_output

        if recomp_norm:
            mlp_output.register_hook(fwd_layer.norm_ckpt2.recompute)


        with fwd_layer.bias_dropout_add_exec_handler():
            hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                (mlp_output, None), residual2, fwd_layer.hidden_dropout
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

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_layer_graph.layer_input.grad)

    WeightGradStore.pop()

    saved_tensors = (
        (attention_graph, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (probs, probs_detached),
        ((perm1_out, perm1_probs), (None, None)),  # perm1 graph
        (None, (detached_perm_a2a_out, detached_perm_prob_a2a_out)),
        ((dispached_input, dispached_input_probs), (detached_dispached_input, detached_dispached_input_probs)), # perm2 graph
        (expert_output, detached_expert_output),  # grouped mlp graph
        (unperm1_out, None),  # unperm1 graph
        (None, detached_unperm_a2a_out),
        (output, None),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors, fwd_layer, checkpointed=checkpoint
    )
    graph.act_ckpt_manager = act_ckpt_manager
    graph.unperm2_swap_manager = unperm2_swap_manager
    graph.fc1_swap_manager = fc1_swap_manager
    graph.probs_swap_manager = probs_swap_manager

    if hasattr(fwd_layer.self_attention, 'swap_managers'):
        graph.attn_swap_managers = fwd_layer.self_attention.swap_managers

    # save original layer output for probs_grad computation
    if swap_unperm2 \
        and next_bwd_layer_graph is not None \
        and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
 
        next_bwd_layer_graph.last_layer_input_grad = bwd_layer_graph.layer_input.grad

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)

    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, getattr(bwd_layer_graph.layer_input, 'grad', None)))
