# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from mindspeed.args_utils import get_full_args
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import (
    async_all_to_all,
)
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func,
    backward_func,
)
from mindspeed.core.transformer.moe.moe_feature import (
    tensor_parallel,
    parallel_state,
    MoELayer,
    permute,
    save_to_aux_losses_tracker,
)


def _build_level0_fc1_reorder_index(split_sizes, sorted_idxs, num_tokens, device):
    """Build the token index equivalent to split/sort/cat before backward."""
    split_sizes_cpu = torch.as_tensor(split_sizes, device="cpu", dtype=torch.long).ravel()
    sorted_idxs_cpu = torch.as_tensor(sorted_idxs, device="cpu", dtype=torch.long).ravel()

    num_chunks = split_sizes_cpu.numel()
    if sorted_idxs_cpu.numel() != num_chunks:
        raise RuntimeError(
            "AlltoAll-overlap level0 chunk reorder metadata mismatch: "
            f"got {num_chunks} split sizes and {sorted_idxs_cpu.numel()} sorted indices."
        )
    if torch.any(split_sizes_cpu < 0):
        raise RuntimeError("AlltoAll-overlap level0 chunk reorder received a negative split size.")
    expected_sorted_idxs = torch.arange(num_chunks, dtype=torch.long)
    if not torch.equal(torch.sort(sorted_idxs_cpu).values, expected_sorted_idxs):
        raise RuntimeError(f"AlltoAll-overlap level0 chunk reorder indices must be a permutation of [0, {num_chunks}).")

    metadata_num_tokens = int(split_sizes_cpu.sum().item())
    if metadata_num_tokens != num_tokens:
        raise RuntimeError(
            "AlltoAll-overlap level0 chunk reorder token count mismatch: "
            f"split sizes describe {metadata_num_tokens} tokens, but expert input "
            f"contains {num_tokens}."
        )

    # The original implementation concatenates chunks in sorted_idxs order.
    # Convert that chunk permutation into a token permutation once in forward,
    # so backward only has to submit one index_select on the recompute stream.
    source_offsets = torch.cumsum(split_sizes_cpu, dim=0) - split_sizes_cpu
    sorted_sizes = split_sizes_cpu.index_select(0, sorted_idxs_cpu)
    output_offsets = torch.cumsum(sorted_sizes, dim=0) - sorted_sizes
    source_chunks = torch.repeat_interleave(sorted_idxs_cpu, sorted_sizes, output_size=num_tokens)
    repeated_output_offsets = torch.repeat_interleave(output_offsets, sorted_sizes, output_size=num_tokens)
    offsets_in_chunks = torch.arange(num_tokens, dtype=torch.long) - repeated_output_offsets
    reorder_index_cpu = source_offsets.index_select(0, source_chunks) + offsets_in_chunks
    return reorder_index_cpu.to(device=device, non_blocking=True)


def _start_level0_fc1_input_recompute(ctx, hidden_states, routing_map):
    """
    Launch the level0 permutation-1 AlltoAll before entering TE expert backward.

    TE's autograd owns activation backward and FC1 dgrad as one graph, so the
    communication must be in flight before that graph starts.
    """
    if ctx.moe_zero_memory != "level0":
        return None

    with torch.no_grad():
        permuted_local_tokens, _, _, _, _ = permute(
            hidden_states.view(-1, hidden_states.shape[-1]),
            routing_map,
            probs=None,
            num_out_tokens=ctx.num_out_tokens,
            fused=ctx.config.moe_permute_fusion,
        )
        ep_group = parallel_state.get_expert_model_parallel_group()
        _, global_input_tokens, input_handle = async_all_to_all(
            permuted_local_tokens,
            ctx.output_splits,
            ctx.input_splits,
            ep_group,
        )

        recompute_stream = getattr(ctx.moe_layer.experts, "_mindspeed_level0_recompute_stream", None)
        if recompute_stream is None:
            recompute_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            ctx.moe_layer.experts._mindspeed_level0_recompute_stream = recompute_stream

        launch_stream = torch.cuda.current_stream()
        global_input_tokens.record_stream(recompute_stream)
        ctx.level0_fc1_reorder_index.record_stream(recompute_stream)
        with torch.cuda.stream(recompute_stream):
            # Work.wait() establishes the HCCL-output dependency on the current
            # stream. The explicit stream dependency also covers the permutation
            # that produced the AlltoAll input.
            recompute_stream.wait_stream(launch_stream)
            input_handle.wait()

            if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                output_split_sizes = ctx.output_splits_tp.tolist() if ctx.output_splits_tp is not None else None
                global_input_tokens = tensor_parallel.gather_from_sequence_parallel_region(
                    global_input_tokens,
                    group=parallel_state.get_expert_tensor_parallel_group(),
                    output_split_sizes=output_split_sizes,
                )

            recomputed_input = torch.index_select(
                global_input_tokens,
                0,
                ctx.level0_fc1_reorder_index,
            )
            recompute_ready_event = recompute_stream.record_event()

    return (
        permuted_local_tokens,
        global_input_tokens,
        recomputed_input,
        recompute_ready_event,
        input_handle,
    )


def _get_level0_recomputed_fc1_input(ctx, recompute_state, expert_input):
    """Finish the early recompute and return a replacement TE FC1 wgrad input."""
    if recompute_state is None:
        return None

    (
        permuted_local_tokens,
        global_input_tokens,
        recomputed_input,
        recompute_ready_event,
        input_handle,
    ) = recompute_state

    current_stream = torch.cuda.current_stream()
    current_stream.wait_event(recompute_ready_event)
    recomputed_input.record_stream(current_stream)

    if tuple(recomputed_input.shape) != tuple(expert_input.shape):
        raise RuntimeError(
            "AlltoAll-overlap level0 FC1 input recompute shape mismatch: "
            f"expected {tuple(expert_input.shape)}, got {tuple(recomputed_input.shape)}."
        )
    if recomputed_input.dtype != expert_input.dtype or recomputed_input.device != expert_input.device:
        raise RuntimeError(
            "AlltoAll-overlap level0 FC1 input recompute dtype/device mismatch: "
            f"expected ({expert_input.dtype}, {expert_input.device}), "
            f"got ({recomputed_input.dtype}, {recomputed_input.device})."
        )
    if expert_input.untyped_storage().size() != 0:
        raise RuntimeError("AlltoAll-overlap level0 expected the TE FC1 input storage to be released before backward.")

    # Keep the communication and postprocess buffers alive until their work is
    # ordered before the current stream. The caller then drops the state tuple.
    del permuted_local_tokens, global_input_tokens, input_handle
    return recomputed_input


class MoELayerOverlapAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, config, moe_layer: MoELayer, input_ids=None):
        if config.moe_zero_memory == "level1":
            raise RuntimeError(
                "MindSpeed alltoall overlap with TEGroupedMLP supports "
                "moe-zero-memory=disable or level0, but not level1."
            )
        ctx.config = config
        save_tensors = []
        ctx.input_shape = hidden_states.shape
        ctx.moe_layer = moe_layer
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        # router
        with torch.enable_grad():
            args = get_full_args()
            if getattr(args, "n_hash_layers", 0) >= 1:
                scores, routing_map = moe_layer.router(hidden_states, input_ids=input_ids)
            else:
                scores, routing_map = moe_layer.router(hidden_states)

        save_tensors.append(scores)
        scores = scores.detach()
        scores.requires_grad = True
        save_tensors.append(scores)
        moe_zero_memory = config.moe_zero_memory
        n_shared_experts = config.n_shared_experts
        ctx.moe_zero_memory = moe_zero_memory
        moe_shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
        group_limited_greedy = (
            hasattr(config, "moe_router_load_balancing_type")
            and config.moe_router_load_balancing_type == "group_limited_greedy"
        )
        ctx.shared_expert_overlap = moe_layer.shared_expert_overlap

        # if shared_expert_overlap, save share_experts graph separately for backward.
        if ctx.shared_expert_overlap:
            ctx.share_experts_graph_list = []
        else:
            ctx.share_experts_graph_list = None

        if n_shared_experts or moe_shared_expert_intermediate_size:
            ctx.shared_experts = moe_layer.shared_experts

        save_tensors.append(routing_map)

        (dispatched_input, tokens_per_expert, global_probs) = moe_layer.token_dispatcher.token_permutation(
            hidden_states, scores, routing_map, save_tensors, ctx
        )

        # forward_func returns its output followed by one detached value per input.
        # pylint: disable-next=unbalanced-tuple-unpacking
        (
            ((expert_output, mlp_bias), activation_checkpoint),
            expert_input_detach,
            _,
            expert_probs_detach,
            _,
        ) = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, global_probs, ctx))
        ctx.expert_activation_checkpoint = activation_checkpoint
        save_tensors.append(expert_output)
        save_tensors.append(expert_input_detach)
        save_tensors.append(expert_probs_detach)
        (output), expert_output_datach, *_ = forward_func(
            moe_layer.token_dispatcher.token_unpermutation,
            (expert_output, mlp_bias, ctx),
        )
        # unpermute1_input_detach
        save_tensors.append(expert_output_datach)
        expert_output_datach.untyped_storage().resize_(0)

        if group_limited_greedy:
            save_tensors.append(moe_layer.router.l_aux)
            moe_layer.router.l_aux = moe_layer.router.l_aux.detach()
            moe_layer.router.l_aux.requires_grad = True
            save_tensors.append(moe_layer.router.l_aux)
            with torch.enable_grad():
                save_to_aux_losses_tracker(
                    "load_balancing_loss",
                    moe_layer.router.l_aux,
                    moe_layer.layer_number,
                    moe_layer.config.num_layers,
                )
                save_to_aux_losses_tracker(
                    "load_balancing_expert_level_loss",
                    moe_layer.router.l_expert_aux / config.moe_aux_loss_coeff,
                    moe_layer.layer_number,
                    moe_layer.config.num_layers,
                )
                if hasattr(moe_layer.router, "l_device_aux"):
                    save_to_aux_losses_tracker(
                        "load_balancing_device_level_loss",
                        moe_layer.router.l_device_aux / config.moe_device_level_aux_loss_coeff,
                        moe_layer.layer_number,
                        moe_layer.config.num_layers,
                    )
                if hasattr(moe_layer.router, "l_comm_aux"):
                    save_to_aux_losses_tracker(
                        "load_balancing_comm_level_loss",
                        moe_layer.router.l_comm_aux / config.moe_comm_aux_loss_coeff,
                        moe_layer.layer_number,
                        moe_layer.config.num_layers,
                    )
                output = MoEAuxLossAutoScaler.apply(output, moe_layer.router.l_aux)
        else:
            save_tensors.append(None)
            save_tensors.append(None)

        # unpermute2_graph
        save_tensors.append(output)
        # detach_input
        save_tensors.append(hidden_states)

        ctx.output_splits = moe_layer.token_dispatcher.output_splits
        ctx.input_splits = moe_layer.token_dispatcher.input_splits
        ctx.router_topk = moe_layer.token_dispatcher.config.moe_router_topk
        ctx.output_splits_tp = moe_layer.token_dispatcher.output_splits_tp
        ctx.num_out_tokens = moe_layer.token_dispatcher.num_out_tokens
        ctx.level0_fc1_reorder_index = None
        if moe_zero_memory == "level0":
            ctx.level0_fc1_reorder_index = _build_level0_fc1_reorder_index(
                moe_layer.token_dispatcher.num_global_tokens_per_local_expert_cpu,
                moe_layer.token_dispatcher.sort_input_by_local_experts,
                dispatched_input.shape[0],
                dispatched_input.device,
            )

        # save shared_experts overlap backwards tensor.
        if moe_layer.shared_expert_overlap:
            ctx.save_for_backward(*ctx.share_experts_graph_list)

        output_sum = output.detach()
        ctx.save_for_backward(*save_tensors)
        return output_sum, mlp_bias

    @staticmethod
    def backward(ctx, *args):
        (
            route_graph,
            detach_scores,
            routing_map,
            permute1_graph,
            permuted_probs_graph,
            permute2_input_detach,
            permute2_graph,
            permute2_prob_detach,
            permute2_prob_graph,
            experts_graph,
            expert_input_detach,
            expert_probs_detach,
            unpermute1_input_detach,
            l_aux_graph,
            l_aux_detach,
            unpermute2_graph,
            detach_input,
        ) = ctx.saved_tensors

        # The recompute collectives are deliberately launched before unpermute
        # and expert backward. They overlap unpermute backward, activation
        # recompute, FC2 dgrad, activation backward, and FC1 dgrad.
        recompute_state = _start_level0_fc1_input_recompute(
            ctx,
            detach_input,
            routing_map,
        )

        # unpermute backward.
        unpermute2_graph.backward(args[0])
        unpermute2_graph = None

        # The shared FC2 branch is part of unpermute2_graph, so its detached
        # FC1-output gradient is ready now. Start the remaining shared-expert
        # backward immediately and let it overlap the main expert backward,
        # the backward AlltoAll collectives, and delayed wgrad computation.
        if ctx.shared_expert_overlap:
            cached_fc1_input_graph, cached_fc1_input_detach = ctx.share_experts_graph_list
            if cached_fc1_input_detach.grad is None:
                raise RuntimeError(
                    "AlltoAll-overlap shared-expert FC2 backward did not produce the FC1-output gradient."
                )
            with torch.cuda.stream(ctx.moe_layer.shared_experts.stream):
                backward_func(cached_fc1_input_graph, cached_fc1_input_detach.grad)
                # Avoid cached_fc1_input memory blast when TP=1.
                if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                    cached_fc1_input_graph.untyped_storage().resize_(0)
                    cached_fc1_input_detach.grad.untyped_storage().resize_(0)

        recomputed_fc2_input = None
        if ctx.expert_activation_checkpoint is not None:
            recomputed_fc2_input = ctx.expert_activation_checkpoint.recompute(True, return_output=True)
            ctx.expert_activation_checkpoint = None
            ctx.moe_layer.experts.linear_fc2.set_recomputed_input_for_delayed_wgrad(recomputed_fc2_input)

        # TE computes both dgrads now and queues both wgrads.
        backward_func(experts_graph, unpermute1_input_detach.grad)
        # FC2 delayed wgrad retains this tensor as grad_output. Releasing its
        # storage here leaves TE with valid shape metadata but no allocated
        # data when backward_dw() later launches the grouped wgrad GEMM.

        if expert_input_detach.grad is None or expert_probs_detach.grad is None:
            raise RuntimeError(
                "AlltoAll-overlap TE expert backward did not produce both "
                "expert input and router-probability gradients."
            )

        # Backpropagate TE's expert-input gradients through permutation 2,
        # then launch the original asynchronous backward AlltoAll collectives.
        backward_func(permute2_prob_graph, expert_probs_detach.grad, retain_graph=True)
        expert_probs_detach.grad.untyped_storage().resize_(0)
        permute2_prob_detach_grad = permute2_prob_detach.grad
        if parallel_state.get_expert_tensor_parallel_world_size() > 1:
            permute2_prob_detach_grad = tensor_parallel.reduce_scatter_to_sequence_parallel_region(
                permute2_prob_detach_grad,
                group=parallel_state.get_expert_tensor_parallel_group(),
                input_split_sizes=(ctx.output_splits_tp.tolist() if ctx.output_splits_tp is not None else None),
            )
        _, permute1_prob_backward_input, bw_permute1_prob_all2all_handle = async_all_to_all(
            permute2_prob_detach_grad,
            ctx.input_splits,
            ctx.output_splits,
            parallel_state.get_expert_model_parallel_group(),
        )

        backward_func(permute2_graph, expert_input_detach.grad)
        expert_input_detach.grad.untyped_storage().resize_(0)
        permute2_input_detach_grad = permute2_input_detach.grad
        if parallel_state.get_expert_tensor_parallel_world_size() > 1:
            permute2_input_detach_grad = tensor_parallel.reduce_scatter_to_sequence_parallel_region(
                permute2_input_detach_grad,
                group=parallel_state.get_expert_tensor_parallel_group(),
                input_split_sizes=(ctx.output_splits_tp.tolist() if ctx.output_splits_tp is not None else None),
            )
        _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            permute2_input_detach_grad,
            ctx.input_splits,
            ctx.output_splits,
            parallel_state.get_expert_model_parallel_group(),
        )

        # FC2 delayed wgrad is already self-contained. Run it while the two
        # backward AlltoAll collectives and the early FC1-input recompute are in flight.
        ctx.moe_layer.experts.linear_fc2.backward_dw()

        # FC1 delayed wgrad alone needs the released level0 input. Wait for the
        # early token-only recompute, install its replacement, and then run FC1.
        recomputed_fc1_input = _get_level0_recomputed_fc1_input(ctx, recompute_state, expert_input_detach)
        recompute_state = None
        ctx.level0_fc1_reorder_index = None
        if recomputed_fc1_input is not None:
            ctx.moe_layer.experts.linear_fc1.set_recomputed_input_for_delayed_wgrad(recomputed_fc1_input)
        ctx.moe_layer.experts.linear_fc1.backward_dw()
        unpermute1_input_detach.grad.untyped_storage().resize_(0)
        if ctx.moe_zero_memory == "level0":
            expert_input_detach.untyped_storage().resize_(0)
            recomputed_fc1_input.untyped_storage().resize_(0)
        if recomputed_fc2_input is not None:
            recomputed_fc2_input.untyped_storage().resize_(0)

        bw_permute1_ep_all2all_handle.wait()
        permute2_input_detach.grad.untyped_storage().resize_(0)

        bw_permute1_prob_all2all_handle.wait()
        # The shared and routed expert paths accumulate into the same detached
        # MoE input. Join the shared stream before the default stream starts
        # permutation/router accumulation to avoid cross-stream grad races.
        if ctx.shared_expert_overlap:
            torch.cuda.current_stream().wait_stream(ctx.moe_layer.shared_experts.stream)

        # permute1_graph and permuted_probs_graph are in the same graph.
        torch.autograd.backward(
            [permute1_graph, permuted_probs_graph],
            grad_tensors=[
                permute1_backward_input,
                permute1_prob_backward_input,
            ],
        )
        permute1_backward_input.untyped_storage().resize_(0)
        permute1_prob_backward_input.untyped_storage().resize_(0)

        router_graphs = [route_graph]
        router_grads = [detach_scores.grad]
        if l_aux_graph is not None:
            router_graphs.insert(0, l_aux_graph)
            router_grads.insert(0, l_aux_detach.grad)
        torch.autograd.backward(router_graphs, grad_tensors=router_grads)
        route_graph = None
        grad_output = detach_input.grad
        return grad_output, None, None, None
