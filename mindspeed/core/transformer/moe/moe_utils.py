# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu
from megatron.core.transformer.moe.moe_utils import permute_with_padded_tokens, unpermute_with_padded_tokens
from megatron.training import get_args
from megatron.core import mpu, parallel_state
from megatron.core.transformer.moe.moe_utils import (reduce_aux_losses_tracker_across_ranks,
                                                     clear_aux_losses_tracker)


AG_TP_HIDDEN_STATUS = None
AG_SHARED_EXPERTS_INPUTS = []
GEMM_BACKWARD_NEED_TENSORS = None
RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = None
SWAP_STREAM = None
SWAP_STREAM2 = None
SWAP_TENSOR = None
MATMUL_OUTPUT_GRAD = None
UNPERMUTED_TOKENS = None
PERMUTE_WITH_EP_LOCAL_INPUT_TOKENS = None


def get_swap_stream():
    global SWAP_STREAM2
    if SWAP_STREAM2 is None:
        _ = torch_npu.npu.Stream(device=torch.npu.current_device())
        SWAP_STREAM2 = torch_npu.npu.Stream(device=torch.npu.current_device())
    stream = SWAP_STREAM2
    return stream


def set_swap_status(tensor):
    global SWAP_TENSOR
    SWAP_TENSOR = tensor


def get_swap_status():
    global SWAP_STREAM
    if SWAP_STREAM is None:
        SWAP_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
    global SWAP_TENSOR
    stream = SWAP_STREAM
    tensor = SWAP_TENSOR
    SWAP_TENSOR = None
    return stream, tensor


def set_prob_backward_need_tensors(matmul_output_grad, unpermuted_tokens):
    global MATMUL_OUTPUT_GRAD
    MATMUL_OUTPUT_GRAD = matmul_output_grad
    global UNPERMUTED_TOKENS
    UNPERMUTED_TOKENS = unpermuted_tokens


def get_prob_backward_need_tensors():
    global SWAP_STREAM2
    if SWAP_STREAM2 is None:
        _ = torch_npu.npu.Stream(device=torch.npu.current_device())
        SWAP_STREAM2 = torch_npu.npu.Stream(device=torch.npu.current_device())
    global MATMUL_OUTPUT_GRAD
    global UNPERMUTED_TOKENS
    stream = SWAP_STREAM2
    matmul_output_grad = MATMUL_OUTPUT_GRAD
    unpermuted_tokens = UNPERMUTED_TOKENS
    MATMUL_OUTPUT_GRAD = None
    UNPERMUTED_TOKENS = None
    return stream, matmul_output_grad, unpermuted_tokens


def set_ag_tp_hidden_status(_inputs):
    global AG_TP_HIDDEN_STATUS
    AG_TP_HIDDEN_STATUS = _inputs


def get_ag_tp_hidden_status():
    global AG_TP_HIDDEN_STATUS
    result = AG_TP_HIDDEN_STATUS
    AG_TP_HIDDEN_STATUS = None
    return result


def set_gemm_backward_need_tensors(_inputs):
    global GEMM_BACKWARD_NEED_TENSORS
    GEMM_BACKWARD_NEED_TENSORS = _inputs


def get_gemm_backward_need_tensors():
    global GEMM_BACKWARD_NEED_TENSORS
    result = GEMM_BACKWARD_NEED_TENSORS
    GEMM_BACKWARD_NEED_TENSORS = None
    return result


def set_permute_with_ep_local_input_tokens(_inputs):
    global PERMUTE_WITH_EP_LOCAL_INPUT_TOKENS
    PERMUTE_WITH_EP_LOCAL_INPUT_TOKENS = _inputs


def get_permute_with_ep_local_input_tokens():
    global PERMUTE_WITH_EP_LOCAL_INPUT_TOKENS
    result = PERMUTE_WITH_EP_LOCAL_INPUT_TOKENS
    PERMUTE_WITH_EP_LOCAL_INPUT_TOKENS = None
    return result


def set_rs_global_hidden_states_grad_with_handle(_inputs):
    global RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE
    RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = _inputs


def get_rs_global_hidden_states_grad_with_handle():
    global RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE
    result = RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE
    RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = None
    return result


ALL2ALL_EXPERTS_OUTPUT = None


def set_all2all_experts_output(_input):
    global ALL2ALL_EXPERTS_OUTPUT
    ALL2ALL_EXPERTS_OUTPUT = _input


def get_all2all_experts_output():
    global ALL2ALL_EXPERTS_OUTPUT
    result = ALL2ALL_EXPERTS_OUTPUT
    ALL2ALL_EXPERTS_OUTPUT = None
    return result


def only_recompute_activation(layer_number):
    args = get_args()
    vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    vpp_size = args.virtual_pipeline_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = args.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = args.num_layers // pp_size
    else:
        layer_per_chunk = args.num_layers

    if vpp_rank is None:
        vpp_rank = 0
    if vpp_size is None:
        vpp_size = 1
    recompute_priority = ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    moe_zero_memory_num_layers = args.moe_zero_memory_num_layers

    if moe_zero_memory_num_layers:
        if recompute_priority < moe_zero_memory_num_layers:
            return False
        else:
            return True
    else:
        return False


def forward_func(func, inputs):
    def detach_tensor(input_):
        if input_.requires_grad and input_.grad_fn is None:
            return input_
        else:
            new_input = input_.detach()
            new_input.requires_grad = True
        return new_input

    detach_inputs = []
    if isinstance(inputs, tuple):
        for input_ in inputs:
            if isinstance(input_, tuple):
                detach_input = []
                for i in input_:
                    if isinstance(i, torch.Tensor) and torch.is_floating_point(i):
                        detach_input.append(detach_tensor(i))
                    else:
                        detach_input.append(i)
                detach_inputs.append(tuple(detach_input))
            else:
                if isinstance(input_, torch.Tensor) and torch.is_floating_point(input_):
                    detach_input = detach_tensor(input_)
                else:
                    detach_input = input_
                detach_inputs.append(detach_input)
    elif isinstance(inputs, torch.Tensor):
        detach_inputs.append(detach_tensor(inputs))

    with torch.enable_grad():
        output = func(*detach_inputs)

    return output, *detach_inputs


def backward_func(func_tensor, gradinputs):
    if gradinputs is None or func_tensor.grad_fn is None:
        return
    if isinstance(gradinputs, torch.Tensor):
        func_tensor.backward(gradinputs)
    elif isinstance(gradinputs, tuple):
        func_tensor.backward(*gradinputs)


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    # previous use argsort, argsort int64 will be run on host cpu
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def permute_with_ep(tokens: torch.Tensor,
                    indices: torch.Tensor,
                    probs: torch.Tensor,
                    topk: int = 1,
                    gb_inputs_splits=None):
    if topk > 1:
        if indices.size(1) != topk:
            raise RuntimeError("indices.size(1) should be equal to topk")
    flatten_indices = indices.view(-1)
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    ep_rank = mpu.get_expert_model_parallel_rank()
    import numpy as np
    gb_inputs_splits_sum = np.cumsum(gb_inputs_splits)
    start = 0
    if ep_rank > 0:
        start = gb_inputs_splits_sum[ep_rank - 1]
    end = gb_inputs_splits_sum[ep_rank]
    result_indices = sorted_indices[start : end]
    permuted_tokens = tokens.index_select(0, result_indices // topk)
    flatten_probs = probs.view(-1)
    permuted_probs = flatten_probs.index_select(0, result_indices)
    return permuted_tokens, permuted_probs, result_indices


def unpermute_with_ep(
        unpermute_with_ep_input_tensors_list,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
        topk: int = 1,
):
    permuted_tokens, sorted_indices, permuted_probs = unpermute_with_ep_input_tensors_list
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if permuted_probs is not None:
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
    unpermuted_tokens = torch.zeros(restore_shape[0], permuted_tokens.size(-1),
                                    dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    sorted_indices = sorted_indices // topk
    unpermuted_tokens = unpermuted_tokens.scatter_add_(0,
                                                       sorted_indices.unsqueeze(1).expand(-1, permuted_tokens.shape[1]),
                                                       permuted_tokens)
    return unpermuted_tokens

	
def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def get_mean(tensor):
    """
    Calculate the mean of a tensor, excluding specified 'noop_layers'.

    Parameters:
        tensor (torch.Tensor): A one-dimensional tensor.

    Returns:
        float: The mean of the tensor, excluding the 'noop_layers' if specified.

    Notes:
        - If `args.noop_layers` is a set and is not empty, the mean is calculated by excluding these layers.
        - If `args.noop_layers` is empty or None, the mean is calculated directly from the tensor.
        - `args.num_layers` represents the total number of layers, used to adjust the mean calculation when
        excluding 'noop_layers'.
    """
    args = get_args()
    if hasattr(args, 'noop_layers') and isinstance(args.noop_layers, set) and len(args.noop_layers) > 0:
        return tensor.sum() / (args.num_layers - len(args.noop_layers))
    return tensor.mean()


def track_moe_metrics(
    loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None, per_layer_logging=False
):
    # Aux loss logging

    reduce_aux_losses_tracker_across_ranks()
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if writer is not None:
        aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            # adaptation for
            loss_list_mean = get_mean(loss_list)
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    # adaptation for loss_list.mean()
                    total_loss_dict[name] = loss_list_mean
                else:
                    # adaptation for loss_list.mean()
                    total_loss_dict[name] += loss_list_mean

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            # adaptation for loss_list.mean()
            writer.add_scalar(name, loss_list_mean, iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                # adaptation for loss_list.mean()
                wandb_writer.log({f"{name}": loss_list_mean}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {
                            f"moe/{name}_layer_{i}": loss
                            for i, loss in enumerate(loss_list.tolist())
                        },
                        iteration,
                    )

    clear_aux_losses_tracker()

