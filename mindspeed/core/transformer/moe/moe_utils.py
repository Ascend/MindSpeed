# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from megatron.core.transformer.moe.moe_utils import permute_with_padded_tokens, unpermute_with_padded_tokens
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import (aggregate_aux_losses_tracker_across_pipeline_parallel,
                                                     clear_aux_losses_tracker, get_aux_losses_tracker)


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    # sorted_indices = torch.argsort(flatten_indices, stable=True)  # argsort int64 will be run on host cpu
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


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

    aggregate_aux_losses_tracker_across_pipeline_parallel()
    if writer is not None:
        aux_losses = {k: v.float() * loss_scale for k, v in get_aux_losses_tracker().items()}
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
