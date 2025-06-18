"""Track moe metrics considering noop transformer situation.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

import logging
from typing import Callable, Optional, List, Union

import torch

logger = logging.getLogger(__name__)


def get_mean(
    values: torch.Tensor, num_layers: int, noop_layers: set = None
) -> torch.Tensor:
    """Calculate the mean of a tensor, excluding specified 'noop_layers'.

    Args:
        values (torch.Tensor): A one-dimensional tensor.
        num_layers (int): total number of layers.
        noop_layers (set): noop-layers index.

    Returns:
        float: The mean of the tensor,
            excluding the 'noop_layers' if specified.

    Notes:
        - If `noop_layers` is a set and is not empty,
        the mean is calculated by excluding these layers.
        - If `noop_layers` is empty or None,
        the mean is calculated directly from the tensor.
        - `num_layers` represents the total number of layers,
          used to adjust the mean calculation when excluding 'noop_layers'.
    """
    if isinstance(noop_layers, set) and noop_layers:
        return values.sum() / (num_layers - len(noop_layers))
    return values.mean()


def track_moe_metrics_impl(
    reduce_aux_losses_tracker_across_ranks: Callable,
    get_moe_layer_wise_logging_tracker: Callable,
    clear_aux_losses_tracker: Callable,
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
    force_initialize: bool = False,
    track_names: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
    moe_layer_freq: Optional[Union[int, List[int]]] = None,
    noop_layers: Optional[set] = None,
):
    """Track metrics of moe during training."""
    # Aux loss logging
    tracker = get_moe_layer_wise_logging_tracker()

    # Initialize the tracker if force_initialize is True
    if force_initialize:
        if track_names is not None:
            for key in track_names:
                if key not in tracker:
                    tracker[key] = {}
                    tracker[key]["values"] = torch.zeros(num_layers, device="cuda")
                    tracker[key]["reduce_group"] = None
                    tracker[key]["avg_group"] = None
    reduce_aux_losses_tracker_across_ranks(track_names)

    # Get number of MoE layers
    if moe_layer_freq is None:
        num_moe_layers = num_layers
    elif isinstance(moe_layer_freq, int):
        if not isinstance(num_layers, int):
            raise AssertionError("num_layer must be int!")
        moe_layer_pattern = [
            1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)
        ]
        num_moe_layers = sum(moe_layer_pattern)
    elif isinstance(moe_layer_freq, list):
        num_moe_layers = sum(moe_layer_freq)
    else:
        raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

    if writer is not None:
        aux_losses = {
            k: v["values"].float() * loss_scale
            for k, v in tracker.items()  # type: ignore
        }
        for name, loss_list in aux_losses.items():
            # adaptation for
            loss_list_mean = get_mean(
                values=loss_list, num_layers=num_moe_layers, noop_layers=noop_layers
            )
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

            # W&B logging lacks support
            # for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first,
            # then we can create a custom panel to manually group them
            # to a single plot.
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
