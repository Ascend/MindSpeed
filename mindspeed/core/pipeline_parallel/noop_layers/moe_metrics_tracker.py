"""Track moe metrics considering noop transformer situation.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

import logging
from argparse import Namespace
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def get_mean(args: Namespace, values: torch.Tensor) -> torch.Tensor:
    """Calculate the mean of a tensor, excluding specified 'noop_layers'.

    Args:
        args (Namespace): Arguments from cli or configure file.
        values (torch.Tensor): A one-dimensional tensor.

    Returns:
        float: The mean of the tensor,
            excluding the 'noop_layers' if specified.

    Notes:
        - If `args.noop_layers` is a set and is not empty,
        the mean is calculated by excluding these layers.
        - If `args.noop_layers` is empty or None,
        the mean is calculated directly from the tensor.
        - `args.num_layers` represents the total number of layers,
          used to adjust the mean calculation when excluding 'noop_layers'.
    """
    if isinstance(args.noop_layers, set) and args.noop_layers:
        try:
            return values.sum() / (args.num_layers - len(args.noop_layers))
        except ZeroDivisionError as e:
            logging.warning(
                "args.num_layers is equal to length of args.noop_layers,"
                "args.num_layers: %s, length of args.noop_layers: %s",
                args.num_layers,
                args.noop_layers,
            )
            raise e
    return values.mean()


def track_moe_metrics(
    args: Namespace,
    reduce_aux_losses_tracker_across_ranks: Callable,
    get_moe_layer_wise_logging_tracker: Callable,
    clear_aux_losses_tracker: Callable,
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
):
    """Track metrics of moe during training."""
    # Aux loss logging
    reduce_aux_losses_tracker_across_ranks()
    tracker = get_moe_layer_wise_logging_tracker()
    if writer is not None:
        aux_losses = {
            k: v["values"].float() * loss_scale
            for k, v in tracker.items()  # type: ignore
        }
        for name, loss_list in aux_losses.items():
            # adaptation for
            loss_list_mean = get_mean(args=args, values=loss_list)
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
