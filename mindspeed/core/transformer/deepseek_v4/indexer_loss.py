# Copyright (c) 2025 NVIDIA CORPORATION.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from megatron.core import parallel_state


class DSAIndexerLossLoggingHelper:
    """MindSpeed-owned per-layer DSA loss accumulator."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(loss, layer_number, num_layers, reduce_group=None, avg_group=None):
        if layer_number is None:
            return
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker or tracker["values"].numel() != int(num_layers):
            tracker["values"] = torch.zeros(int(num_layers), device=loss.device)
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        if "values" in DSAIndexerLossLoggingHelper.tracker:
            DSAIndexerLossLoggingHelper.tracker["values"].zero_()
        DSAIndexerLossLoggingHelper.tracker["reduce_group"] = None
        DSAIndexerLossLoggingHelper.tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        values = tracker["values"]
        torch.distributed.all_reduce(values, group=parallel_state.get_pipeline_model_parallel_group())
        if tracker.get("reduce_group") is not None:
            torch.distributed.all_reduce(values, group=tracker["reduce_group"])
        if tracker.get("avg_group") is not None:
            torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_dsa_indexer_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        loss = tracker["values"].sum() / tracker["values"].numel() * loss_scale
        if total_loss_dict is not None:
            total_loss_dict["dsa_indexer_loss"] = loss
        if writer is not None:
            writer.add_scalar("dsa_indexer_loss", loss, iteration)
        if wandb_writer is not None:
            wandb_writer.log({"dsa_indexer_loss": loss}, iteration)
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def set_deepseek_v4_cp_indexer_loss_scale(scale):
    """Set the gradient scale used by MindSpeed's fused CP Indexer loss."""
    from mindspeed.core.context_parallel.deepseek_v4_context_parallel.ops.npu_sparse_flash_mla import (
        set_sparse_flash_mla_indexer_loss_scale,
    )

    set_sparse_flash_mla_indexer_loss_scale(scale)


def track_deepseek_v4_cp_indexer_metrics(
    loss_scale,
    iteration,
    writer,
    wandb_writer=None,
    total_loss_dict=None,
):
    """Reduce and report the MindSpeed-owned CP Indexer loss tracker."""
    return DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics(
        loss_scale,
        iteration,
        writer,
        wandb_writer,
        total_loss_dict,
    )
