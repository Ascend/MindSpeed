"""This module aims to make adaptor for megatron.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import Namespace
from typing import Optional

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_moe_layer_wise_logging_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker, reduce_aux_losses_tracker_across_ranks)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training import get_args
from megatron.training.training import \
    num_floating_point_operations as origin_flop_calculator
from torch import Tensor

from .calc_flop import calc_flop
from .moe_metrics_tracker import track_moe_metrics
from .transformer import build_with_noop_layers


class NoopTransformerLayer(MegatronModule):
    """Build noop transformer layer which do nothing but return the input.

    Args:
        MegatronModule (torch.nn.Module):
            Base Megatron module inhertied by all Models.
    """

    def __init__(self, layer_number: int):
        """init method

        Args:
            layer_number (int): the order of layer in transformer.
        """
        super().__init__(None)
        self.layer_number = layer_number

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Perform the forward pass through the noop transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params (InferenceParams, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        return hidden_states.clone(), context


def build_layers_adaptor(transformer: TransformerBlock):
    """Adaptor to build noop transformer layer by use megatron dependency.

    Args:
        transformer (TransformerBlock): A megatron `TransformerBlock` object.
    """
    build_with_noop_layers(
        transformer=transformer,
        noop_trasformer=NoopTransformerLayer,
        build_module=build_module,
    )


def calc_flop_adaptor(args: Namespace, batch_size: int) -> float:
    """Adaptor to calculate floating points of operations
    by using megatron dependency.

    Args:
        args (Namespace): Arguments from cli or configure file.
        batch_size (int): Batch size of training data set.

    Returns:
        float: Total number of  floating points of operations
        by considering noop transformer layers.
    """
    return calc_flop(
        func=origin_flop_calculator,
        args=args,
        batch_size=batch_size,
    )


def track_moe_metrics_adaptor(
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
):
    """Adaptor to track moe metrics of training.

    Args:
        loss_scale (float): A scale factor of loss.
        iteration (int): which training iteration step it is.
        writer (Optional[SummaryWriter]): A log writer in pytorch.
        wandb_writer (_type_, optional): A third part writer which
            support logging scalars simultaneously. Defaults to None.
        total_loss_dict (Optional[dict], optional): A dict to store total loss
            during moe training. Defaults to None.
        per_layer_logging (bool): A flag to decide if logging every layer.
            Defaults to False.
    """
    args = get_args()
    track_moe_metrics(
        args,
        reduce_aux_losses_tracker_across_ranks,
        get_moe_layer_wise_logging_tracker,
        clear_aux_losses_tracker,
        loss_scale=loss_scale,
        iteration=iteration,
        writer=writer,
        wandb_writer=wandb_writer,
        total_loss_dict=total_loss_dict,
        per_layer_logging=per_layer_logging,
    )
