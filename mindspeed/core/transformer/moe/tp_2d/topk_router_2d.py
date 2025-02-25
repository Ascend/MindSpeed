# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func
from megatron.core.transformer.moe.moe_utils import z_loss_func
from megatron.core.transformer.moe.router import TopKRouter
from mindspeed.core.tensor_parallel_x_union_cp import TensorParallelXUnionCP
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.moe.utils import MoEAuxLossAutoScaler


class TopKRouter2D(TopKRouter):
    """Route each token to the top-k experts for 2-D tensor parallel."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        setattr(self.weight, 'sequence_parallel', False)
        setattr(self.weight, "2d_tp", True)

    def apply_load_balancing_loss(self, probs: torch.Tensor,
                                  num_local_tokens_per_expert: torch.Tensor,
                                  activation: torch.Tensor):
        """Applies auxiliary loss to the MoE layer.

                Args:
                    probs (torch.Tensor): The probs output by the router for each token, with shape:
                     [sb / (x * cp), E]
                    num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert
                    with shape: [E]
                    activation, ie top_k_probs (torch.Tensor): The activation tensor to attach the gradient function to.
                    with shape: [sb/(x*cp), topK]

                Returns:
                    torch.Tensor: The activation tensor with the attached gradient function with
                    shape: [sb/(x*cp), topK]
         """
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "allgather":
            tp_x_cp_group = TensorParallelXUnionCP()
            sequence_partition_group = tp_x_cp_group.group
        elif self.config.moe_token_dispatcher_type == "alltoall":
            tp_y_cp_group = TensorParallelYUnionCP()
            sequence_partition_group = tp_y_cp_group.group
            moe_aux_loss_coeff /= tp_y_cp_group.get_parallel_group_world_size()

        aux_loss = switch_load_balancing_loss_func(probs, num_local_tokens_per_expert, self.topk,
                                                   moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group)
        save_to_aux_losses_tracker("load_balancing_loss", aux_loss / moe_aux_loss_coeff,
                                        self.layer_number, self.config.num_layers, reduce_group=sequence_partition_group)

        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / TensorParallelXUnionCP().get_parallel_group_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss",
                z_loss / moe_z_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
        return logits
