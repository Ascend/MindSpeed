# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple
import torch
from torch import Tensor
from torch.nn import Module


def single_all_to_all(input_, scatter_idx, gather_idx, group):
    seq_world_size = torch.distributed.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input_.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input_.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    torch.distributed.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    # e.g., [cp, s/cp, b, n/cp, d] -> [s/cp, b, cp, n/cp, d]
    if scatter_idx < 2:
        output = output.transpose(0, 1).transpose(1, 2).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + [inp_shape[gather_idx] * seq_world_size, ] + inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: torch.distributed.ProcessGroup, input_: Tensor, scatter_idx: int,
                gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input_, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class UlyssesContextAttention(torch.nn.Module):
    """Initialization.
    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
            self,
            local_attention: Module,
            sequence_process_group: torch.distributed.ProcessGroup,
            scatter_idx: int = 2,
            gather_idx: int = 0,
    ) -> None:
        super(UlyssesContextAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # in shape : e.g.,  [s/p:h:]
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        # out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args)

        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output
