# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Callable

import torch

from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import (
    TPXCollectiveComm, TPXOverlapCollectiveComm,
    TPYCollectiveComm, TPYOverlapCollectiveComm
)
from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D


class MLP2D(torch.nn.Module):
    """MLP with tp-2d"""

    def __init__(self,
                 config,
                 _initialize_affine_weight_gpu: Callable = None,
                 **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        if not self.config.tp_2d:
            raise ValueError('MLP2D require `tp_2d`.')

        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = ParallelLinear2D(
            self.config.hidden_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            add_bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            ag_comm_intf=TPXCollectiveComm,
            ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
            rs_comm_intf=TPYCollectiveComm,
            rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
            enable_overlap_ag_with_matmul=False,
            enable_overlap_matmul_with_rs=self.config.enable_overlap_matmul_with_rs,
            partition_dim=0,
            enable_backward_overlap_ag_with_matmul=self.config.enable_backward_overlap_ag_with_matmul,
            _initialize_affine_weight_gpu=_initialize_affine_weight_gpu)
        self.linear_fc2 = ParallelLinear2D(
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            add_bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            ag_comm_intf=TPYCollectiveComm,
            ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
            rs_comm_intf=TPXCollectiveComm,
            rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
            enable_overlap_ag_with_matmul=self.config.enable_overlap_ag_with_matmul,
            enable_overlap_matmul_with_rs=False,
            partition_dim=1,
            enable_backward_overlap_ag_with_matmul=self.config.enable_backward_overlap_ag_with_matmul,
            _initialize_affine_weight_gpu=_initialize_affine_weight_gpu)


def mlp_init_2d(self,
                config,
                _initialize_affine_weight_gpu: Callable = None,
                **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config
    if not self.config.tp_2d:
        raise ValueError('MLP2D require `tp_2d`.')

    ffn_hidden_size = self.config.ffn_hidden_size
    if self.config.gated_linear_unit:
        ffn_hidden_size *= 2

    self.linear_fc1 = ParallelLinear2D(
        self.config.hidden_size,
        ffn_hidden_size,
        config=self.config,
        init_method=self.config.init_method,
        add_bias=self.config.add_bias_linear,
        skip_bias_add=True,
        is_expert=False,
        ag_comm_intf=TPXCollectiveComm,
        ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
        rs_comm_intf=TPYCollectiveComm,
        rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
        enable_overlap_ag_with_matmul=False,
        enable_overlap_matmul_with_rs=self.config.enable_overlap_matmul_with_rs,
        partition_dim=0,
        enable_backward_overlap_ag_with_matmul=self.config.enable_backward_overlap_ag_with_matmul,
        _initialize_affine_weight_gpu=_initialize_affine_weight_gpu)
    self.linear_fc2 = ParallelLinear2D(
        self.config.ffn_hidden_size,
        self.config.hidden_size,
        config=self.config,
        init_method=self.config.output_layer_init_method,
        add_bias=self.config.add_bias_linear,
        skip_bias_add=True,
        is_expert=False,
        ag_comm_intf=TPYCollectiveComm,
        ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
        rs_comm_intf=TPXCollectiveComm,
        rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
        enable_overlap_ag_with_matmul=self.config.enable_overlap_ag_with_matmul,
        enable_overlap_matmul_with_rs=False,
        partition_dim=1,
        enable_backward_overlap_ag_with_matmul=self.config.enable_backward_overlap_ag_with_matmul,
        _initialize_affine_weight_gpu=_initialize_affine_weight_gpu)
