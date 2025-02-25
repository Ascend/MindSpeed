# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from typing import Any, Callable

import torch

from megatron.core import ModelParallelConfig
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.utils import divide
from mindspeed.core.tensor_parallel.comm_group_api import CollectiveCommIntf
from mindspeed.core.tensor_parallel.comm_group_api import OverlapCollectiveIntf
from mindspeed.core.tensor_parallel.layers import _initialize_affine_weight_cpu_2d
from mindspeed.core.tensor_parallel.tp_2d.linear_2d_moe_split_along_first_dim import MoELinear2DFC1, MoELinear2DFC2
from mindspeed.core.tensor_parallel.tp_2d.linear_2d_split_along_first_dim import Linear2DSplitAlongFirstDim


class ParallelLinear2D(torch.nn.Module):
    """Linear2D layer with row and column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: If True, do not add the bias term, instead
                       return it to be added by the caller. This
                       enables performance optimations where bias can
                       be fused with other elementwise operations.
        skip_weight_param_allocation: If True, weight parameter is not allocated and must be passed
                                      as a keyword argument `weight` during the forward pass. Note
                                      that this does not affect bias, which will be allocated if
                                      bias is True. Defaults to False.
        is_expert: If True, the layer is treated as an MoE expert layer.
        config: ModelParallelConfig object
        tp_comm_buffer_name: Communication buffer name is not used in
                             non-Transformer-Engine modules.
        partition_dim: divide with dim, column parallel set 0, row parallel set 1
        enable_backward_overlap_ag_with_matmul: enable overlap all-gather with matmul

    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        add_bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=True,
        skip_weight_param_allocation: bool = False,
        is_expert: bool = False,
        ag_comm_intf: CollectiveCommIntf = None,
        ag_sd_rcv_overlap_comm_intf: OverlapCollectiveIntf = None,
        rs_comm_intf: CollectiveCommIntf = None,
        rs_sd_rcv_overlap_comm_intf: OverlapCollectiveIntf = None,
        enable_overlap_ag_with_matmul=False,
        enable_overlap_matmul_with_rs=False,
        partition_dim: int = 0,
        enable_backward_overlap_ag_with_matmul=False,
    ):
        super().__init__()
        self.mp_config: ModelParallelConfig = config
        self.para_init_method = init_method
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.add_bias = add_bias
        self.input_size = input_size
        self.output_size = output_size
        self.ag_comm_intf = ag_comm_intf
        self.rs_comm_intf = rs_comm_intf
        self.ag_comm_world_sz = ag_comm_intf.get_comm_group_world_size()
        self.rs_comm_world_sz = rs_comm_intf.get_comm_group_world_size()
        # when AG comm group is small, do overlap AG with matmul.
        self.enable_overlap_ag_with_matmul = enable_overlap_ag_with_matmul
        self.enable_overlap_matmul_with_rs = enable_overlap_matmul_with_rs
        self.ag_overlap_comm_intf = ag_sd_rcv_overlap_comm_intf
        self.rs_sd_rcv_overlap_comm_intf = rs_sd_rcv_overlap_comm_intf

        if input_size % self.rs_comm_world_sz:
            raise AssertionError("input size should be divisible by tp-y")
        if output_size % self.ag_comm_world_sz:
            raise AssertionError("output size should be divisible by tp-x")

        self.input_size_per_partition = divide(input_size, self.rs_comm_world_sz)
        self.output_size_per_partition = divide(output_size, self.ag_comm_world_sz)
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.enable_backward_overlap_ag_with_matmul = enable_backward_overlap_ag_with_matmul
        if config.sequence_parallel:
            raise RuntimeError(
                "Nd_matmul cannot be used with sequence_parallel."
                "If you want to train long sequences, "
                "you can use ulysess or context_parallel that is compatible with nd_matmul."
            )
        self.partition_dim = partition_dim
        self.init_linear_weights()

    def init_linear_weights(self):
        init_with_cpu = self.mp_config.use_cpu_initialization
        device = None if init_with_cpu else torch.cuda.current_device()

        self.weight = torch.nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size_per_partition,
                device=device,
                dtype=self.mp_config.params_dtype,
            )
        )
        if self.add_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=self.mp_config.params_dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        if init_with_cpu and self.mp_config.perform_initialization:
            _initialize_affine_weight_cpu_2d(self.weight, self.partition_dim, stride=self.stride,
                                             return_master_weight=self.keep_master_weight_for_test,
                                             config=self.mp_config)
        elif self.mp_config.perform_initialization:
            _initialize_affine_weight_gpu(
                self.weight,
                self.para_init_method,
                partition_dim=self.partition_dim,
                stride=self.stride,
                expert_parallel=(self.is_expert and self.expert_parallel),
            )

        setattr(self.weight, "allreduce", not (self.is_expert and self.expert_parallel))

        if self.add_bias and self.mp_config.perform_initialization:
            with torch.no_grad():
                self.bias.zero_()

            setattr(self.bias, "allreduce", not (self.is_expert and self.expert_parallel))
            setattr(self.bias, "sequence_parallel", False)

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None

    def forward(self, activation_input):
        if self.is_expert:
            if self.partition_dim == 0:
                linear_func = MoELinear2DFC1
            else:
                linear_func = MoELinear2DFC2
        else:
            linear_func = Linear2DSplitAlongFirstDim
        matmul_output = linear_func.apply(
            activation_input,
            self.weight,
            self.bias,
            self.ag_comm_intf,
            self.ag_overlap_comm_intf,
            self.rs_comm_intf,
            self.rs_sd_rcv_overlap_comm_intf,
            self.enable_overlap_ag_with_matmul,
            self.enable_overlap_matmul_with_rs,
            self.gradient_accumulation_fusion,
            self.enable_backward_overlap_ag_with_matmul,
            self.partition_dim,
        )

        if not self.skip_bias_add:
            output = (matmul_output + self.bias) if self.bias is not None else matmul_output
            output_bias = None
        else:
            output = matmul_output
            output_bias = self.bias

        return output, output_bias
