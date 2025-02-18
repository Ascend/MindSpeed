# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .ascend_turbo_cfg import ascend_turbo_cfg
from .mc2_linears_seq_parallel import (ColumnSeqParallelLinear, RowSeqParallelLinear,
                                       ColumnSeqParallelLinearWithFrozenWeight, RowSeqParallelLinearWithFrozenWeight)


def column_parallel_forward(self, input_, weight=None):
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass"
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight
    else:
        # Check the weight passed in is the correct shape
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)},"
                f"not {expected_shape} as expected"
            )

    bias = self.bias if not self.skip_bias_add else None

    if not weight.requires_grad:
        output = ColumnSeqParallelLinearWithFrozenWeight.apply(
            input_, weight, bias, ascend_turbo_cfg.get_group()
        )
    else:
        output = ColumnSeqParallelLinear.apply(
            input_, weight, bias, ascend_turbo_cfg.get_group()
        )

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def row_parallel_forward(self, input_):
    if not self.weight.requires_grad:
        output = RowSeqParallelLinearWithFrozenWeight.apply(
            input_, self.weight, None, ascend_turbo_cfg.get_group()
        )
    else:
        output = RowSeqParallelLinear.apply(
            input_, self.weight, None, ascend_turbo_cfg.get_group()
        )

    if not self.skip_bias_add:
        output = output + self.bias if self.bias is not None else output
        output_bias = None
    else:
        output_bias = self.bias

    return output, output_bias


def initialize_cfg_from_framework():
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size

    ascend_turbo_cfg.set_group(get_tensor_model_parallel_group)
    ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)

    ascend_turbo_cfg.set_column_parallel_linear(ColumnParallelLinear)
    ascend_turbo_cfg.set_row_parallel_linear(RowParallelLinear)

    ascend_turbo_cfg.parallel_linear_plugin(
        column_parallel_forward, row_parallel_forward
    )


def initialize_cfg_from_args(args):
    if not args.sequence_parallel or args.tensor_model_parallel_size == 1 or args.use_pipe_experts:
        return
    if args.use_nanopipe:
        return
    ascend_turbo_cfg.set_sequence_parallel(args.sequence_parallel)
    ascend_turbo_cfg.set_all_gather_recomputation(True)
    initialize_cfg_from_framework()
