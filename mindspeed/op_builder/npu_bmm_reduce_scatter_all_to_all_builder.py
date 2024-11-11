# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional
import torch
import torchair
from torch.library import Library, impl
from mindspeed.op_builder.builder import MindSpeedOpBuilder, AS_LIBRARY
from mindspeed.op_builder.npu_all_to_all_all_gather_bmm_builder import CheckDtype
torch_npu_api_version = None
try:
    from torchair import ge
    from torchair import register_fx_node_ge_converter
    from torchair.ge import Tensor, TensorSpec, DataType
except ImportError:
    ge, Tensor, TensorSpec, DataType = None, None, None, None
    from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    torch_npu_api_version = 1
else:
    torch_npu_api_version = 2


class BatchMatMulReduceScatterAlltoAllOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_bmm_reducescatter_alltoall"
    OP_PROTO = "npu_bmm_reducescatter_alltoall(Tensor x, Tensor weight, str group_ep, int group_ep_worldsize, \
        str group_tp, int group_tp_worldsize, *, Tensor? bias=None, int shard_type=0) -> Tensor"

    def __init__(self):
        super(BatchMatMulReduceScatterAlltoAllOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/npu_bmm_reduce_scatter_all_to_all.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        return args
    
    def register_op_ir(self):
        @impl(AS_LIBRARY, "npu_bmm_reducescatter_alltoall", "Meta")
        def npu_bmm_reducescatter_alltoall_forward(x, weight, group_ep, group_ep_worldsize,
                                                   group_tp, group_tp_worldsize, *, bias=None, shard_type=0):
            if group_ep_worldsize == 0:
                raise AssertionError('group_ep_worldsize can not be 0.')
            if group_tp_worldsize == 0:
                raise AssertionError('group_tp_worldsize can not be 0.')
            e = x.size(0) * group_ep_worldsize
            c = x.size(1) // group_ep_worldsize
            h = weight.size(2)

            if x.size(0) == 0:
                raise AssertionError('The first dim of x can not be 0.')
            if x.size(1) == 0:
                raise AssertionError('The second dim of x can not be 0.')
            if x.size(2) == 0:
                raise AssertionError('The last dim of x can not be 0.')
            if weight.size(0) == 0:
                raise AssertionError('The first dim of weight can not be 0.')
            if weight.size(1) == 0:
                raise AssertionError('The second dim of weight can not be 0.')
            if weight.size(2) == 0:
                raise AssertionError('The last dim of weight can not be 0.')

            if shard_type == 0:
                # shard in h dimensions
                h = h // group_tp_worldsize
            else:
                # shard in c dimensions
                c = c // group_tp_worldsize

            return x.new_empty((e, c, h))
        
        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_bmm_reducescatter_alltoall.default)
        def convert_npu_bmm_reducescatter_alltoall(x: Tensor,
                                                   weight: Tensor,
                                                   group_ep: str,
                                                   group_ep_worldsize: int,
                                                   group_tp: str,
                                                   group_tp_worldsize: int,
                                                   *,
                                                   bias: Optional[Tensor] = None,
                                                   shard_type: Optional[int] = 0,
                                                   meta_outputs: TensorSpec = None):
            if torch_npu_api_version != 2:
                raise ValueError(f"torch_npu_api_version {torch_npu_api_version} unsupport")
            CheckDtype(x, weight, bias)
            return BatchMatmulReduceScatterAlltoAll(x,
                                                    weight,
                                                    group_ep,
                                                    group_ep_worldsize,
                                                    group_tp,
                                                    group_tp_worldsize,
                                                    bias=bias,
                                                    shard_type=shard_type)


def BatchMatmulReduceScatterAlltoAll(x: Tensor,
                                     weight: Tensor,
                                     group_ep: str,
                                     group_ep_worldsize: int,
                                     group_tp: str,
                                     group_tp_worldsize: int,
                                     *,
                                     bias: Tensor = None,
                                     shard_type: int = 0):
    transpose_weight = False
    return torchair.ge.custom_op(
        "BatchMatMulReduceScatterAlltoAll",
        inputs={
            "x": x,
            "weight": weight,
            "bias": bias
        },
        attrs={
            "group_ep": ge.attr.Str(group_ep),
            "group_tp": ge.attr.Str(group_tp),
            "ep_world_size": ge.attr.Int(group_ep_worldsize),
            "tp_world_size": ge.attr.Int(group_tp_worldsize),
            "y_shard_type": ge.attr.Int(shard_type),
            "transpose_weight": ge.attr.Bool(transpose_weight)
        },
        outputs=["y"]
    )