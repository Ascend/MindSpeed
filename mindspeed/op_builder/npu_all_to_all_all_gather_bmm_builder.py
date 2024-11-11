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


class AllToAllAllGatherBatchMatMulOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_alltoall_allgather_bmm"
    OP_PROTO = "npu_alltoall_allgather_bmm(Tensor x, Tensor weight, \
        str group_ep, int group_ep_worldsize, \
        str group_tp, int group_tp_worldsize, \
        *, Tensor? bias=None, int shard_type=0, int act_type=0, \
        bool need_allgather_out=False, \
        bool need_activation_feature=False) -> (Tensor, Tensor, Tensor)"

    def __init__(self):
        super(AllToAllAllGatherBatchMatMulOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/npu_all_to_all_all_gather_bmm.cpp']

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
        @impl(AS_LIBRARY, "npu_alltoall_allgather_bmm", "Meta")
        def npu_alltoall_allgather_bmm_forward(x, weight,
                                               group_ep, group_ep_worldsize, group_tp, group_tp_worldsize,
                                               *, bias=None, shard_type=0, act_type=0,
                                               need_allgather_out=False, need_activation_feature=False):
            batch = weight.size(0)
            m = x.size(1) * group_ep_worldsize
            if shard_type == 1:
                m *= group_tp_worldsize
            n = weight.size(2)
            k = weight.size(1)
            
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

            empty_tensor = x.new_empty((0))
            return (x.new_empty((batch, m, n)),
                    x.new_empty((batch, m, k)) if need_allgather_out else empty_tensor,
                    x.new_empty((batch, m, n)) if need_activation_feature else empty_tensor)
        
        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_alltoall_allgather_bmm.default)
        def convert_npu_alltoall_allgather_bmm(
            x: Tensor,
            weight: Tensor,
            group_ep: str,
            group_ep_worldsize: int,
            group_tp: str,
            group_tp_worldsize: int,
            *,
            bias: Optional[Tensor] = None,
            shard_type: Optional[int] = 0,
            act_type: Optional[int] = 0,
            need_allgather_out: Optional[bool] = False,
            need_activation_feature: Optional[bool] = False,
            meta_outputs: List[TensorSpec] = None):
            '''"npu_alltoall_allgather_bmm(Tensor x, Tensor weight, str group_ep, str group_tp,
                int ep_world_size, int tp_world_size, *, Tensor? bias=None, int x_shard_type=0, int act_type=0,
                bool need_allgather_out=False, bool need_activation_feature=False) -> (Tensor, Tensor, Tensor)"'''
            if torch_npu_api_version != 2:
                raise ValueError(f"torch_npu_api_version {torch_npu_api_version} unsupport")
            CheckDtype(x, weight, bias)
            return AllToAllAllGatherBatchMatmul(x,
                                                weight,
                                                group_ep,
                                                group_ep_worldsize,
                                                group_tp,
                                                group_tp_worldsize,
                                                bias=bias,
                                                shard_type=shard_type,
                                                act_type=act_type,
                                                need_allgather_out=need_allgather_out,
                                                need_activation_feature=need_activation_feature)


def CheckDtype(x: Tensor, weight: Tensor, bias: Optional[Tensor]):
    if x.dtype != DataType.DT_BF16 and x.dtype != DataType.DT_FLOAT16:
        raise AssertionError(f'type of x must be DT_FLOAT16/DT_BF16, but got {GeDtypeToStr(x.dtype)}.')
    if weight.dtype != DataType.DT_BF16 and weight.dtype != DataType.DT_FLOAT16:
        raise AssertionError(f'type of weight must be DT_FLOAT16/DT_BF16, but got {GeDtypeToStr(weight.dtype)}.')
    if x.dtype != weight.dtype:
        raise AssertionError(f'type of x and weight must be same, but got x {GeDtypeToStr(x.dtype)} '\
                             f'weight {GeDtypeToStr(weight.dtype)}.')
    if bias is not None:
        if bias.dtype != DataType.DT_FLOAT16 and bias.dtype != DataType.DT_FLOAT:
            raise AssertionError(f'type of bias must DT_FLOAT16/DT_FLOAT32, but got {GeDtypeToStr(bias.dtype)}.')
        if x.dtype == DataType.DT_FLOAT16 and bias.dtype != DataType.DT_FLOAT16:
            raise AssertionError(f'type of bias must DT_FLOAT16 when x is DT_FLOAT16, '\
                                 f'but got {GeDtypeToStr(bias.dtype)}.')
        if x.dtype == DataType.DT_BF16 and bias.dtype != DataType.DT_FLOAT:
            raise AssertionError(f'type of bias must DT_FLOAT32 when x is DT_BF16, '\
                                 f'but got {GeDtypeToStr(bias.dtype)}.')


def GeDtypeToStr(ge_dtype: DataType):
    ge_datatype = {
        DataType.DT_FLOAT: 'DT_FLOAT32',
        DataType.DT_FLOAT16: 'DT_FLOAT16',
        DataType.DT_INT8: 'DT_INT8',
        DataType.DT_INT16: 'DT_INT16',
        DataType.DT_UINT16: 'DT_UINT16',
        DataType.DT_UINT8: 'DT_UINT8',
        DataType.DT_INT32: 'DT_INT32',
        DataType.DT_INT64: 'DT_INT64',
        DataType.DT_UINT32: 'DT_UINT32',
        DataType.DT_UINT64: 'DT_UINT64',
        DataType.DT_BOOL: 'DT_BOOL',
        DataType.DT_DOUBLE: 'DT_DOUBLE',
        DataType.DT_STRING: 'DT_STRING',
        DataType.DT_DUAL_SUB_INT8: 'DT_DUAL_SUB_INT8',
        DataType.DT_DUAL_SUB_UINT8: 'DT_DUAL_SUB_UINT8',
        DataType.DT_COMPLEX64: 'DT_COMPLEX64',
        DataType.DT_COMPLEX128: 'DT_COMPLEX128',
        DataType.DT_QINT8: 'DT_QINT8',
        DataType.DT_QINT16: 'DT_QINT16',
        DataType.DT_QINT32: 'DT_QINT32',
        DataType.DT_QUINT8: 'DT_QUINT8',
        DataType.DT_QUINT16: 'DT_QUINT16',
        DataType.DT_RESOURCE: 'DT_RESOURCE',
        DataType.DT_STRING_REF: 'DT_STRING_REF',
        DataType.DT_DUAL: 'DT_DUAL',
        DataType.DT_VARIANT: 'DT_VARIANT',
        DataType.DT_BF16: 'DT_BF16',
        DataType.DT_UNDEFINED: 'DT_UNDEFINED',
        DataType.DT_INT4: 'DT_INT4',
        DataType.DT_UINT1: 'DT_UINT1',
        DataType.DT_INT2: 'DT_INT2',
        DataType.DT_UINT2: 'DT_UINT2',
        DataType.DT_COMPLEX32: 'DT_COMPLEX32',
        DataType.DT_MAX: 'DT_MAX',
    }
    if ge_dtype in ge_datatype:
        return ge_datatype[ge_dtype]
    else:
        return 'unknown'


def AllToAllAllGatherBatchMatmul(
    x: Tensor,
    weight: Tensor,
    group_ep: str,
    group_ep_worldsize: int,
    group_tp: str,
    group_tp_worldsize: int,
    *,
    bias: Optional[Tensor] = None,
    shard_type: Optional[int] = 0,
    act_type: Optional[int] = 0,
    need_allgather_out: Optional[bool] = False,
    need_activation_feature: Optional[bool] = False):
    """REG_OP(AlltoAllAllGatherBatchMatMul)\n
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))\n
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))\n
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))\n
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_BF16}))\n
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16}))\n
    .OUTPUT(y3, TensorType({DT_FLOAT16, DT_BF16}))\n
    .REQUIRED_ATTR(group_ep, String)\n
    .REQUIRED_ATTR(group_tp, String)\n
    .REQUIRED_ATTR(ep_world_size, int)\n
    .REQUIRED_ATTR(tp_world_size, int)\n
    .ATTR(x_shard_type, Int, 1)\n
    .ATTR(act_type, Int, 0)\n
    .ATTR(need_allgather_out, Bool, False)\n
    .ATTR(need_activation_feature, Bool, False)\n
    .OP_END_FACTORY_REG(AlltoAllAllGatherBatchMatMul)
    use to construct Opdesc
    """
    transpose_weight = False
    return torchair.ge.custom_op(
        "AlltoAllAllGatherBatchMatMul",
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
            "x_shard_type": ge.attr.Int(shard_type),
            "act_type": ge.attr.Int(act_type),
            "transpose_weight": ge.attr.Bool(transpose_weight),
            "output_y2_flag": ge.attr.Bool(need_allgather_out),
            "output_y3_flag": ge.attr.Bool(need_activation_feature)
        },
        outputs=["y1", "y2", "y3"]
    )