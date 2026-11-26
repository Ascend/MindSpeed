# Copyright 2026 Huawei Technologies Co., Ltd.
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

from typing import List, Optional
from collections import namedtuple
import torchair


__all__ = [
    "Tensor",
    "TensorSpec",
    "DataType",
    "GroupedMatmul",
    "conveter_npu_gmm_param",
    "fill_empty_tensor",
    "register_fx_node_ge_converter",
]


torch_npu_api_version = None
try:
    from torchair import ge
    from torchair.ge import Tensor, TensorSpec, DataType
    from torchair import register_fx_node_ge_converter
except ImportError:
    torch_npu_api_version = 1
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
else:
    torch_npu_api_version = 2

if torch_npu_api_version == 2:

    def fill_empty_tensor(dtype):
        return Fill(ge.Const(0), ge.Cast(0.0, dst_type=dtype))
else:

    def fill_empty_tensor(dtype):
        return ge.Fill([0], ge.Cast(0.0, dst_type=dtype))


gmm_param = namedtuple('gmm_param', ['bias', 'scale', 'offset', 'antiquant_scale', 'antiquant_offset'])


def conveter_npu_gmm_param(x: Tensor, bias: Tensor, group_type: int):
    if group_type == 2:
        raise ValueError("GMM: graph mode does not support group_type 2!")
    x_dtype = x.dtype
    if bias is None:
        if x_dtype == DataType.DT_BF16:
            bias = fill_empty_tensor(DataType.DT_FLOAT)
        elif x_dtype == DataType.DT_UINT8:
            bias = fill_empty_tensor(DataType.DT_INT32)
        else:
            bias = fill_empty_tensor(x_dtype)
    scale = [fill_empty_tensor(DataType.DT_UINT64)]
    offset = [fill_empty_tensor(DataType.DT_FLOAT)]
    antiquant_scale = [fill_empty_tensor(DataType.DT_FLOAT16)]
    antiquant_offset = [fill_empty_tensor(DataType.DT_FLOAT16)]
    if x_dtype == DataType.DT_BF16:
        antiquant_scale = [fill_empty_tensor(DataType.DT_BF16)]
        antiquant_offset = [fill_empty_tensor(DataType.DT_BF16)]
    return gmm_param(bias, scale, offset, antiquant_scale, antiquant_offset)


def _ge_gmm_gradient(x, weight, group_list, group_list_type, group_type, transpose_x, transpose_weight):
    result = conveter_npu_gmm_param(x, None, 0)
    return GroupedMatmul(
        [x],
        [weight],
        [result.bias],
        result.scale,
        result.offset,
        result.antiquant_scale,
        result.antiquant_offset,
        group_list,
        split_item=3,
        group_type=group_type,
        dtype=-1,
        transpose_x=transpose_x,
        transpose_weight=transpose_weight,
        group_list_type=group_list_type,
    )[0]


if torch_npu_api_version == 2:

    def Fill(dims: Tensor, value: Tensor):
        """REG_OP(Fill)\n
        .INPUT(dims, TensorType::IndexNumberType())\n
        .INPUT(value, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16, DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))\n
        .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16, DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))\n
        """

        y = torchair.ge.custom_op("Fill", inputs={"dims": dims, "value": value}, outputs=["y"])

        return y


GroupedMatmul = None
if torch_npu_api_version == 2:

    def GroupedMatmulV2(
        x: List[Tensor],
        weight: List[Tensor],
        bias: List[Tensor],
        scale: List[Tensor],
        offset: List[Tensor],
        antiquant_scale: List[Tensor],
        antiquant_offset: List[Tensor],
        group_list: Optional[Tensor] = None,
        per_token_scale: Optional[Tensor] = None,
        *,
        split_item: int = 0,
        dtype: int = 0,
        transpose_weight: bool = False,
        transpose_x: bool = False,
        group_type: int = -1,
        group_list_type: int = 0,
        act_type: int = 0,
    ):
        """REG_OP(GroupedMatmul)\n
        .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))\n
        .DYNAMIC_INPUT(scale, TensorType({DT_UINT64}))\n
        .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))\n
        .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))\n
        .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT}))\n
        .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))\n
        .ATTR(split_item, Int, 0)\n
        .ATTR(dtype, Int, 0)\n
        .ATTR(transpose_weight, Bool, false)\n
        .ATTR(transpose_x, Bool, false)\n
        .ATTR(group_type, Int, -1)\n
        .ATTR(group_list_type, Int, 0)\n
        .ATTR(act_type, Int, 0)\n
        """

        y = torchair.ge.custom_op(
            "GroupedMatmul",
            inputs={
                "x": x,
                "weight": weight,
                "bias": bias,
                "scale": scale,
                "offset": offset,
                "antiquant_scale": antiquant_scale,
                "antiquant_offset": antiquant_offset,
                "group_list": group_list,
                "per_token_scale": per_token_scale,
            },
            attrs={
                "split_item": ge.attr.Int(split_item),
                "dtype": ge.attr.Int(dtype),
                "transpose_weight": ge.attr.Bool(transpose_weight),
                "transpose_x": ge.attr.Bool(transpose_x),
                "group_type": ge.attr.Int(group_type),
                "group_list_type": ge.attr.Int(group_list_type),
                "act_type": ge.attr.Int(act_type),
            },
            outputs=[("y", 1)],
        )

        return y

    GroupedMatmul = GroupedMatmulV2
elif torch_npu_api_version == 1:
    from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name, get_invalid_desc

    def GroupedMatmulV1(
        x: List[Tensor],
        weight: List[Tensor],
        bias: List[Tensor],
        scale: List[Tensor],
        offset: List[Tensor],
        antiquant_scale: List[Tensor],
        antiquant_offset: List[Tensor],
        group_list: Optional[Tensor] = None,
        per_token_scale: Optional[Tensor] = None,
        *,
        split_item: int = 0,
        dtype: int = 0,
        transpose_weight: bool = False,
        transpose_x: bool = False,
        group_type: int = -1,
        group_list_type: int = 0,
        act_type: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(GroupedMatmul)\n
        .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))\n
        .DYNAMIC_INPUT(scale, TensorType({DT_UINT64}))\n
        .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))\n
        .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))\n
        .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT}))\n
        .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))\n
        .ATTR(split_item, Int, 0)\n
        .ATTR(dtype, Int, 0)\n
        .ATTR(transpose_weight, Bool, false)\n
        .ATTR(transpose_x, Bool, false)\n
        .ATTR(group_type, Int, -1)\n
        .ATTR(group_list_type, Int, 0)\n
        .ATTR(act_type, Int, 0)\n
        """

        op = get_default_ge_graph().op.add()
        op.type = "GroupedMatmul"
        op.name = next_unique_name(node_name, "GroupedMatmul")

        # process dependices
        if dependencies is not None:
            for dependency in dependencies:
                op.input.append(dependency.controller)

        _append_grouped_matmul_inputs(
            op, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list, per_token_scale
        )
        _set_grouped_matmul_attrs(
            op, split_item, dtype, transpose_weight, transpose_x, group_type, group_list_type, act_type
        )

        # process outputs
        output_index = 0
        op.output_desc.add().name = "y" + str(output_index)
        y = [Tensor(op, output_index)]

        return y

    def _append_grouped_matmul_inputs(
        op, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list, per_token_scale
    ):
        for name, values in [
            ("x", x),
            ("weight", weight),
            ("bias", bias),
            ("scale", scale),
            ("offset", offset),
            ("antiquant_scale", antiquant_scale),
            ("antiquant_offset", antiquant_offset),
        ]:
            if not isinstance(values, (tuple, list)):
                if name == "x":
                    raise AssertionError
                raise AssertionError(f"{name} must be a tuple or a list.")
            for index, value in enumerate(values):
                op.input.append(value.tensor)
                op.input_desc.add().CopyFrom(value.desc)
                op.input_desc[-1].name = name + str(index)
        for name, value in [("group_list", group_list), ("per_token_scale", per_token_scale)]:
            if value is not None:
                op.input.append(value.tensor)
                op.input_desc.add().CopyFrom(value.desc)
                op.input_desc[-1].name = name
            else:
                op.input.append("")
                op.input_desc.add().CopyFrom(get_invalid_desc())

    def _set_grouped_matmul_attrs(
        op, split_item, dtype, transpose_weight, transpose_x, group_type, group_list_type, act_type
    ):
        for name, value in [
            ("split_item", split_item),
            ("dtype", dtype),
            ("group_type", group_type),
            ("group_list_type", group_list_type),
            ("act_type", act_type),
        ]:
            op.attr[name].i = value
        op.attr["transpose_weight"].b = transpose_weight
        op.attr["transpose_x"].b = transpose_x

    GroupedMatmul = GroupedMatmulV1
else:
    raise ValueError("touch_npu_api_version unsupport!")
