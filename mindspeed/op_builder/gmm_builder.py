from typing import List, Optional
from collections import namedtuple

from mindspeed.op_builder.builder import MindSpeedOpBuilder, AS_LIBRARY

import torch
import torchair
from torch.library import Library, impl

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
    from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
    from torchair.ge_concrete_graph.ge_graph import get_invalid_desc
else:
    torch_npu_api_version = 2


gmm_param = namedtuple('gmm_param', ['bias', 'scale', 'offset', 'antiquant_scale', 'antiquant_offset'])


def conveter_npu_gmm_param(
    x: Tensor,
    bias: Tensor,
    group_type: int
):
    if group_type == 2:
        raise ValueError(f"GMM: graph mode does not support group_type 2!")
    x_dtype = x.dtype

    if torch_npu_api_version == 2:
        if bias is None:
            if x_dtype == DataType.DT_BF16:
                bias = Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_FLOAT))
            elif x_dtype == DataType.DT_UINT8:
                bias = Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_INT32))
            else:
                bias = Fill(ge.Const(0), ge.Cast(0., dst_type=x_dtype))

        scale = [Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_UINT64))]
        offset = [Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_FLOAT))]
        antiquant_scale = [Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_FLOAT16))]
        antiquant_offset = [Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_FLOAT16))]
        if x_dtype == DataType.DT_BF16:
            antiquant_scale = [Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_BF16))]
            antiquant_offset = [Fill(ge.Const(0), ge.Cast(0., dst_type=DataType.DT_BF16))]
    elif torch_npu_api_version == 1:
        if bias is None:
            if x_dtype == DataType.DT_BF16:
                bias = ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_FLOAT))
            elif x_dtype == DataType.DT_UINT8:
                bias = ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_INT32))
            else:
                bias = ge.Fill([0], ge.Cast(0., dst_type=x_dtype))

        scale = [ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_UINT64))]
        offset = [ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_FLOAT))]
        antiquant_scale = [ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_FLOAT16))]
        antiquant_offset = [ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_FLOAT16))]
        if x_dtype == DataType.DT_BF16:
            antiquant_scale = [ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_BF16))]
            antiquant_offset = [ge.Fill([0], ge.Cast(0., dst_type=DataType.DT_BF16))]

    return gmm_param(bias, scale, offset, antiquant_scale, antiquant_offset)


class GMMOpBuilderPublic(MindSpeedOpBuilder):
    def sources(self):
        return ['ops/csrc/cann/gmm.cpp']

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
        if self.TORCH_MAJOR >= 2 and self.TORCH_MINOR >= 1:
            cpp_std = " -std=c++17"
        else:
            cpp_std = " -std=c++14"
        args.append(cpp_std)
        return args


class GMMOpBuilder(GMMOpBuilderPublic):
    OP_NAME = "grouped_matmul"
    OP_PROTO = (
        "npu_gmm.List(Tensor x, Tensor weight, *, Tensor? bias=None, int[]? group_list=None, int? group_type=0) -> Tensor",
        "npu_gmm.Tensor(Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? group_list=None, int? group_type=0) -> Tensor"
    )
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

    def __init__(self):
        super(GMMOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def register_op_ir(self):
        @impl(AS_LIBRARY, "npu_gmm.Tensor", "Meta")
        def npu_gmm_forward(x, weight, *, bias=None, group_list=None, group_type=0):
            BM = x.shape[0]
            N = weight.shape[-1]
            y = x.new_empty((BM, N), dtype=x.dtype)
            return y

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_gmm.Tensor)
        def conveter_npu_gmm(
            x: Tensor,
            weight: Tensor,
            *,
            bias: Optional[Tensor] = None,
            group_list: Optional[Tensor] = None,
            group_type: Optional[int] = 0,
            meta_outputs: TensorSpec = None,
        ):
            """npu_gmm(Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? group_list=None, int? group_type=0) -> Tensor
            """
            result = conveter_npu_gmm_param(x, bias, group_type)

            return GroupedMatmul([x], [weight], [result.bias], result.scale, result.offset, result.antiquant_scale,
                                 result.antiquant_offset, group_list, size_of_y=1, split_item=3, group_type=group_type,
                                 dtype=-1, transpose_weight=False, group_list_type=0)[0]


class GMMV2OpBuilder(GMMOpBuilderPublic):
    OP_NAME = "grouped_matmul_v2"
    OP_PROTO = (
        "npu_gmm_v2.Tensor(Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? group_list=None, int? group_type=0) -> Tensor"
    )
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

    def __init__(self):
        super(GMMV2OpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def register_op_ir(self):
        @impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "Meta")
        def npu_gmm_v2_forward(x, weight, *, bias=None, group_list=None, group_type=0):
            BM = x.shape[0]
            N = weight.shape[-1]
            y = x.new_empty((BM, N), dtype=x.dtype)
            return y

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_gmm_v2.Tensor)
        def conveter_npu_gmm_v2(
            x: Tensor,
            weight: Tensor,
            *,
            bias: Optional[Tensor] = None,
            group_list: Optional[Tensor] = None,
            group_type: Optional[int] = 0,
            meta_outputs: TensorSpec = None,
        ):
            """npu_gmm_v2(Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? group_list=None, int? group_type=0) -> Tensor
            """
            result = conveter_npu_gmm_param(x, bias, group_type)

            return GroupedMatmul([x], [weight], [result.bias], result.scale, result.offset, result.antiquant_scale,
                                 result.antiquant_offset, group_list, size_of_y=1, split_item=3, group_type=group_type,
                                 dtype=-1, transpose_weight=False, group_list_type=1)[0]

if torch_npu_api_version == 2:
    def Fill(dims: Tensor, value: Tensor):
        """REG_OP(Fill)\n
        .INPUT(dims, TensorType::IndexNumberType())\n
        .INPUT(value, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16, DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))\n
        .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16, DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))\n
        """

        y = torchair.ge.custom_op("Fill",
            inputs={
                "dims":dims,
                "value":value
            },
            outputs=["y"]
        )

        # return outputs
        return y

GroupedMatmul = None
if torch_npu_api_version == 2:
    def GroupedMatmulV2(x: List[Tensor], weight: List[Tensor], bias: List[Tensor], scale: List[Tensor],
                        offset: List[Tensor], antiquant_scale: List[Tensor], antiquant_offset: List[Tensor],
                        group_list: Optional[Tensor], *, size_of_y: int, split_item: int = 0, group_type: int = -1,
                        dtype: int = 0, transpose_weight: bool = False, group_list_type: int = 0):
        """REG_OP(GroupedMatmul)\n
        .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))\n
        .DYNAMIC_INPUT(scale, TensorType({DT_UINT64}))\n
        .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))\n
        .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))\n
        .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
        .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))\n
        .ATTR(split_item, Int, 0)\n
        .ATTR(group_type, Int, -1)\n
        .ATTR(dtype, Int, 0)\n
        .ATTR(transpose_weight, Bool, false)\n
        .ATTR(group_list_type, Int, 0)\n
        """

        y = torchair.ge.custom_op("GroupedMatmul",
            inputs={
                "x":x,
                "weight":weight,
                "bias":bias,
                "scale":scale,
                "offset":offset,
                "antiquant_scale":antiquant_scale,
                "antiquant_offset":antiquant_offset,
                "group_list":group_list
            },
            attrs={
                "split_item":ge.attr.Int(split_item),
                "group_type":ge.attr.Int(group_type),
                "dtype":ge.attr.Int(dtype),
                "transpose_weight":ge.attr.Bool(transpose_weight),
                "group_list_type":ge.attr.Int(group_list_type)
            },
            outputs=[("y", 1)]
        )

        # return outputs
        return y
    GroupedMatmul = GroupedMatmulV2
elif torch_npu_api_version == 1:
    def GroupedMatmulV1(x: List[Tensor], weight: List[Tensor], bias: List[Tensor], scale: List[Tensor],
                        offset: List[Tensor], antiquant_scale: List[Tensor], antiquant_offset: List[Tensor],
                        group_list: Optional[Tensor], *, size_of_y: int, split_item: int = 0, group_type: int = -1,
                        dtype: int = 0, transpose_weight: bool = False, group_list_type: int = 0, dependencies=[],
                        node_name=None):
        """REG_OP(GroupedMatmul)\n
        .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))\n
        .DYNAMIC_INPUT(scale, TensorType({DT_UINT64}))\n
        .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))\n
        .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))\n
        .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
        .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))\n
        .ATTR(split_item, Int, 0)\n
        .ATTR(group_type, Int, -1)\n
        .ATTR(dtype, Int, 0)\n
        .ATTR(transpose_weight, Bool, false)\n
        .ATTR(group_list_type, Int, 0)\n
        """

        op = get_default_ge_graph().op.add()
        op.type = "GroupedMatmul"
        op.name = next_unique_name(node_name, "GroupedMatmul")

        # process dependices
        for dependency in dependencies:
            op.input.append(dependency.controller)

        # process inputs
        if not isinstance(x, (tuple, list)):
            raise AssertionError
        for i, v in enumerate(x):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "x" + str(i)
        if not isinstance(weight, (tuple, list)):
            raise AssertionError("weight must be a tuple or a list.")
        for i, v in enumerate(weight):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "weight" + str(i)
        if not isinstance(bias, (tuple, list)):
            raise AssertionError("bias must be a tuple or a list.")
        for i, v in enumerate(bias):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "bias" + str(i)
        if not isinstance(scale, (tuple, list)):
            raise AssertionError("scale must be a tuple or a list.")
        for i, v in enumerate(scale):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "scale" + str(i)
        if not isinstance(offset, (tuple, list)):
            raise AssertionError("offset must be a tuple or a list.")
        for i, v in enumerate(offset):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "offset" + str(i)
        if not isinstance(antiquant_scale, (tuple, list)):
            raise AssertionError("antiquant_scale must be a tuple or a list.")
        for i, v in enumerate(antiquant_scale):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "antiquant_scale" + str(i)
        if not isinstance(antiquant_offset, (tuple, list)):
            raise AssertionError("antiquant_offset must be a tuple or a list.")
        for i, v in enumerate(antiquant_offset):
            op.input.append(v.tensor)
            op.input_desc.add().CopyFrom(v.desc)
            op.input_desc[-1].name = "antiquant_offset" + str(i)
        if group_list is not None:
            op.input.append(group_list.tensor)
            op.input_desc.add().CopyFrom(group_list.desc)
            op.input_desc[-1].name = "group_list"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "group_list"

        # process attrs
        op.attr["split_item"].i = split_item
        op.attr["group_type"].i = group_type
        op.attr["dtype"].i = dtype
        op.attr["transpose_weight"].b = transpose_weight
        op.attr["group_list_type"].i = group_list_type

        # process outputs
        output_index = 0
        y = []
        for i in range(output_index, output_index + size_of_y):
            op.output_desc.add().name = "y" + str(i - output_index)
            y.append(Tensor(op, i))
        output_index += size_of_y

        # return outputs
        return y
    GroupedMatmul = GroupedMatmulV1
else:
    raise ValueError("touch_npu_api_version unsupport!")
