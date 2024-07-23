from typing import List, Optional
import torch
import torchair
from torchair import ge
from torch.library import Library, impl
from torchair.ge import Tensor, TensorSpec, DataType
from torchair import register_fx_node_ge_converter
from mindspeed.op_builder.builder import MindSpeedOpBuilder, AS_LIBRARY


class GMMOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "grouped_matmul"
    OP_PROTO = (
        "npu_gmm.List(Tensor x, Tensor weight, *, Tensor? bias, int[]? group_list, int? group_type) -> Tensor",
        "npu_gmm.Tensor(Tensor x, Tensor weight, *, Tensor? bias, Tensor? group_list, int? group_type) -> Tensor"
    )
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

    def __init__(self):
        super(GMMOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

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
            group_list: Tensor,
            *,
            bias: Optional[Tensor] = None,
            group_type: Optional[int] = 0,
            meta_outputs: TensorSpec = None,
        ):
            """npu_gmm(Tensor x, Tensor weight, Tensor group_list, *, Tensor? bias=None, int? group_type=0) -> Tensor
            """
            x_dtype = x.dtype

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

            return GroupedMatmul([x], [weight], [bias], scale, offset, antiquant_scale, antiquant_offset, group_list,
                                 size_of_y=1, split_item=3, group_type=group_type, dtype=-1, transpose_weight=False)[0]


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


def GroupedMatmul(x: List[Tensor], weight: List[Tensor], bias: List[Tensor], scale: List[Tensor],
                  offset: List[Tensor], antiquant_scale: List[Tensor], antiquant_offset: List[Tensor],
                  group_list: Optional[Tensor], *, size_of_y: int, split_item: int = 0, group_type: int = -1,
                  dtype: int = 0, transpose_weight: bool = False):
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
    .ATTR(group_type, Int, -1)
    .ATTR(dtype, Int, 0)\n
    .ATTR(transpose_weight, Bool, false)\n
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
            "transpose_weight":ge.attr.Bool(transpose_weight)
        },
        outputs=[("y", 1)]
    )

    # return outputs
    return y
