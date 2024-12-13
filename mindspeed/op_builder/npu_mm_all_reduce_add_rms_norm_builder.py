# Copyright (c) 2024, Huawei Technologies.


from typing import List, Optional

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
    from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
    from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
    from torchair.ge_concrete_graph.ge_graph import get_invalid_desc
else:
    torch_npu_api_version = 2

DataType = dict(
    DT_FLOAT16=1,
    DT_INT8=2,
    DT_INT32=3,
    DT_INT64=9,
    DT_UINT64=10,
    DT_BF16=27,
)


class MatmulAllReduceAddRmsNormOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_mm_all_reduce_add_rms_norm"
    OP_PROTO = "npu_mm_all_reduce_add_rms_norm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, str hcom, *, \
        str reduce_op='sum', float epsilon=1e-06, Tensor? bias=None, Tensor? antiquant_scale=None, Tensor? \
        antiquant_offset=None, Tensor? dequant_scale=None, int antiquant_group_size=0, int comm_turn=0) \
        -> (Tensor, Tensor)"

    def __init__(self):
        super(MatmulAllReduceAddRmsNormOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/npu_mm_all_reduce_add_rms_norm.cpp']

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
        @impl(AS_LIBRARY, "npu_mm_all_reduce_add_rms_norm", "Meta")
        def npu_mm_all_reduce_add_rms_norm_forward(x1, x2, residual, gamma, hcom, reduce_op='sum', epsilon=1e-6,
                                                   bias=None, antiquant_scale=None, antiquant_offset=None,
                                                   dequant_scale=None, antiquant_group_size=0, comm_turn=0):
            return (torch.empty_like(residual, dtype=residual.dtype),
                    torch.empty_like(residual, dtype=residual.dtype))

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_mm_all_reduce_add_rms_norm.default)
        def convert_npu_mm_all_reduce_add_rms_norm(
                x1: Tensor,
                x2: Tensor,
                residual: Tensor,
                gamma: Tensor,
                hcom: str,
                *,
                reduce_op: str = 'sum',
                epsilon: float = 1e-6,
                bias: Optional[Tensor] = None,
                antiquant_scale: Optional[Tensor] = None,
                antiquant_offset: Optional[Tensor] = None,
                dequant_scale: Optional[Tensor] = None,
                antiquant_group_size: int = 0,
                comm_turn: int = 0,
                meta_outputs: List[TensorSpec] = None
        ):
            # transpose_x1 is set to False by default
            transpose_x1 = False
            transpose_x2 = False
            '''"npu_mm_all_reduce_add_rms_norm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, str hcom,
                    *, str reduce_op='sum', float epsilon=1e-06, Tensor? bias=None, Tensor? antiquant_scale=None,
                    Tensor? antiquant_offset=None, Tensor? dequant_scale=None, int antiquant_group_size=0,
                    int comm_turn=0) -> (Tensor, Tensor)"'''
            CheckDtype(x1, x2, bias=bias, residual=residual, gamma=gamma, antiquant_scale=antiquant_scale,
                       antiquant_offset=antiquant_offset, dequant_scale=dequant_scale)
            return MatmulAllReduceAddRmsNorm(x1,
                                             x2,
                                             bias=bias,
                                             residual=residual,
                                             gamma=gamma,
                                             antiquant_scale=antiquant_scale,
                                             antiquant_offset=antiquant_offset,
                                             dequant_scale=dequant_scale,
                                             group=hcom,
                                             reduce_op=reduce_op,
                                             is_trans_a=transpose_x1,
                                             is_trans_b=transpose_x2,
                                             comm_turn=comm_turn,
                                             antiquant_group_size=antiquant_group_size,
                                             epsilon=epsilon)


def CheckDtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor], residual: Tensor, gamma: Tensor,
               antiquant_scale: Optional[Tensor], antiquant_offset: Optional[Tensor],
               dequant_scale: Optional[Tensor]):
    if residual.dtype != gamma.dtype:
        raise AssertionError('type of residual and gamma must be same.')
    if x1.dtype in (DataType["DT_FLOAT16"], DataType["DT_BF16"]) and \
          x2.dtype in (DataType["DT_FLOAT16"], DataType["DT_BF16"]):
        if x2.dtype != x1.dtype:
            raise AssertionError('type of x1 and x2 must be same.')
        if bias is not None and bias.dtype != x1.dtype:
            raise AssertionError('type of x1 and bias must be same.')
        if residual.dtype != x1.dtype:
            raise AssertionError('type of x1 and residual must be same.')
    elif x1.dtype is DataType["DT_INT8"] and x2.dtype is DataType["DT_INT8"]:
        if bias is not None and bias.dtype != DataType["DT_INT32"]:
            raise AssertionError('type of bias must be int32.')
        if dequant_scale is None:
            raise AssertionError('dequant_scale must not be None.')
        if dequant_scale.dtype in (DataType["DT_INT64"], DataType["DT_UINT64"]):
            if residual.dtype != DataType["DT_FLOAT16"]:
                raise AssertionError('when dequant_scale is int64(uint64), residual type must be fp16.')
        elif dequant_scale.dtype is DataType["DT_BF16"]:
            if residual.dtype != DataType["DT_BF16"]:
                raise AssertionError('type of dequant_scale and residual should be bf16.')
        else:
            raise AssertionError('dequant_scale type must be int64, uint64 or bf16')
    elif x1.dtype in (DataType["DT_FLOAT16"], DataType["DT_BF16"]) and \
        x2.dtype is DataType["DT_INT8"]:
        if bias is not None and bias.dtype != x1.dtype:
            raise AssertionError('type of x1 and bias must be same.')
        if antiquant_scale is None:
            raise AssertionError('antiquant_scale must not be None.')
        if antiquant_scale.dtype != x1.dtype:
            raise AssertionError('type of x1 and antiquant_scale must be same.')
        if antiquant_offset is not None and antiquant_offset.dtype != antiquant_scale.dtype:
            raise AssertionError('type of antiquant_scale and antiquant_offset must be same.')
        if residual.dtype != x1.dtype:
            raise AssertionError('type of x1 and residual must be same.')
    else:
        raise AssertionError("the type of x1 and x2 should be suit the not quant scenario, "\
                    "dequant scenario, antiquant scenario.")

MatmulAllReduceAddRmsNorm = None
if torch_npu_api_version == 2:
    def MatmulAllReduceAddRmsNormV2(x1: Tensor,
                                x2: Tensor,
                                bias: Optional[Tensor],
                                residual: Tensor,
                                gamma: Tensor,
                                antiquant_scale: Optional[Tensor],
                                antiquant_offset: Optional[Tensor],
                                dequant_scale: Optional[Tensor],
                                *,
                                group: str,
                                reduce_op: str = "sum",
                                is_trans_a: bool = False,
                                is_trans_b: bool = False,
                                comm_turn: int = 0,
                                antiquant_group_size: int = 0,
                                epsilon: float = 0.000001):
        """REG_OP(MatmulAllReduceAddRmsNorm)\n
        .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4}))\n
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
        .REQUIRED_ATTR(group, String)\n
        .ATTR(reduce_op, String, "sum")\n
        .ATTR(is_trans_a, Bool, false)\n
        .ATTR(is_trans_b, Bool, false)\n
        .ATTR(comm_turn, Int, 0)\n
        .ATTR(antiquant_group_size, Int, 0)\n
        .ATTR(epsilon, Float, 1e-6)\n
        .OP_END_FACTORY_REG(MatmulAllReduceAddRmsNorm)
        """

        y, norm_out = torchair.ge.custom_op(
            "MatmulAllReduceAddRmsNorm",
            inputs={
                "x1" : x1,
                "x2" : x2,
                "bias" : bias,
                "residual" : residual,
                "gamma" : gamma,
                "antiquant_scale" : antiquant_scale,
                "antiquant_offset" : antiquant_offset,
                "dequant_scale" : dequant_scale,
                },
            attrs={
                "group" : ge.attr.Str(group),
                "reduce_op" : ge.attr.Str(reduce_op),
                "is_trans_a" : ge.attr.Bool(is_trans_a),
                "is_trans_b" : ge.attr.Bool(is_trans_b),
                "comm_turn" : ge.attr.Int(comm_turn),
                "antiquant_group_size" : ge.attr.Int(antiquant_group_size),
                "epsilon" : ge.attr.Float(epsilon),
            },
            outputs=[
                "y",
                "norm_out"
            ]
        )
        return y, norm_out
    MatmulAllReduceAddRmsNorm = MatmulAllReduceAddRmsNormV2
elif torch_npu_api_version == 1:
    def MatmulAllReduceAddRmsNormV1(x1: Tensor,
                              x2: Tensor,
                              bias: Optional[Tensor],
                              residual: Tensor,
                              gamma: Tensor,
                              antiquant_scale: Optional[Tensor],
                              antiquant_offset: Optional[Tensor],
                              dequant_scale: Optional[Tensor],
                              *,
                              group: str,
                              reduce_op: str = "sum",
                              is_trans_a: bool = False,
                              is_trans_b: bool = False,
                              comm_turn: int = 0,
                              antiquant_group_size: int = 0,
                              epsilon: float = 0.000001,
                              dependencies=None,
                              node_name=None):
        op = get_default_ge_graph().op.add()
        op.type = "MatmulAllReduceAddRmsNorm"
        op.name = next_unique_name(node_name, "MatmulAllReduceAddRmsNorm")

        # process dependices
        if dependencies is not None:
            for dependency in dependencies:
                op.input.append(dependency.controller)

        # process inputs
        op.input.append(x1.tensor)
        op.input_desc.add().CopyFrom(x1.desc)
        op.input_desc[-1].name = "x1"
        op.input.append(x2.tensor)
        op.input_desc.add().CopyFrom(x2.desc)
        op.input_desc[-1].name = "x2"
        if bias is not None:
            op.input.append(bias.tensor)
            op.input_desc.add().CopyFrom(bias.desc)
            op.input_desc[-1].name = "bias"
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "bias"
        op.input.append(residual.tensor)
        op.input_desc.add().CopyFrom(residual.desc)
        op.input_desc[-1].name = "residual"
        op.input.append(gamma.tensor)
        op.input_desc.add().CopyFrom(gamma.desc)
        op.input_desc[-1].name = "gamma"
        if antiquant_scale is not None:
            op.input.append(antiquant_scale.tensor)
            op.input_desc.add().CopyFrom(antiquant_scale.desc)
            op.input_desc[-1].name = "antiquant_scale"
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "antiquant_scale"
        if antiquant_offset is not None:
            op.input.append(antiquant_offset.tensor)
            op.input_desc.add().CopyFrom(antiquant_offset.desc)
            op.input_desc[-1].name = "antiquant_offset"
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "antiquant_offset"
        if dequant_scale is not None:
            op.input.append(dequant_scale.tensor)
            op.input_desc.add().CopyFrom(dequant_scale.desc)
            op.input_desc[-1].name = "dequant_scale"
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "dequant_scale"

        # process attrs
        op.attr["group"].s = compat_as_bytes(group)
        op.attr["reduce_op"].s = compat_as_bytes(reduce_op)
        op.attr["is_trans_a"].b = is_trans_a
        op.attr["is_trans_b"].b = is_trans_b
        op.attr["comm_turn"].i = comm_turn
        op.attr["antiquant_group_size"].i = antiquant_group_size
        op.attr["epsilon"].f = epsilon

        # process outputs
        output_index = 0
        op.output_desc.add().name = "y"
        y = Tensor(op, output_index)
        output_index += 1
        op.output_desc.add().name = "norm_out"
        norm_out = Tensor(op, output_index)
        output_index += 1

        return y, norm_out
    MatmulAllReduceAddRmsNorm = MatmulAllReduceAddRmsNormV1
else:
    raise ValueError("torch_npu_api_version unsupport")