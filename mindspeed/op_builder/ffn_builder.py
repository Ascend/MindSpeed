# Copyright (c) 2024, Huawei Technologies.


from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

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


class FFNOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_ffn"
    OP_PROTO = "npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, Tensor? expert_tokens=None, \
        Tensor? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None, \
        Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None, \
        Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None, Tensor? antiquant_offset2=None, \
        int? inner_precise=None, ScalarType? output_dtype=None) -> Tensor"

    def __init__(self):
        super(FFNOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/ffn.cpp']

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
        @impl(AS_LIBRARY, "npu_ffn", "Meta")
        def npu_ffn_forward(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None,
                            bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None,
                            antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None,
                            antiquant_offset2=None, inner_precise=0, output_dtype=None):
            dim_list = []
            for i in range(0, x.dim() - 1):
                dim_list.append(x.size(i))
            dim_list.append(weight2.size(weight2.dim() - 1))
            if x.dtype == torch.int8:
                if output_dtype is not None and output_dtype == torch.bfloat16:
                    return x.new_empty(tuple(dim_list), dtype=torch.bfloat16)
                else:
                    return x.new_empty(tuple(dim_list), dtype=torch.float16)
            else:
                return x.new_empty(tuple(dim_list))

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_ffn.default)
        def convert_npu_ffn(
            x: Tensor,
            weight1: Tensor,
            weight2: Tensor,
            activation: str,
            *,
            expert_tokens: Optional[Tensor] = None,
            expert_tokens_index: Optional[Tensor] = None,
            bias1: Optional[Tensor] = None,
            bias2: Optional[Tensor] = None,
            scale: Optional[Tensor] = None,
            offset: Optional[Tensor] = None,
            deq_scale1: Optional[Tensor] = None,
            deq_scale2: Optional[Tensor] = None,
            antiquant_scale1: Optional[Tensor] = None,
            antiquant_scale2: Optional[Tensor] = None,
            antiquant_offset1: Optional[Tensor] = None,
            antiquant_offset2: Optional[Tensor] = None,
            inner_precise: Optional[int] = 0,
            output_dtype: Optional[int] = None,
            meta_outputs: TensorSpec = None
        ):
            '''"npu::npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, Tensor? expert_tokens=None,
                             Tensor? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None,
                             Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None,
                             Tensor? antiquant_scale1=None, Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None,
                             Tensor? antiquant_offset2=None, int? inner_precise=None, ScalarType? output_dtype=None)
                             -> Tensor
            "'''
            tokens_index_flag = False
            if expert_tokens is not None and expert_tokens_index is not None:
                raise ValueError("Cannot assign the value to expert_tokens and expert_tokens_index simultaneously!")
            elif expert_tokens_index is not None:
                tokens_index_flag = True
                expert_tokens = expert_tokens_index

            y_dtype = -1
            if x.dtype == DataType.DT_INT8 and output_dtype is not None:
                if output_dtype == torch.float16:
                    y_dtype = 0
                elif output_dtype == torch.bfloat16:
                    y_dtype = 1
                else:
                    raise NotImplementedError("In the quant scenario, output_dtype should be float16 or bfloat16,"
                        "otherwise it should be None!")

            return FFN(x, weight1, weight2, expert_tokens=expert_tokens, bias1=bias1, bias2=bias2, scale=scale,
                        offset=offset, deq_scale1=deq_scale1, deq_scale2=deq_scale2, antiquant_scale1=antiquant_scale1,
                        antiquant_scale2=antiquant_scale2, antiquant_offset1=antiquant_offset1,
                        antiquant_offset2=antiquant_offset2, activation=activation, inner_precise=inner_precise,
                        output_dtype=y_dtype, tokens_index_flag=tokens_index_flag)


FFN = None
if torch_npu_api_version == 2:
    def FFNV2(x: Tensor,
              weight1: Tensor,
              weight2: Tensor,
              expert_tokens: Optional[Tensor],
              bias1: Optional[Tensor],
              bias2: Optional[Tensor],
              scale: Optional[Tensor],
              offset: Optional[Tensor],
              deq_scale1: Optional[Tensor],
              deq_scale2: Optional[Tensor],
              antiquant_scale1: Optional[Tensor],
              antiquant_scale2: Optional[Tensor],
              antiquant_offset1: Optional[Tensor],
              antiquant_offset2: Optional[Tensor],
              *,
              activation: str,
              inner_precise: int = 0,
              output_dtype: int = -1,
              tokens_index_flag: bool = False):
        """REG_OP(FFN)\n
        .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))\n
        .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))\n
        .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))\n
        .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))\n
        .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))\n
        .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_BF16}))\n
        .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))\n
        .REQUIRED_ATTR(activation, String)\n
        .ATTR(inner_precise, Int, 0)\n
        .ATTR(output_dtype, Int, -1)\n
        .ATTR(tokens_index_flag, Bool, false)\n
        """

        y = torchair.ge.custom_op("FFN",
            inputs={
                "x": x,
                "weight1": weight1,
                "weight2": weight2,
                "expert_tokens": expert_tokens,
                "bias1": bias1,
                "bias2": bias2,
                "scale": scale,
                "offset": offset,
                "deq_scale1": deq_scale1,
                "deq_scale2": deq_scale2,
                "antiquant_scale1": antiquant_scale1,
                "antiquant_scale2": antiquant_scale2,
                "antiquant_offset1": antiquant_offset1,
                "antiquant_offset2": antiquant_offset2
            },
            attrs={
                "activation": ge.attr.Str(activation),
                "inner_precise": ge.attr.Int(inner_precise),
                "output_dtype": ge.attr.Int(output_dtype),
                "tokens_index_flag": ge.attr.Bool(tokens_index_flag)
            },
            outputs=[
                "y"
            ])

        return y
    FFN = FFNV2
elif torch_npu_api_version == 1:
    def FFNV1(x: Tensor,
              weight1: Tensor,
              weight2: Tensor,
              expert_tokens: Optional[Tensor],
              bias1: Optional[Tensor],
              bias2: Optional[Tensor],
              scale: Optional[Tensor],
              offset: Optional[Tensor],
              deq_scale1: Optional[Tensor],
              deq_scale2: Optional[Tensor],
              antiquant_scale1: Optional[Tensor],
              antiquant_scale2: Optional[Tensor],
              antiquant_offset1: Optional[Tensor],
              antiquant_offset2: Optional[Tensor],
              *,
              activation: str,
              inner_precise: int = 0,
              output_dtype: int = -1,
              tokens_index_flag: bool = False,
              dependencies=[],
              node_name=None):
        """REG_OP(FFN)\n
        .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))\n
        .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))\n
        .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))\n
        .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))\n
        .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))\n
        .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_BF16}))\n
        .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))\n
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))\n
        .REQUIRED_ATTR(activation, String)\n
        .ATTR(inner_precise, Int, 0)\n
        .ATTR(output_dtype, Int, -1)\n
        .ATTR(tokens_index_flag, Bool, false)\n
        """

        op = get_default_ge_graph().op.add()
        op.type = "FFN"
        op.name = next_unique_name(node_name, "FFN")

        # process dependices
        for dependency in dependencies:
            op.input.append(dependency.controller)

        # process inputs
        op.input.append(x.tensor)
        op.input_desc.add().CopyFrom(x.desc)
        op.input_desc[-1].name = "x"
        op.input.append(weight1.tensor)
        op.input_desc.add().CopyFrom(weight1.desc)
        op.input_desc[-1].name = "weight1"
        op.input.append(weight2.tensor)
        op.input_desc.add().CopyFrom(weight2.desc)
        op.input_desc[-1].name = "weight2"
        if expert_tokens is not None:
            op.input.append(expert_tokens.tensor)
            op.input_desc.add().CopyFrom(expert_tokens.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "expert_tokens"
        if bias1 is not None:
            op.input.append(bias1.tensor)
            op.input_desc.add().CopyFrom(bias1.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "bias1"
        if bias2 is not None:
            op.input.append(bias2.tensor)
            op.input_desc.add().CopyFrom(bias2.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "bias2"
        if scale is not None:
            op.input.append(scale.tensor)
            op.input_desc.add().CopyFrom(scale.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "scale"
        if offset is not None:
            op.input.append(offset.tensor)
            op.input_desc.add().CopyFrom(offset.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "offset"
        if deq_scale1 is not None:
            op.input.append(deq_scale1.tensor)
            op.input_desc.add().CopyFrom(deq_scale1.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "deq_scale1"
        if deq_scale2 is not None:
            op.input.append(deq_scale2.tensor)
            op.input_desc.add().CopyFrom(deq_scale2.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "deq_scale2"
        if antiquant_scale1 is not None:
            op.input.append(antiquant_scale1.tensor)
            op.input_desc.add().CopyFrom(antiquant_scale1.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "antiquant_scale1"
        if antiquant_scale2 is not None:
            op.input.append(antiquant_scale2.tensor)
            op.input_desc.add().CopyFrom(antiquant_scale2.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "antiquant_scale2"
        if antiquant_offset1 is not None:
            op.input.append(antiquant_offset1.tensor)
            op.input_desc.add().CopyFrom(antiquant_offset1.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "antiquant_offset1"
        if antiquant_offset2 is not None:
            op.input.append(antiquant_offset2.tensor)
            op.input_desc.add().CopyFrom(antiquant_offset2.desc)
        else:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "antiquant_offset2"

        # process attrs
        op.attr["activation"].s = compat_as_bytes(activation)
        op.attr["inner_precise"].i = inner_precise
        op.attr["output_dtype"].i = output_dtype
        op.attr["tokens_index_flag"].b = tokens_index_flag

        # process outputs
        output_index = 0
        op.output_desc.add().name = "y"
        y = Tensor(op, output_index)
        output_index += 1

        return y
    FFN = FFNV1
else:
    raise ValueError("touch_npu_api_version unsupport!")
