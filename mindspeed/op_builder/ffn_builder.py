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

from mindspeed.utils import print_rank_0_once
from mindspeed.op_builder.builder import MindSpeedOpBuilder, AS_LIBRARY

import torch
import torchair
from torch.library import Library, impl

_graph_mode_available = True
try:
    from torchair import ge
    from torchair.ge import Tensor, TensorSpec, DataType
    from torchair import register_fx_node_ge_converter
except ImportError:
    _graph_mode_available = False
    Tensor, TensorSpec, DataType = None, None, None
    print_rank_0_once("[WARNING] Custom ops not supported for graph mode due to mismatch of torch_npu version.")


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

        if not _graph_mode_available:
            return

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


def FFN(x: Tensor,
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
