# Copyright (c) 2024, Huawei Technologies.


from typing import List, Optional

from mindspeed.op_builder.builder import MindSpeedOpBuilder, AS_LIBRARY
from mindspeed.op_builder.npu_mm_all_reduce_add_rms_norm_builder import MatmulAllReduceAddRmsNorm, CheckDtype

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


class InplaceMatmulAllReduceAddRmsNormOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_mm_all_reduce_add_rms_norm_"
    OP_PROTO = "npu_mm_all_reduce_add_rms_norm_(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, \
        str hcom, *, str reduce_op='sum', float epsilon=1e-06, Tensor? bias=None, Tensor? antiquant_scale=None, \
        Tensor? antiquant_offset=None, Tensor? dequant_scale=None, int antiquant_group_size=0, int comm_turn=0) \
        -> (Tensor, Tensor)"

    def __init__(self):
        super(InplaceMatmulAllReduceAddRmsNormOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/npu_mm_all_reduce_add_rms_norm_.cpp']

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
        @impl(AS_LIBRARY, "npu_mm_all_reduce_add_rms_norm_", "Meta")
        def npu_inplace_mm_all_reduce_add_rms_norm_forward(
                x1, x2, residual, gamma, hcom, reduce_op='sum', epsilon=1e-6,
                bias=None, antiquant_scale=None, antiquant_offset=None,
                dequant_scale=None, antiquant_group_size=0, comm_turn=0):
            return (torch.empty_like(residual, dtype=residual.dtype),
                    torch.empty_like(residual, dtype=residual.dtype))

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_mm_all_reduce_add_rms_norm_.default)
        def convert_npu_mm_all_reduce_add_rms_norm_(
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
            '''npu_mm_all_reduce_add_rms_norm_(Tensor x1, Tensor x2, Tensor(a!) residual, Tensor gamma,
                str hcom, *, str reduce_op='sum', float epsilon=1e-06, Tensor? bias=None,
                Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? dequant_scale=None,
                int antiquant_group_size=0, int comm_turn=0) -> (Tensor(a!), Tensor)'''
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
