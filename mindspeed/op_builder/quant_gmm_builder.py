from typing import Optional
from functools import partial
import torch
from torch.library import impl

from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.op_builder.gmm_builder import GMMOpBuilderPublic, GroupedMatmul, fill_empty_tensor
from mindspeed.op_builder.gmm_builder import ge, Tensor, TensorSpec, DataType, register_fx_node_ge_converter


class QuantGMMOpBuilder(GMMOpBuilderPublic):
    OP_NAME = "quant_grouped_matmul"
    OP_PROTO = (
        "npu_quant_gmm(Tensor x, Tensor weight, Tensor scale, *, Tensor? offset=None, Tensor? per_token_scale=None, \
         Tensor? bias=None, Tensor? group_list=None, int? group_list_type=0, ScalarType? output_dtype=None, \
         int? act_type=0) -> Tensor"
    )

    def __init__(self):
        super(QuantGMMOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/quant_gmm.cpp']

    def register_op_ir(self):
        @impl(AS_LIBRARY, "npu_quant_gmm", "Meta")
        def npu_quant_gmm_forward(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                                  group_list_type=0, output_dtype=None, act_type=0):
            BM = x.shape[0]
            N = weight.shape[-1]
            output_dtype = output_dtype or torch.float16
            return x.new_empty((BM, N), dtype=output_dtype)

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_quant_gmm.default)
        def conveter_npu_quant_gmm(
            x: Tensor,
            weight: Tensor,
            scale: Tensor,
            *,
            offset: Optional[Tensor] = None,
            per_token_scale: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            group_list: Optional[Tensor] = None,
            group_list_type: Optional[int] = 0,
            output_dtype: Optional[DataType] = None,
            act_type: Optional[int] = 0,
            meta_outputs: TensorSpec = None,
        ):
            bias = bias or fill_empty_tensor(DataType.DT_INT32)
            offset = offset or fill_empty_tensor(DataType.DT_FLOAT)
            antiquant_scale = fill_empty_tensor(DataType.DT_FLOAT16)
            antiquant_offset = fill_empty_tensor(DataType.DT_FLOAT16)

            y_dtype = 0
            if output_dtype is None or output_dtype == torch.float16:
                y_dtype = 0
            elif output_dtype == torch.bfloat16:
                y_dtype = 1
            elif output_dtype == torch.int8:
                raise ValueError("output_dtype not support int8 yet for graph mode")
            else:
                raise ValueError(f"output_dtype should be int8, float16 or bfloat16, "
                                 f"otherwise it should be None, but got {output_dtype}")

            return GroupedMatmul([x], [weight], [bias], [scale], [offset], [antiquant_scale], [antiquant_offset],
                                 group_list, per_token_scale, split_item=3, group_type=0,
                                 group_list_type=group_list_type, dtype=y_dtype, act_type=act_type)[0]
