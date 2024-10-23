from typing import Optional
from functools import partial
import torch
from torch.library import impl

from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.op_builder.gmm_builder import GMMOpBuilderPublic, GroupedMatmul, fill_empty_tensor
from mindspeed.op_builder.gmm_builder import ge, Tensor, TensorSpec, DataType, register_fx_node_ge_converter


class WeightQuantGMMOpBuilder(GMMOpBuilderPublic):
    OP_NAME = "weight_quant_grouped_matmul"
    OP_PROTO = (
        "npu_weight_quant_gmm(Tensor x, Tensor weight, Tensor antiquant_scale, *, Tensor? antiquant_offset=None, \
         Tensor? bias=None, Tensor? group_list=None, int? group_list_type=0, int? act_type=0) -> Tensor"
    )

    def __init__(self):
        super(WeightQuantGMMOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/weight_quant_gmm.cpp']

    def register_op_ir(self):
        @impl(AS_LIBRARY, "npu_weight_quant_gmm", "Meta")
        def npu_weight_quant_gmm_forward(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None,
                                         group_list=None, group_list_type=0, act_type=0):
            BM = x.shape[0]
            N = weight.shape[-1]
            output_dtype = x.dtype
            return x.new_empty((BM, N), dtype=output_dtype)

        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_weight_quant_gmm.default)
        def conveter_npu_weight_quant_gmm(
            x: Tensor,
            weight: Tensor,
            antiquant_scale: Tensor,
            *,
            antiquant_offset: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            group_list: Optional[Tensor] = None,
            group_list_type: Optional[int] = 0,
            act_type: Optional[int] = 0,
            meta_outputs: TensorSpec = None,
        ):
            x_dtype = x.dtype
            if bias is None:
                if x_dtype == DataType.DT_BF16:
                    bias = fill_empty_tensor(DataType.DT_FLOAT)
                elif x_dtype == DataType.DT_FLOAT16:
                    bias = fill_empty_tensor(DataType.DT_FLOAT16)
            antiquant_offset = antiquant_offset or fill_empty_tensor(antiquant_scale.dtype)
            scale = fill_empty_tensor(DataType.DT_UINT64)
            offset = fill_empty_tensor(DataType.DT_FLOAT)


            return GroupedMatmul([x], [weight], [bias], [scale], [offset], [antiquant_scale], [antiquant_offset],
                                 group_list, split_item=3, group_type=0,
                                 group_list_type=group_list_type, act_type=act_type)[0]
