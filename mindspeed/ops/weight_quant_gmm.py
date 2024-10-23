import torch
from torch.library import impl
from mindspeed.op_builder import WeightQuantGMMOpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.ops import gmm

__all__ = ["npu_weight_quant_gmm", "npu_weight_quant_gmm_v2"]


op_builder = WeightQuantGMMOpBuilder()


@impl(AS_LIBRARY, "npu_weight_quant_gmm", "PrivateUse1")
def _npu_weight_quant_gmm(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None, group_list=None,
                          group_list_type=0, act_type=0):
    bias = [] if bias is None else [bias]
    antiquant_scale = [] if antiquant_scale is None else [antiquant_scale]
    antiquant_offset = [] if antiquant_offset is None else [antiquant_offset]
    outputs = op_builder.load().npu_weight_quant_gmm([x], [weight], antiquant_scale, antiquant_offset, bias, group_list,
                                                     group_list_type, act_type)
    return outputs[0]


def _npu_weight_quant_gmm_common(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None, group_list=None,
                                 group_list_type=0, act_type=0):
    if x.dtype != torch.float16 and x.dtype != torch.bfloat16:
        raise ValueError(f"Input x only accept float16/fp16, but got x[{x.dtype}]")
    if weight.dtype != torch.int8:
        raise ValueError(f"Weight only support int8, but got weight[{weight.dtype}]")
    gmm.npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list,
                                   group_type=0, group_list_type=group_list_type)
    gmm.check_optional_tensor(antiquant_scale, x.device, "antiquant_scale")
    gmm.check_optional_tensor(antiquant_offset, x.device, "antiquant_offset")
    return torch.ops.mindspeed.npu_weight_quant_gmm(x, weight, antiquant_scale, antiquant_offset=antiquant_offset,
                                                    bias=bias, group_list=group_list, group_list_type=group_list_type, 
                                                    act_type=act_type)


def npu_weight_quant_gmm(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None, 
                         group_list=None, act_type=0):
    return _npu_weight_quant_gmm_common(x, weight, antiquant_scale, antiquant_offset=antiquant_offset, bias=bias,
                                        group_list=group_list, group_list_type=0, act_type=act_type)


def npu_weight_quant_gmm_v2(x, weight, antiquant_scale, *, antiquant_offset=None, bias=None, 
                            group_list=None, act_type=0):
    return _npu_weight_quant_gmm_common(x, weight, antiquant_scale, antiquant_offset=antiquant_offset, bias=bias,
                                        group_list=group_list, group_list_type=1, act_type=act_type)
