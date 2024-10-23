import torch
from torch.library import impl
from mindspeed.op_builder import QuantGMMOpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.ops import gmm

__all__ = ["npu_quant_gmm", "npu_quant_gmm_v2"]


op_builder = QuantGMMOpBuilder()


@impl(AS_LIBRARY, "npu_quant_gmm", "PrivateUse1")
def _npu_quant_gmm(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                   group_list_type=0, output_dtype=None, act_type=0):
    bias = [] if bias is None else [bias]
    scale = [] if scale is None else [scale]
    offset = [] if offset is None else [offset]
    per_token_scale = [] if per_token_scale is None else [per_token_scale]
    if output_dtype is None or output_dtype == torch.bfloat16:
        output_dtype_value = 1
    elif output_dtype == torch.float16:
        output_dtype_value = 0
    elif output_dtype == torch.int8:
        output_dtype_value = -1
    else:
        raise ValueError(f"output_dtype should be int8, float16, bfloat16 or None, but got {output_dtype}")
    outputs = op_builder.load().npu_quant_gmm([x], [weight], scale, offset, per_token_scale, bias, group_list,
                                              group_list_type, output_dtype_value, act_type)
    return outputs[0]


def _npu_quant_gmm_common(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                          group_list_type=0, output_dtype=None, act_type=0):
    if x.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError(f"Quant gmm only accept quant case, but got x[{x.dtype}] weight[{weight.dtype}]")
    gmm.npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list,
                                   group_type=0, group_list_type=group_list_type)
    gmm.check_optional_tensor(scale, x.device, "scale")
    gmm.check_optional_tensor(offset, x.device, "offset")
    gmm.check_optional_tensor(per_token_scale, x.device, "per_token_scale")
    return torch.ops.mindspeed.npu_quant_gmm(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                             bias=bias, group_list=group_list, group_list_type=group_list_type,
                                             output_dtype=output_dtype, act_type=act_type)


def npu_quant_gmm(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                  output_dtype=None, act_type=0):
    return _npu_quant_gmm_common(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                 bias=bias, group_list=group_list, group_list_type=0, output_dtype=output_dtype,
                                 act_type=act_type)


def npu_quant_gmm_v2(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                     output_dtype=None, act_type=0):
    return _npu_quant_gmm_common(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                 bias=bias, group_list=group_list, group_list_type=1, output_dtype=output_dtype,
                                 act_type=act_type)
