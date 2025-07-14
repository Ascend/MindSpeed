# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from megatron.core.transformer.module import conversion_helper


def fp32_to_float16(val, float16_convertor):
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if val_typecheck is not None and val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)