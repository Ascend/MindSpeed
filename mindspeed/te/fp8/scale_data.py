import math
from typing import List

import torch
import torch_npu
from torch.autograd import Function

from mindspeed.te.fp8.constants import AMAX_COMPUTE_MAP, Format
from mindspeed.te.fp8.mxfp8_tensor import MXFP8Tensor
from mindspeed.te.fp8.recipes.recipe import RecipeConfig
from mindspeed.te.fp8 import cast_to_fp8, Float8Tensor


class ScaleData:
    def __init__(self, recipe_config: RecipeConfig, scale_shape: List[int] = None):
        if scale_shape is None:
            scale_shape = [1]
        self.config = recipe_config
        self.fp8_format = self.config.fp8_format.value
        self.fp8_max = self.config.fp8_format.value.max
        self.margin = self.config.margin
        self.amax_history_len = self.config.amax_history_len
        self.amax_history_current_len = 0
        self.scale_shape = scale_shape
        if self.config.amax_compute_algo not in AMAX_COMPUTE_MAP:
            raise AssertionError('Unsupported amax compute algo {}'.format(self.config.amax_compute_algo))
        self.amax_compute = AMAX_COMPUTE_MAP[self.config.amax_compute_algo]
        self.device = 'npu:{}'.format(torch.npu.current_device())
        self.ori_dtype = None
        self.scale = torch.ones(self.scale_shape, device=self.device)
        self.scale_inv = 1 / self.scale

        # 存储结构 -> tensor([amax_len, block])
        self.amax_history = torch.zeros([self.amax_history_len] + self.scale_shape, device=self.device)
        self.amax = torch.zeros(self.scale_shape, device=self.device)

    def append_amax(self, amax):
        if self.amax_history_current_len < self.amax_history_len:
            self.amax_history[self.amax_history_current_len, :].copy_(amax)
            self.amax_history_current_len += 1
        else:
            self.amax_history = self.amax_history.roll(-1, 1)
            self.amax_history[self.amax_history_len - 1, :].copy_(amax)

    def reduce_amax(self, group=None, async_op=False, amax=None):
        if group is not None and torch.distributed.get_world_size(group) > 1:
            # current策略传入amax，不需要使用amax history
            if amax is None:
                if self.amax_history_current_len < self.amax_history_len:
                    amax = self.amax_history[self.amax_history_current_len - 1, :]
                else:
                    amax = self.amax_history[self.amax_history_len - 1, :]
            handle = torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=group, async_op=async_op)
            return handle
        else:
            return None

    def update_scale(self, current_recipe=False):
        if not current_recipe:
            self.amax_compute(self.amax, self.amax_history)
        self.scale.copy_((self.fp8_max / self.amax) / (2 ** self.margin))
        self.scale_inv = 1 / self.scale

    def reshape_to_2D(self, input_tensor):
        # Convert the tensor shapes to 2D for execution compatibility
        if len(input_tensor.shape) != 2:
            input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
        return input_tensor

    def mxfp8_quantization(self, tensor: torch.Tensor, axis: int = -1):
        self.ori_dtype = tensor.dtype
        # 当前MXFP8的quantmatmul只支持MK*NK的形式，因此axis轴均为-1轴K轴
        # input、weight、grad三者均需要将转置和非转置全部处理，因此这里无传参控制是否采集转置数据
        y, mxscale = torch_npu.npu_dynamic_mx_quant(tensor, axis=axis, dst_type=self.fp8_format.dtype)
        y_t, mxscale_t = torch_npu.npu_dynamic_mx_quant(self.reshape_to_2D(tensor).t(), axis=axis, dst_type=self.fp8_format.dtype)
        return MXFP8Tensor(self.fp8_format, y, mxscale, y_t, mxscale_t, self.ori_dtype)

    def quantization(self, tensor: torch.Tensor):
        self.ori_dtype = tensor.dtype
        if self.scale.numel() == 1:
            quant_tensor = tensor * self.scale
        else:
            quant_tensor = tensor * self.scale[0][0]

        # cast to fp8
        fp8_tensor = cast_to_fp8(quant_tensor, self.fp8_format)
        return Float8Tensor(fp8_tensor, self.fp8_format, self.scale_inv, self.scale, self.ori_dtype)

    def dequantization(self, tensor: torch.Tensor):
        if self.scale.numel() == 1:
            dequant_tensor = tensor * self.scale_inv
        else:
            dequant_tensor = tensor * self.scale_inv[0][0]
        return dequant_tensor.to(dtype=self.ori_dtype)
