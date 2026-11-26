# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
import torch_npu
from torch.library import impl
from torch._guards import TracingContext
from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.ops._gmm_graph import (
    Tensor,
    TensorSpec,
    GroupedMatmul,
    conveter_npu_gmm_param,
    _ge_gmm_gradient,
    register_fx_node_ge_converter,
)


__all__ = ["npu_gmm", "npu_gmm_v2"]


AS_LIBRARY.define(
    'npu_gmm.Tensor(Tensor original_weight, Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? '
    'group_list=None, int? group_type=0, bool? gemm_fusion=False) -> Tensor'
)
AS_LIBRARY.define(
    'npu_gmm.List(Tensor original_weight, Tensor x, Tensor weight, *, Tensor? bias=None, int[]? '
    'group_list=None, int? group_type=0, bool? gemm_fusion=False) -> Tensor'
)
AS_LIBRARY.define(
    '_npu_gmm_backward(Tensor dy, Tensor x, Tensor weight, Tensor? group_list, int group_list_type, bool '
    'need_x, bool need_weight) -> (Tensor?, Tensor?)'
)

AS_LIBRARY.define(
    'npu_gmm_v2.Tensor(Tensor original_weight, Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? '
    'group_list=None, int? group_type=0, bool? gemm_fusion=False) -> Tensor'
)


@impl(AS_LIBRARY, "_npu_gmm_backward", "Meta")
def backward_meta(dy, x, weight, group_list, group_list_type, need_x, need_weight):
    return x.new_empty(x.shape) if need_x else None, weight.new_empty(weight.shape) if need_weight else None


@register_fx_node_ge_converter(torch.ops.mindspeed._npu_gmm_backward.default)
def backward_converter(
    dy: Tensor,
    x: Tensor,
    weight: Tensor,
    group_list: Optional[Tensor],
    group_list_type: int,
    need_x: bool,
    need_weight: bool,
    meta_outputs=None,
):
    dx = _ge_gmm_gradient(dy, weight, group_list, group_list_type, 0, False, True) if need_x else None
    dw = _ge_gmm_gradient(x, dy, group_list, group_list_type, 2, True, False) if need_weight else None
    return dx, dw


@impl(AS_LIBRARY, "npu_gmm.Tensor", "Meta")
def npu_gmm_forward(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    BM = x.shape[0]
    N = weight.shape[-1]
    y = x.new_empty((BM, N), dtype=x.dtype)
    return y


@register_fx_node_ge_converter(torch.ops.mindspeed.npu_gmm.Tensor)
def conveter_npu_gmm(
    original_weight: Tensor,
    x: Tensor,
    weight: Tensor,
    *,
    bias: Optional[Tensor] = None,
    group_list: Optional[Tensor] = None,
    group_type: Optional[int] = 0,
    gemm_fusion: Optional[bool] = False,
    meta_outputs: TensorSpec = None,
):
    """npu_gmm(Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? group_list=None, int? group_type=0) -> Tensor"""
    result = conveter_npu_gmm_param(x, bias, group_type)

    return GroupedMatmul(
        [x],
        [weight],
        [result.bias],
        result.scale,
        result.offset,
        result.antiquant_scale,
        result.antiquant_offset,
        group_list,
        split_item=3,
        group_type=group_type,
        dtype=-1,
        transpose_weight=False,
        group_list_type=0,
    )[0]


@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "Meta")
def npu_gmm_v2_forward(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    BM = x.shape[0]
    N = weight.shape[-1]
    y = x.new_empty((BM, N), dtype=x.dtype)
    return y


@register_fx_node_ge_converter(torch.ops.mindspeed.npu_gmm_v2.Tensor)
def conveter_npu_gmm_v2(
    original_weight: Tensor,
    x: Tensor,
    weight: Tensor,
    *,
    bias: Optional[Tensor] = None,
    group_list: Optional[Tensor] = None,
    group_type: Optional[int] = 0,
    gemm_fusion: Optional[bool] = False,
    meta_outputs: TensorSpec = None,
):
    """npu_gmm_v2(Tensor x, Tensor weight, *, Tensor? bias=None, Tensor? group_list=None, int? group_type=0) -> Tensor"""
    result = conveter_npu_gmm_param(x, bias, group_type)

    return GroupedMatmul(
        [x],
        [weight],
        [result.bias],
        result.scale,
        result.offset,
        result.antiquant_scale,
        result.antiquant_offset,
        group_list,
        split_item=3,
        group_type=group_type,
        dtype=-1,
        transpose_weight=False,
        group_list_type=1,
    )[0]


def check_optional_tensor(tensor, device, name):
    if not isinstance(tensor, (torch.Tensor, type(None))):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)}.")
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at least two devices, "
            f"{device}(arg0) and {tensor.device}({name})!"
        )


def _grouped_matmul(x, weight, bias, group_list, group_type, group_list_type):
    return torch_npu.npu_grouped_matmul.default(
        [x],
        [weight],
        bias=[] if bias is None else [bias],
        group_list=group_list,
        split_item=3,
        group_type=group_type,
        group_list_type=group_list_type,
    )[0]


@torch.jit.script
def _scripted_gmm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    groups: Optional[torch.Tensor],
    group_list_type: int,
    need_x: bool,
    need_weight: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    dx: Optional[torch.Tensor] = None
    dw: Optional[torch.Tensor] = None
    if need_x:
        dx = torch.ops.npu.npu_grouped_matmul(
            [dy],
            [weight.transpose(-1, -2)],
            bias=[],
            group_list=groups,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
        )[0]
    if need_weight:
        if weight.stride(-2) == 1 and weight.stride(-1) == weight.size(-2):
            dw = torch.ops.npu.npu_grouped_matmul(
                [dy.contiguous().transpose(-1, -2)],
                [x],
                bias=[],
                group_list=groups,
                split_item=3,
                group_type=2,
                group_list_type=group_list_type,
            )[0].transpose(-1, -2)
        else:
            dw = torch.ops.npu.npu_grouped_matmul(
                [x.transpose(-1, -2)],
                [dy],
                bias=[],
                group_list=groups,
                split_item=3,
                group_type=2,
                group_list_type=group_list_type,
            )[0]
        if weight.dim() != 3:
            dw = dw.reshape(weight.shape)
    return dx, dw


@impl(AS_LIBRARY, "_npu_gmm_backward", "PrivateUse1")
def _npu_gmm_backward(dy, x, weight, group_list, group_list_type, need_x, need_weight):
    return _scripted_gmm_backward(x, weight, dy, group_list, group_list_type, need_x, need_weight)


def _fused_weight_gradient(x, grad_outputs, weight, original_weight, group_list, group_list_type):
    npu_groupmatmul_add_fp32(
        x,
        grad_outputs,
        group_list,
        original_weight.main_grad,
        group_list_type=group_list_type,
    )
    if not hasattr(original_weight, 'grad_added_to_main_grad'):
        return None
    factory = torch.zeros if getattr(original_weight, 'zero_out_wgrad', False) else torch.empty
    grad_weight = factory(weight.shape, dtype=x.dtype, device=weight.device, requires_grad=False)
    original_weight.grad_added_to_main_grad = True
    return grad_weight


class GMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, _ = group_args
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        if group_list_type not in (0, 1):
            raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")
        if gemm_fusion and (original_weight is None or not hasattr(original_weight, 'main_grad')):
            raise ValueError("gemm_fusion requires original_weight with a main_grad tensor.")
        if isinstance(group_list, list):
            group_list = torch.tensor(group_list, dtype=torch.int64, device=x.device)
        outputs = _grouped_matmul(x, weight, bias, group_list, group_type, group_list_type)
        ctx.save_for_backward(x, weight, group_list, original_weight)
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight, group_list, original_weight = ctx.saved_tensors
        # AOTAutograd traces backward outside Dynamo; its tracing context is still active.
        if torch.compiler.is_compiling() or TracingContext.try_get() is not None:
            dx, dw = torch.ops.mindspeed._npu_gmm_backward(
                grad_outputs,
                x,
                weight,
                group_list,
                ctx.group_list_type,
                ctx.needs_input_grad[1],
                ctx.needs_input_grad[2] and not ctx.gemm_fusion,
            )
        else:
            dx, dw = _scripted_gmm_backward(
                x,
                weight,
                grad_outputs,
                group_list,
                ctx.group_list_type,
                ctx.needs_input_grad[1],
                ctx.needs_input_grad[2] and not ctx.gemm_fusion,
            )
        if ctx.gemm_fusion:
            dw = _fused_weight_gradient(x, grad_outputs, weight, original_weight, group_list, ctx.group_list_type)
        return None, dx, dw, None, None


def npu_gmm_param_verification(x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"arg0 must be a torch.Tensor, got {type(x)}.")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"arg1 must be a torch.Tensor, got {type(weight)}.")
    if not isinstance(bias, (torch.Tensor, type(None))):
        raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}.")
    if group_list_type == 0:
        if not (
            isinstance(group_list, (torch.Tensor, type(None)))
            or (isinstance(group_list, list) and all(isinstance(x, int) for x in group_list))
        ):
            raise TypeError(f"group_list must be a List of int64, torch.Tensor or None, got {type(group_list)}.")
    else:
        if not (isinstance(group_list, (torch.Tensor, type(None)))):
            raise TypeError(f"group_list must be a torch.Tensor or None, got {type(group_list)}.")
    if isinstance(group_list, torch.Tensor):
        if len(group_list.shape) > 1:
            raise ValueError(
                f"If group_list is not None, it must be an one-dimensional tensor, "
                f"got dimension of group_list: {len(group_list.shape)}!"
            )
        if group_list.dtype != torch.int64:
            raise TypeError(
                f"group_list must be a List of int64, got group_list type: {type(group_list)}, "
                f"dtype: {group_list.dtype}!"
            )
    if not isinstance(group_type, (int, type(None))):
        raise TypeError(f"group_type must be an int or None, got {type(group_type)}.")
    # Ensure all tensors on the same device
    x_device = x.device
    device_warning = "Expected all tensors to be on the same device, but found at least two devices"
    if weight.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {weight.device}(arg1)!")
    if bias is not None and bias.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {bias.device}(bias)!")
    if isinstance(group_list, torch.Tensor) and group_list.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {group_list.device}(group_list)!")


def _npu_gmm_common(
    original_weight, x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0, gemm_fusion=False
):
    support_dtype = [torch.float16, torch.bfloat16, torch.float32]
    if weight.dtype not in support_dtype:
        raise TypeError(f"Only support non quant case, but got weight dtype {weight.dtype}.")
    npu_gmm_param_verification(
        x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=group_list_type
    )
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")
    if not torch.compiler.is_compiling():
        group_args = (group_list, group_type, gemm_fusion, group_list_type, 1)
        return GMMFunction.apply(original_weight, x, weight, bias, group_args)
    op = torch.ops.mindspeed.npu_gmm if group_list_type == 0 else torch.ops.mindspeed.npu_gmm_v2
    return op(
        original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion
    )


@impl(AS_LIBRARY, "npu_gmm.List", "AutogradPrivateUse1")
@impl(AS_LIBRARY, "npu_gmm.Tensor", "AutogradPrivateUse1")
@impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
@impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
def _npu_gmm(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    group_args = (group_list, group_type, gemm_fusion, 0, 1)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(
        original_weight,
        x,
        weight,
        bias=bias,
        group_list=group_list,
        group_type=group_type,
        group_list_type=0,
        gemm_fusion=gemm_fusion,
    )


@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "AutogradPrivateUse1")
@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
def _npu_gmm_v2(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    group_args = (group_list, group_type, gemm_fusion, 1, 1)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(
        original_weight,
        x,
        weight,
        bias=bias,
        group_list=group_list,
        group_type=group_type,
        group_list_type=1,
        gemm_fusion=gemm_fusion,
    )
