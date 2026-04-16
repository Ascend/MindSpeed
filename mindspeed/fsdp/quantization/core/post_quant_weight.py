# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
from typing import Dict, Any, Dict

import torch


class PostQuantWeight(torch.Tensor):
    """
    Tensor subclass that carries quantized data, scale metadata, original dtype,
    and axiswise quantization layout for MX/FP8 linear forward and backward paths.
    """

    _weight_fwd: torch.Tensor
    _scale_fwd: torch.Tensor
    _weight_bwd: torch.Tensor
    _scale_bwd: torch.Tensor
    _orig_dtype: torch.dtype
    __slots__ = [
        "_weight_fwd",
        "_scale_fwd",
        "_weight_bwd",
        "_scale_bwd",
        "_orig_dtype",
    ]

    def __new__(
            cls,
            weight_fwd: Optional[torch.Tensor],
            scale_fwd: Optional[torch.Tensor],
            weight_bwd: Optional[torch.Tensor],
            scale_bwd: Optional[torch.Tensor],
            orig_dtype: torch.dtype,
    ):
        reference_tensor = weight_fwd if weight_fwd is not None else weight_bwd
        if reference_tensor is None:
            raise ValueError("At least one of data or data2 must be provided")

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            reference_tensor.size(),
            strides=reference_tensor.stride(),
            storage_offset=reference_tensor.storage_offset(),
            dtype=orig_dtype,
            layout=reference_tensor.layout,
            requires_grad=reference_tensor.requires_grad,
            device=reference_tensor.device,
        )

        self._weight_fwd = weight_fwd
        self._scale_fwd = scale_fwd
        self._weight_bwd = weight_bwd
        self._scale_bwd = scale_bwd
        self._orig_dtype = orig_dtype
        return self

    def __repr__(self):
        return "QuantTrainingTensor(" f"lp_dtype={self._weight_fwd.dtype}"

    def __tensor_flatten__(self):
        metadata = {
            "_orig_dtype": self._orig_dtype,
        }
        tensors = ["_weight_fwd", "_scale_fwd", "_weight_bwd", "_scale_bwd"]
        return tensors, metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        return PostQuantWeight(
            inner_tensors["_weight_fwd"],
            inner_tensors["_scale_fwd"],
            inner_tensors["_weight_bwd"],
            inner_tensors["_scale_bwd"],
            metadata["_orig_dtype"],
        )

    def to_original_precision(self):
        raise NotImplementedError()

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        def allowed_subclasses(type_):
            return issubclass(cls, type_) or issubclass(torch._subclasses.fake_tensor.FakeTensor, type_) or issubclass(
                torch._subclasses.functional_tensor.FunctionalTensor, type_)

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented
        if func in OPS_TABLE:
            return OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    __torch_function__ = torch._C._disabled_torch_function_impl


aten = torch.ops.aten
OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    def decorator(func):
        for op in aten_ops:
            if op in OPS_TABLE:
                raise RuntimeError(f"Float8 op {op} is already registered to {OPS_TABLE[op].__name__}")
            OPS_TABLE[op] = func
        return func

    return decorator


@implements(
    [
        aten.view.default,
        aten._unsafe_view.default,
        aten.as_strided.default,
        aten.clone.default,
        aten.slice.Tensor,
        aten.fill_.Scalar,
        aten.reshape.default,
    ]
)
def float8_desugar_op(aten_op, args, kwargs=None):
    arg0 = args[0]
    new_data_fwd, new_scale_fwd, new_data_bwd, new_scale_bwd = None, None, None, None

    if hasattr(arg0, "_weight_fwd") and arg0._weight_fwd is not None and arg0._weight_fwd.numel() > 0:
        new_data_fwd = aten_op(arg0._weight_fwd, *args[1:], **kwargs)
        new_scale_fwd = arg0._scale_fwd
    if hasattr(arg0, "_weight_bwd") and arg0._weight_bwd is not None and arg0._weight_bwd.numel() > 0:
        new_data_bwd = aten_op(arg0._weight_bwd, *args[1:], **kwargs)
        new_scale_bwd = arg0._scale_bwd

    return PostQuantWeight(
        new_data_fwd,
        new_scale_fwd,
        new_data_bwd,
        new_scale_bwd,
        args[0]._orig_dtype,
    )


@implements([aten.detach.default])
def float8_desugar_data_and_scale_op(aten_op, args, kwargs=None):
    arg0 = args[0]
    new_data_fwd, new_scale_fwd, new_data_bwd, new_scale_bwd = None, None, None, None

    if hasattr(arg0, "_weight_fwd") and arg0._weight_fwd is not None and arg0._weight_fwd.numel() > 0:
        new_data_fwd = aten_op(arg0._weight_fwd, *args[1:], **kwargs)
        new_scale_fwd = arg0._scale_fwd
    if hasattr(arg0, "_weight_bwd") and arg0._weight_bwd is not None and arg0._weight_bwd.numel() > 0:
        new_data_bwd = aten_op(arg0._weight_bwd, *args[1:], **kwargs)
        new_scale_bwd = arg0._scale_bwd

    return PostQuantWeight(
        new_data_fwd,
        new_scale_fwd,
        new_data_bwd,
        new_scale_bwd,
        args[0]._orig_dtype,
    )
