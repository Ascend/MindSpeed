# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple, Callable
import warnings

import torch
import torch.utils._pytree as pytree
from torch.distributed.tensor import DTensor

from mindspeed.fsdp.quantization.core.cache import cached_quant, set_current_cacheable_weight
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig
from mindspeed.fsdp.quantization.core.post_quant_weight import PostQuantWeight


def _get_module_fsdp_state(module):
    """
    If module is an FSDP module, return its _FSDPState.
    Otherwise, return the _FSDPState of the closest parent FSDP module
    in the module hierarchy the module belongs to.
    """
    if hasattr(module, "_get_fsdp_state"):
        fsdp_state = module._get_fsdp_state()
    elif getattr(module, "_cached_parent_fsdp_state", None) is not None:
        fsdp_state = module._cached_parent_fsdp_state
    else:
        from torch.distributed._composable_state import _module_state_mapping

        min_nodes_in_parent = float("inf")
        closest_parent_fsdp_mod = None
        for fsdp_mod in _module_state_mapping.keys():
            all_submodules = list(fsdp_mod.modules())
            for submodule in all_submodules:
                if submodule is module and min_nodes_in_parent > len(all_submodules):
                    closest_parent_fsdp_mod = fsdp_mod
                    min_nodes_in_parent = len(all_submodules)
        if closest_parent_fsdp_mod is None:
            raise RuntimeError(
                "Module is not FSDP-wrapped and does not have any FSDP-wrapped parent modules."
            )
        fsdp_state = closest_parent_fsdp_mod._get_fsdp_state()
        module._cached_parent_fsdp_state = fsdp_state
    return fsdp_state


_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.expand.default,
    torch.ops.npu._npu_dtype_cast.default,
}


class PreQuantWeight(torch.Tensor):

    @staticmethod
    def __new__(
            cls,
            tensor: torch.Tensor,
            quantizer: Callable,
            config: Optional[Any] = None,
            dtype: Optional[torch.dtype] = None,
            name: Optional[str] = None,
            **kwargs,
    ):
        if "_tensor" in kwargs:
            tensor = kwargs["_tensor"]

        if tensor is None:
            return torch.Tensor._make_wrapper_subclass(cls, torch.Size([0]))
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
            self,
            tensor: torch.Tensor,
            quantizer: Callable[[torch.Tensor], Any],
            config: Optional[Any] = None,
            dtype: Optional[torch.dtype] = None,
            name: Optional[str] = None,
            **kwargs,
    ):
        self._tensor = tensor.contiguous()
        self.config = config
        self._dtype = dtype if dtype is not None else tensor.dtype
        self._name = name
        self._quantizer = quantizer

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return PreQuantWeight(
                args[0]._tensor,
                args[0]._quantizer,
                args[0].config,
                args[0]._dtype,
                args[0]._name,
            )
        config: Optional[QuantizeConfig] = None
        dtype: Optional[torch.dtype] = None
        name: Optional[str] = None
        quantizer: Callable = None

        def unwrap(t):
            nonlocal config, dtype, name, quantizer

            if config is None:
                config = t.config
                dtype = t._dtype
                name = t._name
                quantizer = t._quantizer
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            PreQuantWeight, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            warnings.warn(
                f"PreQuantWeight type is not preserved for Operator {func}, enable_fsdp_low_precision_all_gather is disabled")
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: PreQuantWeight(x, quantizer, config, dtype, name),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        metadata = {"config": self.config, "dtype": self._dtype, "name": self._name, "quantizer": self._quantizer}
        return tensors, metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return PreQuantWeight(
            inner_tensors["_tensor"],
            flatten_spec["quantizer"],
            flatten_spec["config"],
            flatten_spec["dtype"],
            flatten_spec["name"],
        )

    def __repr__(self):
        return "PreQuantWeight(" f"tensor={self._tensor}, config={self.config}, dtype={self._dtype}"

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Functions FSDP2 calls before all-gather of the
        weights for both forward and backward passes.
        Args:
            mesh (torch.distributed.DeviceMesh): DeviceMesh used by FSDP2
            to shard the weights.
            orig_size (torch.Size): Original size of the weight tensor.(For us same as self.shape)
            contiguous_orig_stride (Tuple[int]): Original stride of the weight tensor
            (For us same as self.stride()).
            module (FSDPModule): FSDP module. FSDP wrapped module wrapped using fully_shard
            that contains this tensor.
            mp_policy (MixedPrecisionPolicy): Mixed precision policy used by FSDP2.

        Returns:
            sharded_tensors: Tuple[torch.Tensor, ...]: Tuple of tensors
            that need to be all-gathered.
            metadata: Tuple[Any]: Metadata needed for reconstructing the
            PostQuantTensor after all-gather.
        """

        fsdp_state = _get_module_fsdp_state(module)
        reshard_after_forward = fsdp_state._fsdp_param_group._reshard_after_forward

        hp_tensor = self._tensor
        if hp_tensor.dtype == torch.float32:
            hp_tensor = hp_tensor.to(torch.bfloat16)
        if not hp_tensor.is_contiguous():
            hp_tensor = hp_tensor.contiguous()

        # The MXFP8 quantized operator requires the last two dimensions of the weight to be divisible by 64.
        weight_shape = hp_tensor.shape
        dim_m2, dim_m1 = weight_shape[-2], weight_shape[-1]

        if dim_m2 % 64 != 0:
            raise RuntimeError(
                f"mxfp8 quantization requires weight.dim(-2) to be divisible by 64. "
                f"Got dim(-2) = {dim_m2}, which is not divisible by 64."
                f"Please configure the quantization module to ignore this module：'{self._name.split('.')[-1]}'"
            )
        if dim_m1 % 64 != 0:
            raise RuntimeError(
                f"mxfp8 quantization requires weight.dim(-1) to be divisible by 64. "
                f"Got dim(-1) = {dim_m1}, which is not divisible by 64. "
                f"Please configure the quantization module to ignore this module：'{self._name.split('.')[-1]}'"
            )

        set_current_cacheable_weight(self._tensor)
        weight_fwd, scale_fwd, weight_bwd, scale_bwd = cached_quant(
            hp_tensor,
            self._quantizer,
        )

        world_size = mesh.size() if mesh is not None else 1

        # Prepare tensors for all-gather based on training phase.
        # If resharding after forward pass, only gather necessary tensors
        # (weight_fwd/scale_fwd for forward; weight_bwd/scale_bwd for backward).
        # Otherwise, gather all tensors regardless of phase.
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState

        training_state = fsdp_state._fsdp_param_group._training_state

        if reshard_after_forward and self.config.fsdp_low_precision_all_gather_mode != "all":
            is_backward_pass = training_state == TrainingState.PRE_BACKWARD
            # Allgather only the necessary tensors based on forward/backward pass
            fwd_usage = not is_backward_pass
            bwd_usage = is_backward_pass

            if is_backward_pass:
                sharded_tensors = (weight_bwd,) if world_size == 1 else (weight_bwd, scale_bwd)
            else:
                sharded_tensors = (weight_fwd,) if world_size == 1 else (weight_fwd, scale_fwd)

        else:
            fwd_usage = bwd_usage = True

            sharded_tensors = (weight_fwd, weight_bwd) if world_size == 1 else (weight_fwd, scale_fwd, weight_bwd,
                                                                                scale_bwd)

        metadata = (fwd_usage, bwd_usage, scale_fwd, scale_bwd)
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
            self,
            all_gather_outputs: Tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
    ):
        fwd_usage, bwd_usage, local_scale_fwd, local_scale_bwd = metadata

        weight_fwd, scale_fwd = None, None
        weight_bwd, scale_bwd = None, None

        num_expected_weights = int(fwd_usage) + int(bwd_usage)

        if len(all_gather_outputs) == num_expected_weights:
            weight_fwd = all_gather_outputs[0] if fwd_usage else None
            scale_fwd = local_scale_fwd if fwd_usage else None

            weight_bwd = all_gather_outputs[-1] if bwd_usage else None
            scale_bwd = local_scale_bwd if bwd_usage else None

        elif len(all_gather_outputs) == num_expected_weights * 2:
            weight_fwd = all_gather_outputs[0] if fwd_usage else None
            scale_fwd = all_gather_outputs[1] if fwd_usage else None

            weight_bwd = all_gather_outputs[-2] if bwd_usage else None
            scale_bwd = all_gather_outputs[-1] if bwd_usage else None

        else:
            raise ValueError(f"Unexpected gather outputs length: {len(all_gather_outputs)}")

        if out is not None:
            if isinstance(out, DTensor):
                out = out.to_local()
            if not isinstance(out, PostQuantWeight):
                raise TypeError(f"Expected PostQuantWeight, but got {type(out).__name__}")

            if fwd_usage:
                out._weight_fwd = weight_fwd
                out._scale_fwd = scale_fwd
            if bwd_usage:
                out._weight_bwd = weight_bwd
                out._scale_bwd = scale_bwd
        else:
            out = PostQuantWeight(
                weight_fwd,
                scale_fwd,
                weight_bwd,
                scale_bwd,
                param_dtype,
            )
        return out, all_gather_outputs


torch.serialization.add_safe_globals([PreQuantWeight])
