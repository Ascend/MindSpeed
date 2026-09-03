# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps

from megatron.core.extensions.transformer_engine import (
    split_te_layernorm_column_parallel_linear,
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.identity_op import IdentityOp

from mindspeed.core.memory.recompute.norm.should_recompute import should_recompute_norm
from mindspeed.core.memory.recompute.recompute_common import CheckpointWithoutOutput


def _update_sharded_state_dict_mapping(layer, norm_name, fused_linear_name):
    submodules_config = getattr(layer, "submodules_config", None)
    key_mapping = getattr(submodules_config, "sharded_state_dict_keys_map", None)
    if key_mapping is None:
        return

    key_mapping.update(
        {
            f"{norm_name}.weight": f"{fused_linear_name}.layer_norm_weight",
            f"{norm_name}.bias": f"{fused_linear_name}.layer_norm_bias",
        }
    )


def _split_fused_norm_linear(layer, parent, submodule_name, norm_name, fused_linear_name):
    fused_linear = getattr(parent, submodule_name, None)
    if fused_linear is None:
        raise NotImplementedError(
            f"Transformer Engine norm recompute cannot find submodule {submodule_name} on {type(parent)}."
        )

    tp_group = getattr(fused_linear, "tp_group", None)
    if tp_group is None:
        tp_group = getattr(fused_linear, "_tp_group", None)

    # Megatron 0.18's splitter reads these attributes unconditionally, while
    # TELayerNormColumnParallelLinear only guarantees _tp_group and may leave
    # ub_name unset when TP communication overlap is disabled.
    if not hasattr(fused_linear, "tp_group"):
        fused_linear.tp_group = tp_group
    if not hasattr(fused_linear, "ub_name"):
        fused_linear.ub_name = {
            "linear_qkv": "qkv",
            "linear_fc1": "fc1",
        }.get(submodule_name)

    try:
        norm, linear = split_te_layernorm_column_parallel_linear(
            fused_linear,
            layer.config,
            tp_group=tp_group,
        )
    except (AttributeError, TypeError) as error:
        raise NotImplementedError(
            f"Transformer Engine norm recompute cannot split {type(fused_linear)} "
            f"submodule {submodule_name}: {type(error).__name__}: {error}"
        ) from error

    setattr(layer, norm_name, norm)
    setattr(parent, submodule_name, linear)
    _update_sharded_state_dict_mapping(layer, norm_name, fused_linear_name)


def _configure_fused_norm_recompute(layer, parent, submodule_name, norm_name, fused_linear_name):
    fused_linear = getattr(parent, submodule_name, None)
    enable_recompute_norm = getattr(fused_linear, "enable_recompute_norm", None)
    if callable(enable_recompute_norm):
        return True

    # Megatron 0.18's TELayerNormColumnParallelLinear no longer guarantees the
    # MindSpeed-specific enable_recompute_norm() extension. Split the selected
    # fused module and let Megatron own the standalone norm checkpoint instead.
    _split_fused_norm_linear(layer, parent, submodule_name, norm_name, fused_linear_name)
    return False


def _enable_native_input_norm_recompute(layer):
    layer.recompute_input_layernorm = True
    if not (getattr(layer.config, "fp8", None) or getattr(layer.config, "fp4", None)):
        return

    set_for_recompute = getattr(layer.self_attention, "set_for_recompute_input_layernorm", None)
    if not callable(set_for_recompute):
        raise NotImplementedError(
            f"FP8/FP4 input norm recompute requires {type(layer.self_attention)} "
            "to provide set_for_recompute_input_layernorm()."
        )
    set_for_recompute()


def _enable_native_pre_mlp_norm_recompute(layer):
    layer.recompute_pre_mlp_layernorm = True
    if not (getattr(layer.config, "fp8", None) or getattr(layer.config, "fp4", None)):
        return

    set_for_recompute = getattr(layer.mlp, "set_for_recompute_pre_mlp_layernorm", None)
    if callable(set_for_recompute):
        set_for_recompute()
        return

    from megatron.core.extensions.transformer_engine import set_save_original_input

    set_save_original_input(layer.mlp.linear_fc1)


def _enable_fused_norm_recompute(layer, checkpoint_manager, submodule_name):
    target_layer = getattr(layer, submodule_name, None)
    enable_recompute_norm = getattr(target_layer, "enable_recompute_norm", None)
    if not callable(enable_recompute_norm):
        raise NotImplementedError(
            f"Transformer Engine norm recompute requires {type(target_layer)} "
            f"submodule {submodule_name} to provide enable_recompute_norm()."
        )
    enable_recompute_norm(checkpoint_manager)


def _discard_output_and_register_recompute(checkpoint_manager, hook_output):
    if checkpoint_manager.outputs is None:
        raise RuntimeError("The fused norm module did not register a checkpoint output.")

    checkpoint_manager.discard_output()
    candidates = hook_output if isinstance(hook_output, (tuple, list)) else (hook_output,)
    for candidate in candidates:
        if getattr(candidate, "requires_grad", False):
            candidate.register_hook(checkpoint_manager.recompute)
            return


def norm_recompute_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        self.mindspeed_recompute_fused_input_layernorm = False
        self.mindspeed_recompute_fused_pre_mlp_layernorm = False
        if not should_recompute_norm(getattr(self, "layer_number", None), self.config):
            return

        if isinstance(self.input_layernorm, IdentityOp):
            if self.config.transformer_impl == "transformer_engine":
                self.mindspeed_recompute_fused_input_layernorm = _configure_fused_norm_recompute(
                    self,
                    self.self_attention,
                    "linear_qkv",
                    "input_layernorm",
                    "self_attention.linear_qkv",
                )
                if not self.mindspeed_recompute_fused_input_layernorm:
                    _enable_native_input_norm_recompute(self)
        else:
            # Megatron 0.18 owns the non-fused norm checkpoint implementation.
            _enable_native_input_norm_recompute(self)

        if isinstance(self.pre_mlp_layernorm, IdentityOp):
            if self.config.transformer_impl == "transformer_engine":
                self.mindspeed_recompute_fused_pre_mlp_layernorm = _configure_fused_norm_recompute(
                    self,
                    self.mlp,
                    "linear_fc1",
                    "pre_mlp_layernorm",
                    "mlp.linear_fc1",
                )
                if not self.mindspeed_recompute_fused_pre_mlp_layernorm:
                    _enable_native_pre_mlp_norm_recompute(self)
        else:
            _enable_native_pre_mlp_norm_recompute(self)

    return wrapper


def norm_recompute_forward_attention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "mindspeed_recompute_fused_input_layernorm", False):
            return fn(self, *args, **kwargs)

        self.norm_ckpt1 = CheckpointWithoutOutput(get_cuda_rng_tracker)
        _enable_fused_norm_recompute(self.self_attention, self.norm_ckpt1, "linear_qkv")
        hidden_states, context = fn(self, *args, **kwargs)
        if self.training:
            _discard_output_and_register_recompute(self.norm_ckpt1, hidden_states)
        return hidden_states, context

    return wrapper


def norm_recompute_forward_mlp_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "mindspeed_recompute_fused_pre_mlp_layernorm", False):
            return fn(self, *args, **kwargs)

        self.norm_ckpt2 = CheckpointWithoutOutput(get_cuda_rng_tracker)
        _enable_fused_norm_recompute(self.mlp, self.norm_ckpt2, "linear_fc1")
        output = fn(self, *args, **kwargs)
        if self.training:
            _discard_output_and_register_recompute(self.norm_ckpt2, output)
        return output

    return wrapper
