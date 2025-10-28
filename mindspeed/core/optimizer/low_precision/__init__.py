"""Precision-aware optimizer helpers for MindSpeed."""

from .precision_aware_adamw import AdamW, FusedTorchAdamW, adamw
from .grad_clip import (
    clip_grad_by_total_norm_fp32_wrapper,
    get_grad_norm_fp32,
)
from .optimizer_hooks import (
    collect_main_grad_data_for_unscaling_wrapper,
    copy_model_grads_to_main_grads_wrapper,
    get_optimizer_builder_wrapper,
    optimizer_config_init_wrapper,
    optimizer_config_post_init_wrapper,
    prepare_grads_impl,
    step_with_ready_grads_impl,
    mixed_precision_optimizer_step_impl,
    unscale_main_grads_and_check_for_nan,
    get_main_grads_for_grad_norm,
    zero_grad_group_helper_wrapper,
)
from .distributed_hooks import copy_model_grads_to_main_grads_wrapper as distrib_copy_model_grads_to_main_grads_wrapper

__all__ = [
    "AdamW",
    "FusedTorchAdamW",
    "adamw",
    "clip_grad_by_total_norm_fp32_wrapper",
    "get_grad_norm_fp32",
    "collect_main_grad_data_for_unscaling_wrapper",
    "copy_model_grads_to_main_grads_wrapper",
    "distrib_copy_model_grads_to_main_grads_wrapper",
    "get_optimizer_builder_wrapper",
    "optimizer_config_init_wrapper",
    "optimizer_config_post_init_wrapper",
    "prepare_grads_impl",
    "step_with_ready_grads_impl",
    "mixed_precision_optimizer_step_impl",
    "unscale_main_grads_and_check_for_nan",
    "get_main_grads_for_grad_norm",
    "zero_grad_group_helper_wrapper",
]
