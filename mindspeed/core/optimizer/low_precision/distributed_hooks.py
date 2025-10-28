from functools import wraps

import torch

from megatron.training import get_args

_GRAD_DTYPE_ALIAS = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def _is_precision_enabled(args_namespace) -> bool:
    return getattr(args_namespace, 'use_precision_aware_optimizer', False)


def _is_quant_enabled(args_namespace) -> bool:
    return getattr(args_namespace, 'use_quant_optimizer', False)


def _resolve_dtype(value, default: torch.dtype = torch.float32) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    token_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    if isinstance(value, str):
        return token_map.get(value, default)
    return default


def _cast_quant_grad_if_needed(grad_tensor, dtype_token):
    if dtype_token is None:
        return grad_tensor
    target_dtype = _GRAD_DTYPE_ALIAS.get(dtype_token)
    if target_dtype is None:
        return grad_tensor
    if hasattr(grad_tensor, 'meta') and getattr(grad_tensor, 'meta', None) is not None:
        return grad_tensor
    if grad_tensor.dtype == target_dtype:
        return grad_tensor
    return grad_tensor.to(dtype=target_dtype, non_blocking=True)


def collect_main_grad_data_for_unscaling_wrapper(func):
    @wraps(func)
    def _collect_main_grad_data_for_unscaling(self):
        args_namespace = get_args()
        if not _is_quant_enabled(args_namespace):
            return func(self)

        main_grads = func(self)
        meta_grads_scale_inv = []
        if getattr(args_namespace, 'quant_grads', False):
            main_groups = getattr(self, 'fp32_from_float16_groups', None)
            fallback_groups = getattr(self, 'float16_groups', None)
            if main_groups is None:
                main_groups = getattr(self, 'shard_fp32_from_float16_groups', [])
                fallback_groups = getattr(self, 'shard_float16_groups', fallback_groups or [])
            main_groups = list(main_groups or [])
            fallback_groups = list(fallback_groups or [])
            for group_idx, main_group in enumerate(main_groups):
                fallback_group = fallback_groups[group_idx] if group_idx < len(fallback_groups) else None
                for param_idx, main_param in enumerate(main_group):
                    target_param = main_param
                    if target_param is None and fallback_group is not None:
                        target_param = fallback_group[param_idx]
                    if target_param is None:
                        continue
                    quant_grad = getattr(target_param, 'quant_grad', None)
                    if quant_grad is not None and getattr(quant_grad, 'meta', None) is not None:
                        meta_grads_scale_inv.append(quant_grad.meta.scale_inv)
        if meta_grads_scale_inv:
            return main_grads, meta_grads_scale_inv
        return main_grads

    return _collect_main_grad_data_for_unscaling


def copy_model_grads_to_main_grads_wrapper(func):
    @wraps(func)
    def _copy_model_grads_to_main_grads(self):
        args_namespace = get_args()
        precision_enabled = _is_precision_enabled(args_namespace)
        quant_enabled = _is_quant_enabled(args_namespace)

        if quant_enabled:
            if not getattr(args_namespace, 'quant_grads', False):
                return func(self)
            quant_dtype_token = getattr(args_namespace, 'quant_grads_dtype', None)

            def copy_group_grads(model_groups, shard_main_groups, shard_model_groups):
                for model_group, shard_main_group, shard_model_group in zip(model_groups, shard_main_groups, shard_model_groups):
                    for model_param, shard_main_param, shard_model_param in zip(model_group, shard_main_group, shard_model_group):
                        param_range_map = self._get_model_param_range_map(model_param)
                        param_range = param_range_map['param']
                        model_grad = getattr(model_param, 'main_grad', None)
                        if model_grad is None:
                            continue
                        shard_model_grad = model_grad.view(-1)[param_range.start:param_range.end]
                        target_param = shard_main_param if shard_main_param is not None else shard_model_param
                        if target_param is None:
                            continue
                        if hasattr(model_grad, 'meta'):
                            target_param.quant_grad = shard_model_grad
                            target_param.grad = None
                            target_param.quant_grad.meta = model_grad.meta
                        else:
                            target_param.quant_grad = _cast_quant_grad_if_needed(shard_model_grad, quant_dtype_token)
                            target_param.grad = None

            float16_fallback = getattr(self, 'shard_float16_groups', getattr(self, 'model_float16_groups', []))
            fp32_fallback = getattr(self, 'shard_fp32_groups', getattr(self, 'model_fp32_groups', []))

            copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups, float16_fallback)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups, fp32_fallback)
            return

        if precision_enabled:
            func(self)
            grad_dtype = _resolve_dtype(getattr(args_namespace, 'main_grads_dtype', torch.float32))
            for group in getattr(self.optimizer, 'param_groups', []):
                for param in group['params']:
                    grad = getattr(param, 'decoupled_grad', None)
                    if grad is not None:
                        if grad.dtype != grad_dtype:
                            param.decoupled_grad = grad.to(dtype=grad_dtype)
                        continue
                    grad = param.grad
                    if grad is None:
                        continue
                    param.decoupled_grad = grad.to(dtype=grad_dtype)
                    param.grad = None
            return

        return func(self)

    return _copy_model_grads_to_main_grads
