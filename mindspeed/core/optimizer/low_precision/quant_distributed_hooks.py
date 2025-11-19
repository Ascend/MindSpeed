from functools import wraps
import torch

from megatron.training import get_args
from megatron.core.transformer.cuda_graphs import is_graph_capturing


def _iter_group_triplets(model_groups, primary_groups, fallback_groups):
    model_groups = list(model_groups or [])
    primary_groups = list(primary_groups or [])
    fallback_groups = list(fallback_groups or [])
    for idx, model_group in enumerate(model_groups):
        primary = primary_groups[idx] if idx < len(primary_groups) else None
        fallback = fallback_groups[idx] if idx < len(fallback_groups) else None
        max_len = len(model_group)
        if primary is not None:
            max_len = max(max_len, len(primary))
        if fallback is not None:
            max_len = max(max_len, len(fallback))
        for pos in range(max_len):
            if pos >= len(model_group):
                continue
            model_param = model_group[pos]
            shard_main_param = primary[pos] if primary is not None and pos < len(primary) else None
            shard_model_param = fallback[pos] if fallback is not None and pos < len(fallback) else None
            yield model_param, shard_main_param, shard_model_param


def _iter_optimizer_param_triplets(optimizer):
    yield from _iter_group_triplets(
        getattr(optimizer, "model_float16_groups", []),
        getattr(optimizer, "shard_fp32_from_float16_groups", []),
        getattr(optimizer, "shard_float16_groups", []),
    )
    yield from _iter_group_triplets(
        getattr(optimizer, "model_fp32_groups", []),
        getattr(optimizer, "shard_fp32_groups", []),
        getattr(optimizer, "shard_fp32_groups", []),
    )


def collect_main_grad_data_for_unscaling_wrapper(func):
    @wraps(func)
    def _collect_main_grad_data_for_unscaling(self):
        args = get_args()
        main_grads = func(self)
        if getattr(args, "quant_grads", False):
            if not isinstance(main_grads, list):
                main_grads = list(main_grads)
            seen_ids = {id(tensor) for tensor in main_grads}
            seen_scale_ids = set()
            meta_grads_scale_inv = []

            def _register_quant_tensor(tensor):
                if tensor is None:
                    return
                if id(tensor) not in seen_ids:
                    main_grads.append(tensor)
                    seen_ids.add(id(tensor))
                meta = getattr(tensor, "meta", None)
                if meta is not None and getattr(meta, "qtype", None) == 4:
                    meta.mxfp8_scale_convert()
                scale_inv = getattr(meta, "scale_inv", None) if meta is not None else None
                if scale_inv is not None and id(scale_inv) not in seen_scale_ids:
                    meta_grads_scale_inv.append(scale_inv)
                    seen_scale_ids.add(id(scale_inv))

            for model_param, shard_main_param, shard_model_param in _iter_optimizer_param_triplets(self):
                _register_quant_tensor(getattr(model_param, "quant_grad", None))
                if shard_main_param is not None:
                    _register_quant_tensor(getattr(shard_main_param, "quant_grad", None))
                if shard_model_param is not None:
                    _register_quant_tensor(getattr(shard_model_param, "quant_grad", None))

            inner_optimizer = getattr(self, "optimizer", None)
            if inner_optimizer is not None:
                for group in getattr(inner_optimizer, "param_groups", []):
                    for param in group["params"]:
                        _register_quant_tensor(getattr(param, "quant_grad", None))
            return main_grads, meta_grads_scale_inv
        return main_grads

    return _collect_main_grad_data_for_unscaling


def copy_model_grads_to_main_grads_wrapper(func):
    @wraps(func)
    def _copy_model_grads_to_main_grads(self):
        args = get_args()
        ret = None

        if getattr(args, "quant_grads", False):
            for model_param, shard_main_param, shard_model_param in _iter_optimizer_param_triplets(self):
                param_range_map = self._get_model_param_range_map(model_param)
                param_range = param_range_map["param"]
                model_grad = getattr(model_param, "main_grad", None)
                if model_grad is None:
                    continue
                shard_model_grad = model_grad.view(-1)[param_range.start:param_range.end]
                meta = getattr(model_grad, "meta", None)
                if meta is not None:
                    shard_model_grad.meta = meta
                for target_param in (shard_main_param, shard_model_param):
                    if target_param is None:
                        continue
                    target_param.quant_grad = shard_model_grad
                    target_param.grad = None
                model_param.quant_grad = model_grad
                model_param.grad = None
        else:
            ret = func(self)
        return ret
    return _copy_model_grads_to_main_grads


def collect_main_grad_data_for_unscaling_quant(self):
    main_grads = []
    meta_grads_scale_inv = []
    seen_quant_ids = set()
    seen_scale_ids = set()
    for group in self.optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                main_grads.append(param.grad.data)
            quant_grad = getattr(param, "quant_grad", None)
            if quant_grad is None:
                continue
            if id(quant_grad) not in seen_quant_ids:
                seen_quant_ids.add(id(quant_grad))
                main_grads.append(quant_grad)
            meta = getattr(quant_grad, "meta", None)
            scale_inv = getattr(meta, "scale_inv", None) if meta is not None else None
            if scale_inv is not None and id(scale_inv) not in seen_scale_ids:
                seen_scale_ids.add(id(scale_inv))
                meta_grads_scale_inv.append(scale_inv)
    for model_param, shard_main_param, shard_model_param in _iter_optimizer_param_triplets(self):
        for tensor in (
            getattr(model_param, "quant_grad", None),
            getattr(shard_main_param, "quant_grad", None) if shard_main_param is not None else None,
            getattr(shard_model_param, "quant_grad", None) if shard_model_param is not None else None,
        ):
            if tensor is None:
                continue
            if id(tensor) not in seen_quant_ids:
                seen_quant_ids.add(id(tensor))
                main_grads.append(tensor)
            meta = getattr(tensor, "meta", None)
            if meta is not None and getattr(meta, "qtype", None) == 4:
                meta.mxfp8_scale_convert()
            scale_inv = getattr(meta, "scale_inv", None) if meta is not None else None
            if scale_inv is not None and id(scale_inv) not in seen_scale_ids:
                seen_scale_ids.add(id(scale_inv))
                meta_grads_scale_inv.append(scale_inv)
    return main_grads, meta_grads_scale_inv


def copy_model_grads_to_main_grads_quant(self):
    args = get_args()
    propagate_quant = getattr(args, "quant_grads", False)
    for model_param, shard_main_param, shard_model_param in _iter_optimizer_param_triplets(self):
        param_range_map = self._get_model_param_range_map(model_param)
        param_range = param_range_map["param"]
        model_grad = getattr(model_param, "main_grad", None)
        if model_grad is None:
            continue
        shard_model_grad = model_grad.view(-1)[param_range.start: param_range.end]
        meta = getattr(model_grad, "meta", None)
        if meta is not None:
            shard_model_grad.meta = meta
        targets = [shard_main_param, shard_model_param]
        for target_param in targets:
            if target_param is None:
                continue
            if propagate_quant:
                target_param.quant_grad = shard_model_grad
                target_param.grad = None
            else:
                target_param.grad = shard_model_grad.to(dtype=target_param.dtype, copy=True)
        if propagate_quant:
            model_param.quant_grad = model_grad
            model_param.grad = None


def unscale_main_grads_and_check_for_nan(self):
    collected = self._collect_main_grad_data_for_unscaling()
    if isinstance(collected, tuple):
        main_grads, meta_grads_scale_inv = collected
    else:
        main_grads = collected
        meta_grads_scale_inv = []

    self.found_inf.fill_(0.0)
    torch._amp_foreach_non_finite_check_and_unscale_(main_grads, self.found_inf, self.grad_scaler.inv_scale)
    if meta_grads_scale_inv:
        torch._amp_foreach_non_finite_check_and_unscale_(meta_grads_scale_inv, self.found_inf, self.grad_scaler.inv_scale)
    torch.distributed.all_reduce(
        self.found_inf,
        op=torch.distributed.ReduceOp.MAX,
        group=self.get_grad_stats_parallel_group(),
    )
    return self.found_inf.item() > 0


def _add_to_quant_grad(target_tensor: torch.Tensor, grad_tensor: torch.Tensor) -> None:
    meta = getattr(target_tensor, "meta", None)
    if meta is None:
        target_tensor.add_(grad_tensor.data)
        return
    fp32_tensor = meta.dequantization(target_tensor.data)
    fp32_tensor.add_(grad_tensor.data)
    target_tensor.data.copy_(meta.quantization(fp32_tensor))


def ddp_make_backward_post_hook_wrapper(make_hook_func):
    @wraps(make_hook_func)
    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        args = get_args()
        if not getattr(args, "quant_grads", False):
            return make_hook_func(self, param)

        def hook(*unused):
            if is_graph_capturing():
                return
            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), "param.grad being None is not safe when overlap_grad_reduce is True"
                grad_tensor = param.grad
                if grad_tensor is not None and (
                    not param.grad_added_to_main_grad or getattr(param, "zero_out_wgrad", False)
                ):
                    main_grad = getattr(param, "main_grad", None)
                    if main_grad is not None and getattr(main_grad, "meta", None) is not None:
                        _add_to_quant_grad(main_grad, grad_tensor)
                    elif main_grad is not None:
                        main_grad.add_(grad_tensor.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)

        return hook

    return _make_backward_post_hook
