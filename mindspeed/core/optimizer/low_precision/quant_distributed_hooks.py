# Copyright (c) Microsoft Corporation.

from functools import wraps
import torch

from megatron.training import get_args
from megatron.core.transformer.cuda_graphs import is_graph_capturing


def collect_main_grad_data_for_unscaling_quant(self):
    main_grads, meta_grads_scale_inv = [], []
    for group in self.optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                main_grads.append(param.grad.data)
    args = get_args()
    if args.quant_grads:
        meta_grads_scale_inv = [
            param.quant_grad.meta.scale_inv
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]

    return main_grads, meta_grads_scale_inv


def copy_model_grads_to_main_grads_quant(self):
    args = get_args()

    def copy_group_grads(model_groups, shard_main_groups):
        for model_group, shard_main_group in zip(model_groups, shard_main_groups):
            for model_param, shard_main_param in zip(model_group, shard_main_group):
                param_range_map = self._get_model_param_range_map(model_param)
                param_range = param_range_map["param"]
                assert param_range.size == shard_main_param.nelement()

                model_grad = model_param.main_grad
                shard_model_grad = model_grad.view(-1)[param_range.start: param_range.end]
                if args.quant_grads:
                    shard_main_param.quant_grad = shard_model_grad
                    shard_main_param.quant_grad.meta = model_grad.meta
                else:
                    shard_main_param.grad = shard_model_grad.float()

    # Copy model groups to shard groups.
    copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
    copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)


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
