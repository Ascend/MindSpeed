# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import functools
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core.distributed.custom_fsdp.fully_sharded_data_parallel import TrainingState, RegisterFSDPBackwardFunction
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import PrefetchOrder
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.utils import is_submodule


def _register_fsdp_hooks(self, root_module):
    for m in root_module.modules():
        setattr(m, "_training_state", TrainingState.IDLE)

    self.forward_pre_hooks = {}
    self.forward_hooks = {}
    self.backward_pre_hooks = {}

    fsdp_unit_modules = self.fsdp_unit_modules

    def release_module_parameters(module, *unused):
        for param in module.parameters():
            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
            self.all_gather_pipeline.release_bucket(bucket_id)

        if not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
            release_params_fp8_transpose_cache(module.parameters())

    def release_params_fp8_transpose_cache(params):
        for param in params:
            if is_float8tensor(param):
                param._transpose_invalid = True
                param._transpose = None

    def all_gather_module_parameters(
        module,
        *unused,
        prefetch=True,
        prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
        wait_bucket_ready=True,
    ):
        ag_pipeline = self.all_gather_pipeline
        ag_pipeline.all_gather_params(
            params=list(module.parameters()),
            prefetch=prefetch,
            prefetch_order=prefetch_order,
            suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
        )
        if wait_bucket_ready:
            for param in module.parameters():
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                ag_pipeline.wait_bucket_ready(bucket_id)

    def _grad_acc(param):
        group_id = self.param_and_grad_buffer.param_to_param_group[param]
        group = self.param_and_grad_buffer.parameter_groups[group_id]
        if not group.requires_grad:
            return

        overwrite_main_grad = self.ddp_config.data_parallel_sharding_strategy in [
            "optim_grads",
            "optim_grads_params",
        ]
        if overwrite_main_grad:
            if not param.grad_added_to_main_grad:
                if param.grad is not None:
                    param.main_grad.copy_(param.grad)
                    del param.grad
                else:
                    param.main_grad.zero_()
        else:
            if not param.grad_added_to_main_grad:
                if param.grad is not None:
                    param.main_grad.add_(param.grad)
                    del param.grad

        param.grad_added_to_main_grad = False

    self._params_require_handle_grad = set()

    def _post_backward(module, *unused):
        if isinstance(module, tuple(fsdp_unit_modules)):
            if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                release_module_parameters(module)
                module._training_state = TrainingState.IDLE
            param_list = list(module.parameters())
        else:
            param_list = list(module.parameters(recurse=False))

        for param in param_list:
            _grad_acc(param)
            self._params_require_handle_grad.discard(param)

        grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
            "optim_grads",
            "optim_grads_params",
        ]
        if grad_reduce_every_bprop or self.is_last_microbatch:
            self.grad_reduce_pipeline.reduce_gradients(
                param_list, suggested_queue_capacity=self.suggested_RS_queue_capacity
            )

    def _pre_forward_param_unshard(
        module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ):
        input_training_state = module._training_state
        fsdp_forward_prefetch = True
        if input_training_state == TrainingState.PRE_BACKWARD:
            fsdp_forward_prefetch = False
        else:
            module._training_state = TrainingState.FORWARD

        if isinstance(module, tuple(fsdp_unit_modules)):
            param_list = list(module.parameters())
            self.all_gather_pipeline.all_gather_params(
                params=param_list,
                prefetch=fsdp_forward_prefetch,
                suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
            )
            for param in param_list:
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                self.all_gather_pipeline.wait_bucket_ready(bucket_id)
        else:
            param_list = list(module.parameters(recurse=False))
            self.all_gather_pipeline.all_gather_params(
                params=param_list,
                prefetch=fsdp_forward_prefetch,
                suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
            )
            for param in param_list:
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                self.all_gather_pipeline.wait_bucket_ready(bucket_id)
        return args, kwargs

    def _register_post_backward_hook(
        post_backward_hook: callable,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        if not torch.is_grad_enabled():
            return args, kwargs

        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)

        if len(inp_tensors) == 0:
            return args, kwargs

        inp_tensors = RegisterFSDPBackwardFunction.apply(
            functools.partial(post_backward_hook, module), *inp_tensors
        )

        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list):]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)

        return args, kwargs

    fsdp_modules = []
    for name, module in root_module.named_modules():
        if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
            continue

        if isinstance(module, tuple(fsdp_unit_modules)):
            fsdp_modules.append(module)

        self.forward_pre_hooks[f'module {name} parameter unshard'] = (
            module.register_forward_pre_hook(
                _pre_forward_param_unshard, prepend=True, with_kwargs=True
            )
        )
        self.forward_pre_hooks[f"module {name} register post-backward hook"] = (
            module.register_forward_pre_hook(
                functools.partial(_register_post_backward_hook, _post_backward),
                with_kwargs=True,
            )
        )

    def _root_post_backward(*unused):
        for param in self._params_require_handle_grad:
            _grad_acc(param)

        grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
            "optim_grads",
            "optim_grads_params",
        ]
        if grad_reduce_every_bprop or self.is_last_microbatch:
            self.grad_reduce_pipeline.reduce_gradients(
                list(self._params_require_handle_grad),
                suggested_queue_capacity=self.suggested_RS_queue_capacity,
            )
            self.grad_reduce_pipeline.reset()

        self._root_pre_backward_hook_issued = False

    def _pre_backward(module: nn.Module, *unused):
        module._training_state = TrainingState.PRE_BACKWARD
        if isinstance(module, tuple(fsdp_unit_modules)):
            all_gather_module_parameters(
                module, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER
            )

    self._root_pre_backward_hook_issued = False

    def _root_pre_backward(module: nn.Module, *unused):
        if self._root_pre_backward_hook_issued:
            return
        self._root_pre_backward_hook_issued = True

        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            for module in root_module.modules():
                if isinstance(module, tuple(fsdp_unit_modules)):
                    module._training_state = TrainingState.PRE_BACKWARD
                    for param in module.parameters():
                        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                        self.all_gather_pipeline.wait_bucket_ready(bucket_id, empty_ok=True)
                        self.all_gather_pipeline.release_bucket(bucket_id)
        self._params_require_handle_grad = set()
        for param_group in self.param_and_grad_buffer.parameter_groups:
            if not param_group.requires_grad:
                continue
            self._params_require_handle_grad |= set(param_group.params)
            for param in param_group.params:
                param.grad_added_to_main_grad = False

    def _post_forward(module: nn.Module, input: Any, output: Any):
        if module._training_state == TrainingState.PRE_BACKWARD:
            return output

        release_module_parameters(module)
        module._training_state = TrainingState.IDLE

        return output

    def _release_module_fp8_transpose_cache(module: nn.Module, *unused):
        release_params_fp8_transpose_cache(module.parameters(recurse=False))

    if len(fsdp_unit_modules) != 0:
        fsdp_modules = []
        for name, module in root_module.named_modules():
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue

            if isinstance(module, tuple(fsdp_unit_modules)):
                fsdp_modules.append(module)
                self.forward_hooks[f"release module {name} parameters"] = (
                    module.register_forward_hook(_post_forward, prepend=False)
                )
                self.backward_pre_hooks[f"all-gather module {name} parameters"] = (
                    module.register_full_backward_pre_hook(_pre_backward)
                )
            elif not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                self.forward_hooks[f"remove module {name} fp8 transpose cache"] = (
                    module.register_forward_hook(
                        _release_module_fp8_transpose_cache, prepend=False
                    )
                )

    for name, module in root_module.named_modules():
        if len(list(module.parameters())) != len(list(root_module.parameters())):
            continue

        self.backward_pre_hooks[f"{name} _root_pre_backward"] = (
            module.register_full_backward_pre_hook(_root_pre_backward)
        )
    self._root_pre_backward_hook_handle = root_module.register_full_backward_pre_hook(
        _root_pre_backward
    )

    # Equivalent implementation for torch.autograd.Variable._execution_engine.queue_callback(hook_fn)
    # Using Module.register_full_backward_hook on root_module to achieve the same mechanism
    self._root_post_backward_hook_handle = root_module.register_full_backward_hook(
        _root_post_backward
    )