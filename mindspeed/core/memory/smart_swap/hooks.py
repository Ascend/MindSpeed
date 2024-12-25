# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch

from .swap_utils import print_with_rank, PrintLevel


def get_module_name(module: torch.nn.Module):
    return module.__module__ + "." + module.__class__.__name__


class SwapHookRegister:
    id = 0

    def __init__(self):
        self.id = SwapHookRegister.id
        SwapHookRegister.id += 1

        self.fwd_pre_hook_handle = None
        self.fwd_post_hook_handle = None
        self.bwd_pre_hook_handle = None
        self.bwd_post_hook_handle = None
        self.fwd_begin_module: torch.nn.Module = None
        self.fwd_end_module: torch.nn.Module = None
        self.bwd_begin_module: torch.nn.Module = None
        self.bwd_end_module: torch.nn.Module = None
        self.fwd_idx = 0
        self.bwd_idx = 0
        self.prehook_handles = []
        self.posthook_handls = []

        self.fwd_pre_hook_custom_func = None
        self.fwd_post_hook_custom_func = None
        self.bwd_pre_hook_custom_func = None
        self.bwd_post_hook_custom_func = None

    def __del__(self):
        r"""if not need swap hook to module, del it."""

        self.reset()

        if self.fwd_pre_hook_handle:
            self.fwd_pre_hook_handle.remove()
        if self.fwd_post_hook_handle:
            self.fwd_post_hook_handle.remove()
        if self.bwd_pre_hook_handle:
            self.bwd_pre_hook_handle.remove()
        if self.bwd_post_hook_handle:
            self.bwd_post_hook_handle.remove()

    def reset(self):
        self.fwd_begin_module = None
        self.fwd_end_module = None
        self.bwd_begin_module = None
        self.bwd_end_module = None

        self.fwd_idx = 0
        self.bwd_idx = 0
        for hdl in self.prehook_handles:
            hdl.remove()
        for hdl in self.posthook_handls:
            hdl.remove()
        self.prehook_handles.clear()
        self.posthook_handls.clear()

    def register_custom_func(
        self, fwd_pre_hook_custom_func, fwd_post_hook_custom_func, bwd_pre_hook_custom_func, bwd_post_hook_custom_func
    ):
        r"""
        custom_func(instance_id, fwd_or_bwd_idx)
        """
        self.fwd_pre_hook_custom_func = fwd_pre_hook_custom_func
        self.fwd_post_hook_custom_func = fwd_post_hook_custom_func
        self.bwd_pre_hook_custom_func = bwd_pre_hook_custom_func
        self.bwd_post_hook_custom_func = bwd_post_hook_custom_func

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="SwapHook", print_level=print_level)

    def register_hook_to_grad_fn(self, input_tensor, position, is_bwd_pre):

        def grad_fn_bwd_pre_hook(grad_outputs):
            self.bwd_idx += 1
            self.print_with_rank(f"grad_fn_bwd_pre_hook: bwd begin, id[{self.id}], bwd_idx[{self.bwd_idx}]")
            # border
            if self.bwd_pre_hook_custom_func:
                self.bwd_pre_hook_custom_func(self.id, self.bwd_idx)
            return grad_outputs

        def grad_fn_bwd_post_hook(grad_inputs, _):
            self.print_with_rank(f"grad_fn_bwd_post_hook: bwd end, id[{self.id}], bwd_idx[{self.bwd_idx}]")
            # border
            if self.bwd_post_hook_custom_func:
                self.bwd_post_hook_custom_func(self.id, self.bwd_idx)
            return grad_inputs

        if is_bwd_pre:
            self.print_with_rank(f"{position}, register grad_fn_bwd_pre_hook to grad_fn: {input_tensor.grad_fn}")
            self.prehook_handles.append(input_tensor.grad_fn.register_prehook(grad_fn_bwd_pre_hook))
        else:
            self.print_with_rank(f"{position}, register grad_fn_bwd_post_hook to grad_fn: {input_tensor.grad_fn}")
            self.posthook_handls.append(input_tensor.grad_fn.register_hook(grad_fn_bwd_post_hook))

    def register_hook_to_bwd_end_module(self, module, inputs, position):
        if not self.bwd_end_module or (self.bwd_end_module and module is self.bwd_end_module):
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            if isinstance(inputs, tuple):
                for input_item in inputs:
                    if not isinstance(input_item, torch.Tensor):
                        continue
                    if (input_item.requires_grad and not input_item.is_leaf) and input_item.grad_fn:
                        if not self.bwd_end_module:
                            self.bwd_end_module = module
                            self.print_with_rank(f"{position}, set bwd_end_module: {get_module_name(module)}")

                        self.register_hook_to_grad_fn(input_item, position, is_bwd_pre=False)
                        break

    def register_hook_to_bwd_begin_module(self, module, inputs, position):
        if self.bwd_begin_module and module is self.bwd_begin_module:
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            if isinstance(inputs, tuple):
                for input_item in inputs:
                    if not isinstance(input_item, torch.Tensor):
                        continue
                    if (input_item.requires_grad and not input_item.is_leaf) and input_item.grad_fn:

                        self.register_hook_to_grad_fn(input_item, position, is_bwd_pre=True)
                        break

    def fwd_pre_hook(self, module, args):
        self.print_with_rank(f"fwd_pre_hook, {get_module_name(module)}")

        if not self.fwd_begin_module:
            self.fwd_begin_module = module
            self.fwd_end_module = module
            self.bwd_begin_module = module
            self.print_with_rank(
                f"fwd_pre_hook: set fwd_begin_module, fwd_end_module and bwd_begin_module: {get_module_name(module)}"
            )

        if self.fwd_begin_module and module is self.fwd_begin_module:
            self.fwd_idx += 1
            self.print_with_rank(
                f"fwd_pre_hook: fwd begin, id[{self.id}], fwd_idx[{self.fwd_idx}], {get_module_name(module)}"
            )
            # border
            if self.fwd_pre_hook_custom_func:
                self.fwd_pre_hook_custom_func(self.id, self.fwd_idx)

        self.register_hook_to_bwd_end_module(module, args, "fwd_pre_hook")

        return None

    def fwd_post_hook(self, module, _, outputs):
        self.print_with_rank(f"fwd_post_hook, {get_module_name(module)}")

        if self.fwd_end_module and module is self.fwd_end_module:
            self.print_with_rank(
                f"fwd_post_hook: fwd end, id[{self.id}], fwd_idx[{self.fwd_idx}], {get_module_name(module)}"
            )
            # border
            if self.fwd_post_hook_custom_func:
                self.fwd_post_hook_custom_func(self.id, self.fwd_idx)

        self.register_hook_to_bwd_begin_module(module, outputs, "fwd_post_hook")
        self.register_hook_to_bwd_end_module(module, outputs, "fwd_post_hook")

        return None

    def register_hooks_to_modules_recursively(self, module, name=""):
        self.print_with_rank(f"register_hooks_to_modules_recursively, {get_module_name(module)}")

        for child_name, child in module.named_children():
            self.register_hooks_to_modules_recursively(child, name + child_name)

        def module_fwd_pre_hook(module, args):
            return self.fwd_pre_hook(module, args)

        def module_fwd_post_hook(module, args, outputs):
            return self.fwd_post_hook(module, args, outputs)

        self.fwd_pre_hook_handle = module.register_forward_pre_hook(module_fwd_pre_hook)
        self.fwd_post_hook_handle = module.register_forward_hook(module_fwd_post_hook)


def register_swap_hooks_to_modules(
    module,
    fwd_pre_hook_custom_func=None,
    fwd_post_hook_custom_func=None,
    bwd_pre_hook_custom_func=None,
    bwd_post_hook_custom_func=None,
):
    r"""
    usage:

    # before training
    models = [model_1, model_2, ...]
    swap_hook_registers = []

    def fwd_pre_hook_custom_func(swap_hook_register_id, fwd_idx):
        ...

    def fwd_post_hook_custom_func(swap_hook_register_id, fwd_idx):
        ...

    def bwd_pre_hook_custom_func(swap_hook_register_id, bwd_idx):
        ...

    def bwd_post_hook_custom_func(swap_hook_register_id, bwd_idx):
        ...

    for model in models:
        import smart_swap
        swap_hook_register = smart_swap.xxx.register_swap_hooks_to_modules(.
            model,
            fwd_pre_hook_custom_func, fwd_post_hook_custom_func
            bwd_pre_hook_custom_func, bwd_post_hook_custom_func)

        swap_hook_registers.append(swap_hook_register)

    # when training
    for step in range(train_steps):
        for swap_hook_register in swap_hook_registers:
            swap_hook_register.reset()

        train_step(xxx)

    # after training
    for swap_hook_register in swap_hook_registers:
        del swap_hook_register

    """

    swap_hook_register = SwapHookRegister()
    swap_hook_register.register_hooks_to_modules_recursively(module)
    swap_hook_register.register_custom_func(
        fwd_pre_hook_custom_func, fwd_post_hook_custom_func, bwd_pre_hook_custom_func, bwd_post_hook_custom_func
    )

    return swap_hook_register
