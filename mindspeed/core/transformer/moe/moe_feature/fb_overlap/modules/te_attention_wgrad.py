# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""Bridge dense TE attention wgrad into the existing FB-overlap schedule."""

import weakref
from copy import copy

from .weight_grad_store import WeightGradStore


class AttentionWeightGradStore:
    """Use TE's operand queue, but let FB overlap choose when it is consumed."""

    def __init__(self, module):
        self.module_ref = weakref.ref(module)
        self.store = type(module.wgrad_store)(delay_wgrad_compute=True)
        self.consuming = False
        self.pending = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def delay_wgrad_compute(self):
        if self.consuming:
            return True
        if not WeightGradStore.is_decoupleBlock:
            return False
        module = self.module_ref()
        # Quantized/debug operands have their own lifetime and precision rules.
        # Preserve their existing immediate path; this bridge owns dense wgrad.
        return (
            module is not None
            and not getattr(module, 'fp8', False)
            and not getattr(module, 'fp8_calibration', False)
            # is_debug_iter() itself queries this store; read its cached result
            # to avoid recursion when TE enters a debug iteration.
            and not getattr(module, 'debug_enabled_in_this_iteration', False)
        )

    def put(self, tensors, compute):
        if not self.delay_wgrad_compute():
            raise RuntimeError('Attention wgrad queued outside an FB-overlap decoupling region.')
        # run_graph_backward explicitly resizes graph-output / gradient storage.
        # Strong references alone cannot protect those operands. Match the
        # native FB linear's detach+clone ownership contract; never copy weights.
        operands = [tensor.detach().clone() for tensor in tensors]
        self.store.put(operands, compute)
        self.pending += 1
        WeightGradStore.put_te_linear(self)

    def backward_dw(self):
        module = self.module_ref()
        if module is None or not self.pending:
            raise RuntimeError('Missing module or queued attention wgrad at the FB boundary.')
        self.consuming = True
        try:
            # MCore's wrapper and TE's base both test delay_wgrad_compute.
            # Keep it enabled during consumption even after end_decouple().
            module.backward_dw()
            self.pending -= 1
        finally:
            self.consuming = False

    def skip_autograd_post_hook(self):
        return not self.consuming and self.pending > 0 and self.delay_wgrad_compute()


def configure_attention_wgrad(model):
    """Install before DDP registers hooks; touch only attention Linear weights."""
    from transformer_engine.pytorch import LayerNormLinear, Linear

    for owner in model.modules():
        attention = getattr(owner, 'self_attention', None)
        if attention is None:
            continue
        for module in attention.modules():
            if not isinstance(module, (Linear, LayerNormLinear)):
                continue
            if isinstance(module.wgrad_store, AttentionWeightGradStore):
                continue
            config = getattr(module, 'config', None)
            if (
                getattr(module, 'primary_weights_in_fp8', False)
                or getattr(config, 'fp8', None)
                or getattr(config, 'fp4', None)
                or getattr(config, 'quant_recipe', None)
            ):
                continue
            if module.wgrad_store.delay_wgrad_compute():
                raise RuntimeError('FB-overlap attention already has a different delayed-wgrad owner.')
            weight_names = set(module.weight_names)
            weights = [
                param
                for name, param in module.named_parameters(recurse=False)
                if name in weight_names and param.requires_grad
            ]
            if not weights:
                continue
            store = AttentionWeightGradStore(module)
            module.wgrad_store = store
            if config is not None:
                # Avoid changing the shared layer/global config or DDP schedule.
                module.config = copy(config)
                module.config.delay_wgrad_compute = True
            for param in weights:
                param.skip_backward_post_hook = True
                param._mindspeed_fb_attention_wgrad_store = store
            # Bias and LayerNorm/RMSNorm parameter gradients remain eager.
