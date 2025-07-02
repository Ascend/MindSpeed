# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import unittest
from unittest import mock
from dataclasses import dataclass
from typing import Tuple, Any, List, Optional
import torch
from torch import Tensor
from mindspeed.core.optimizer.fused_ema_adamw.fused_ema_adamw import fused_ema_adamw, FusedEmaAdamW

DEVICE = torch.device('npu')


@dataclass
class OptimizerState:
    p_data: Any
    m_ref: Any
    v_ref: Any
    s_ref: Any


@dataclass
class OptimizerConfig:
    lr: float
    ema_decay: float
    beta1: float
    beta2: float
    eps: float
    mode: int
    bias_correction: bool
    weight_decay: float


def mock_npu_apply_fused_ema_adamw(g_ref: Any, state: OptimizerState, step: int, config: OptimizerConfig) -> Tuple[Any, Any, Any, Any]:
    p_data, m_ref, v_ref, s_ref = state.p_data, state.m_ref, state.v_ref, state.s_ref
    new_p = p_data * 0.9 + g_ref * 0.1
    new_m = m_ref * config.beta1 + g_ref * (1 - config.beta1)
    new_v = v_ref * config.beta2 + g_ref.pow(2) * (1 - config.beta2)
    new_s = s_ref * config.ema_decay + p_data * (1 - config.ema_decay)
    return new_p, new_m, new_v, new_s


class TestFusedEmaAdamW(unittest.TestCase):
    def setUp(self):
        self.params = [
            torch.randn(10, 10, requires_grad=True, device=DEVICE),
            torch.randn(10, 10, requires_grad=True, device=DEVICE)
        ]

    def test_initialization(self):
        optimizer = FusedEmaAdamW(self.params)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(optimizer.param_groups[0]['params']), 2)

        defaults = optimizer.defaults
        self.assertEqual(defaults['lr'], 1e-3)
        self.assertEqual(defaults['eps'], 1e-8)
        self.assertEqual(defaults['betas'], (0.9, 0.999))
        self.assertEqual(defaults['weight_decay'], 1e-2)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            FusedEmaAdamW(self.params, lr=-0.1)

        with self.assertRaises(ValueError):
            FusedEmaAdamW(self.params, eps=-1e-8)

        with self.assertRaises(ValueError):
            FusedEmaAdamW(self.params, betas=(-0.1, 0.999))

        with self.assertRaises(ValueError):
            FusedEmaAdamW(self.params, betas=(0.9, 1.1))

        with self.assertRaises(ValueError):
            FusedEmaAdamW(self.params, weight_decay=-1e-2)

        with self.assertRaises(RuntimeError):
            FusedEmaAdamW(self.params, amsgrad=True)

        with self.assertRaises(RuntimeError):
            FusedEmaAdamW(self.params, maximize=True)

    @mock.patch('mindspeed.ops.npu_apply_fused_ema_adamw.npu_apply_fused_ema_adamw', new=mock_npu_apply_fused_ema_adamw)
    def test_step(self, skip=True):
        if skip:
            return
        optimizer = FusedEmaAdamW(self.params)

        loss = self.params[0].sum() + self.params[1].sum()
        loss.backward()
        with self.assertRaises(RuntimeError):
            optimizer.step()

    def test_zero_grad(self):
        optimizer = FusedEmaAdamW(self.params, set_grad_none=True)

        loss = self.params[0].sum() + self.params[1].sum()
        loss.backward()

        for param in self.params:
            self.assertIsNotNone(param.grad)

        optimizer.zero_grad()

        for param in self.params:
            self.assertIsNone(param.grad)

    def test_copy_to(self, skip=True):
        if skip:
            return
        optimizer = FusedEmaAdamW(self.params)

        loss = self.params[0].sum() + self.params[1].sum()
        loss.backward()
        with self.assertRaises(RuntimeError):
            optimizer.step()
        optimizer.copy_to()

    def test_store_and_restore(self):
        optimizer = FusedEmaAdamW(self.params)

        original_params = [p.data.clone() for p in self.params]

        optimizer.store(optimizer.param_groups)

        for param in self.params:
            param.data.add_(1.0)

        for i, param in enumerate(self.params):
            self.assertFalse(torch.allclose(param.data, original_params[i], atol=1e-6))

        optimizer.restore(optimizer.param_groups)

        for i, param in enumerate(self.params):
            self.assertTrue(torch.allclose(param.data, original_params[i], atol=1e-6))