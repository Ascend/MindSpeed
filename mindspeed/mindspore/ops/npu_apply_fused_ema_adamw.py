# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch_npu


def npu_apply_fused_ema_adamw(grad,
                              var,
                              m,
                              v,
                              s,
                              step,
                              lr: float = 1e-3,
                              ema_decay: float = 0.9999,
                              beta1: float = 0.9,
                              beta2: float = 0.999,
                              eps: float = 1e-8,
                              mode: int = 1,
                              bias_correction: bool = True,
                              weight_decay: float = 0.0):
    return torch_npu.npu_apply_fused_ema_adamw(grad,
                                                var,
                                                m,
                                                v,
                                                s,
                                                step,
                                                lr,
                                                ema_decay,
                                                beta1,
                                                beta2,
                                                eps,
                                                mode,
                                                bias_correction,
                                                weight_decay)
