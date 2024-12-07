import torch
from mindspeed.op_builder import FusedEmaAdamWOpBuilder

__all__ = ["npu_apply_fused_ema_adamw"]

fused_ema_adamw_op_builder = FusedEmaAdamWOpBuilder()


def npu_apply_fused_ema_adamw(grad: torch.Tensor,
                              var: torch.Tensor,
                              m: torch.Tensor,
                              v: torch.Tensor,
                              s: torch.Tensor,
                              step: torch.Tensor,
                              lr: float = 1e-3,
                              ema_decay: float = 0.9999,
                              beta1: float = 0.9,
                              beta2: float = 0.999,
                              eps: float = 1e-8,
                              mode: int = 1,
                              bias_correction: bool = True,
                              weight_decay: float = 0.0):
    fused_ema_adamw_ops = fused_ema_adamw_op_builder.load()
    return fused_ema_adamw_ops.npu_apply_fused_ema_adamw(grad,
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