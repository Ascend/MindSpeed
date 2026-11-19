import torch

__all__ = ["npu_apply_fused_adamw_v2"]


def npu_apply_fused_adamw_v2(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    max_exp_avg_sq: torch.Tensor,
    state_step: torch.Tensor,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    amsgrad: bool = False,
    maximize: bool = False,
):
    torch._fused_adamw_(
        [param],
        [grad],
        [exp_avg],
        [exp_avg_sq],
        [max_exp_avg_sq] if amsgrad else [],
        [state_step],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
    )
