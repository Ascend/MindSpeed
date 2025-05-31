import torch
from mindspeed.op_builder.fused_adamw_v2_builder import FusedAdamWV2OpBuilder

__all__ = ["npu_apply_fused_adamw_v2"]

op_builder = FusedAdamWV2OpBuilder()

tmp_tensor = torch.Tensor([1])


def npu_apply_fused_adamw_v2(param: torch.Tensor,
                             grad: torch.Tensor,
                             exp_avg: torch.Tensor,
                             exp_avg_sq: torch.Tensor,
                             max_exp_avg_sq: torch.Tensor,
                             state_step: int,
                             lr: float = 1e-3,
                             beta1: float = 0.9,
                             beta2: float = 0.999,
                             weight_decay: float = 0.0,
                             eps: float = 1e-8,
                             amsgrad: bool = False,
                             maximize: bool = False,
                             ):
    fused_adamw_ops = op_builder.load()
    if max_exp_avg_sq is None:
        max_exp_avg_sq = tmp_tensor
    return fused_adamw_ops.npu_apply_fused_adamw_v2(param,
                                                    grad,
                                                    exp_avg,
                                                    exp_avg_sq,
                                                    max_exp_avg_sq,
                                                    state_step,
                                                    lr,
                                                    beta1,
                                                    beta2,
                                                    weight_decay,
                                                    eps,
                                                    amsgrad,
                                                    maximize)
