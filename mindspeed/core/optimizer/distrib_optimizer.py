# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch

HAVE_APEX_OR_TE = True
USING_TE_OPTIMIZER = False
USING_APEX_OPTIMIZER = False
try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam

    USING_TE_OPTIMIZER = True
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam

        USING_APEX_OPTIMIZER = False
    except ImportError:
        from torch.optim import Adam as Adam

        HAVE_APEX_OR_TE = False

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer


def state_dict(self):
    """
    The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
    related) optimizer variables. The returned state dict can be stored in
    the standard model/RNG checkpoint file. The parameter and dependent
    optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
    checkpoint file by calling 'save_parameter_state()'.
    """
    inner_state_dict = self.optimizer.state_dict()
    state_dict = {}

    # Extract 'step', for non-Apex/TE support.
    if not HAVE_APEX_OR_TE:
        steps = list(set([s["step"].item() for s in inner_state_dict["state"].values()]))
        assert len(steps) == 1
        step = steps[0]
    elif isinstance(self.optimizer, HybridDeviceOptimizer):
        step = None
        for optimizer in self.optimizer.sub_optimizers:
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                if len(optimizer.state) == 0:
                    continue
                steps = list(set([s["step"].item() for s in optimizer.state.values()]))
                assert len(steps) == 1, f"steps: {optimizer.state}"
                step = steps[0]
                break
    elif USING_TE_OPTIMIZER or USING_APEX_OPTIMIZER:
        # Extract 'step', for TE FusedAdam support.
        steps = list(
            set(
                [
                    g["step"]
                    for g in inner_state_dict["param_groups"]
                    if len(g["params"]) > 0 and "step" in g
                ]
            )
        )
        assert len(steps) <= 1, f"steps: {steps}"
        step = steps[0] if len(steps) == 1 else None

    # Optimizer state (do not store parameter state here).
    state_dict['optimizer'] = {k: v for k, v in inner_state_dict.items() if k != "state"}
    for param_group in state_dict["optimizer"]["param_groups"]:
        del param_group["params"]
        if not HAVE_APEX_OR_TE:
            # Native PyTorch param group requires step (i.e., iteration).
            param_group["step"] = step
        elif (
            USING_TE_OPTIMIZER
            or USING_APEX_OPTIMIZER
            or isinstance(self.optimizer, HybridDeviceOptimizer)
        ) and step is not None:
            # TE FusedAdam will not accumulate step for empty param groups, so we need to
            # align the step across param groups.
            param_group["step"] = int(step)

    # Grad scaler state.
    if self.grad_scaler:
        state_dict['grad_scaler'] = self.grad_scaler.state_dict()

    return state_dict