# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Patch that wraps ``megatron.training.training.train_step`` with the
HiF8 pre-commit recovery runner.

When ``--fp8-recipe hif8_delayed`` is used, step recovery is enabled by
default.  Use ``--no-hif8-step-recovery`` to explicitly disable it.
When the feature is disabled, the wrapper short-circuits to the original
``train_step`` without any overhead.

When enabled, the wrapper builds a :class:`RecoverableStepRunner` and calls
``run_train_step`` which temporarily patches optimizer methods to catch
NaN/Inf *before* the optimizer commits parameter updates.
"""

from functools import wraps

from mindspeed.args_utils import get_full_args
from mindspeed.core.transformer_engine.step_recovery.controller import RecoverableStepRunner


def train_step_recovery_wrapper(fn):
    """Wrap Megatron train_step with HiF8 pre-commit recovery."""

    @wraps(fn)
    def wrapper(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        args = get_full_args()

        fp8_recipe = getattr(args, "fp8_recipe", None)
        fp8_recipe = getattr(fp8_recipe, "value", fp8_recipe)

        if fp8_recipe != "hif8_delayed" or getattr(args, "no_hif8_step_recovery", False):
            return fn(
                forward_step_func,
                data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                config,
                forward_backward_func,
                iteration=iteration,
            )

        runner = RecoverableStepRunner()

        return runner.run_train_step(
            original_train_step=fn,
            logical_step=iteration if iteration is not None else 0,
            data_iterator=data_iterator,
            forward_step_func=forward_step_func,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            config=config,
            forward_backward_func=forward_backward_func,
        )

    return wrapper
