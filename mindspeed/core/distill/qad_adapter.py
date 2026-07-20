# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""QAD (Quantization-Aware Distillation) patches for training pipeline.

Patches model_provider, forward_step, and loss_func to support QAD distillation:
a BF16 teacher guides an MXFP4 W4A4 student via KL divergence on logits.
"""

import logging
from functools import partial, wraps

import torch

from megatron.training import get_args
from megatron.core import mpu

from mindspeed.core.distill import (
    TeacherModelManager,
    LogitsKLLoss,
)

logger = logging.getLogger(__name__)

_teacher_manager = None


def get_teacher_manager():
    """Lazily create and cache the global TeacherModelManager from get_args()."""
    global _teacher_manager
    if _teacher_manager is None:
        args = get_args()
        from mindspeed.core.distill.config import QADConfig

        config = QADConfig(
            enabled=getattr(args, 'qad_enable', False),
            teacher_checkpoint_path=getattr(args, 'qad_teacher_load', ''),
            kl_temperature=getattr(args, 'kl_temperature', 1.0),
            kl_loss_weight=getattr(args, 'kl_loss_weight', 1.0),
            kl_reduction=getattr(args, 'kl_loss_reduction', 'mean'),
        )
        _teacher_manager = TeacherModelManager(config)
    return _teacher_manager


def reset_teacher_manager():
    """Clear the cached singleton (for test isolation / re-init)."""
    global _teacher_manager
    _teacher_manager = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_kl_loss_module = None


def _get_kl_loss_module(temperature):
    """Return a cached LogitsKLLoss instance (rebuilt if temperature changes)."""
    global _kl_loss_module
    if _kl_loss_module is None or _kl_loss_module.temperature != temperature:
        _kl_loss_module = LogitsKLLoss(temperature=temperature, reduction='none')
    return _kl_loss_module


def reset_kl_loss_module():
    """Clear the cached KL loss module (for test isolation / re-init)."""
    global _kl_loss_module
    _kl_loss_module = None


def _all_reduce_kl_for_logging(kl_loss_value):
    """Sum KL loss across TP ranks for logging. No-op when TP=1.

    The KL loss computed on each TP rank is a *partial* divergence over that
    rank's local vocabulary partition.  The full KL divergence is the **sum**
    of all partials (because KL = Σ_v p_teacher(v)·log(p_teacher(v)/p_student(v))
    and the vocabulary is sharded across TP ranks).

    Previously this divided by ``tp_size`` after the sum-reduce, which produced
    the *mean* of partials — an incorrect value that understates the true
    divergence by a factor of ``tp_size``.  The gradient path (via
    ``dist_nn_func.all_reduce`` inside ``LogitsKLLoss``) was already correct;
    only the logged scalar was wrong.
    """
    kl = kl_loss_value.detach().clone()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        torch.distributed.all_reduce(kl, group=mpu.get_tensor_model_parallel_group())
    return kl


def _apply_loss_mask(kl_loss_value, loss_mask, reduction):
    """Apply loss_mask to per-token KL loss [s,b] and reduce to scalar."""
    if loss_mask is None:
        return kl_loss_value.mean()
    mask = loss_mask.view(-1).float()
    kl = kl_loss_value.view(-1) * mask
    if reduction == 'mean':
        return kl.sum() / (mask.sum() + 1e-12)
    return kl.sum()


def _align_logits(logits, ref_dim0):
    """Transpose [b,s,v] → [s,b,v] if first dim doesn't match ref."""
    if isinstance(logits, torch.Tensor) and logits.dim() == 3 and logits.shape[0] != ref_dim0:
        return logits.transpose(0, 1).contiguous()
    return logits


class _BatchCacheIterator:
    """Iterator wrapper that replays the first batch once.

    Ensures ``forward_step_func`` (CE) and ``_compute_kl_loss`` (KL) operate
    on the *same* batch.  Without this, ``get_batch`` is called twice,
    consuming two different batches and mixing CE/KL gradients from
    different data.
    """

    def __init__(self, data_iterator):
        self._data_iterator = data_iterator
        self._cached_batch = None
        self._replay_done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._cached_batch is not None and not self._replay_done:
            self._replay_done = True
            return self._cached_batch
        batch = next(self._data_iterator)
        if self._cached_batch is None:
            self._cached_batch = batch
        return batch


def _compute_kl_loss(args, data_iterator, model, teacher_manager):
    """Run student + teacher logits forward and compute masked KL loss.

    Returns:
        torch.Tensor: The masked KL loss on success, or None if get_batch
        cannot be imported (caller decides fallback based on QAD mode).
    """
    try:
        from pretrain_gpt import get_batch
    except ImportError:
        logger.warning("[QAD] Cannot import get_batch from pretrain_gpt")
        return None

    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    # Student logits: forward WITHOUT labels, WITH gradients (KL must backprop)
    student_logits = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask)
    student_logits = _align_logits(student_logits, tokens.shape[0])

    # Teacher logits: frozen, no gradients
    with torch.no_grad():
        teacher_logits = teacher_manager.get_logits(tokens, attention_mask, position_ids)
    teacher_logits = _align_logits(teacher_logits, student_logits.shape[0])

    kl_loss = _get_kl_loss_module(getattr(args, 'kl_temperature', 1.0))(student_logits, teacher_logits)
    kl_loss = _apply_loss_mask(kl_loss, loss_mask, getattr(args, 'kl_loss_reduction', 'mean'))

    if getattr(args, 'iteration', 0) <= 2:
        logger.debug(
            "[QAD-DIAG] iter %s: KL loss = %.6f, student %s, teacher %s, rank %s",
            getattr(args, 'iteration', 0),
            kl_loss.item(),
            list(student_logits.shape),
            list(teacher_logits.shape),
            mpu.get_tensor_model_parallel_rank(),
        )

    return kl_loss


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


class QADForwardStepPatch:
    """Patch setup_model_and_optimizer() and train_step() to support QAD distillation.

    Instead of patching a consumer app's ``pretrain`` (which would create an
    inverted library→consumer dependency), QAD patches two megatron-level
    functions that every pretrain path — including MindSpeed-LLM's own
    reimplementation — calls:

    * ``megatron.training.training.setup_model_and_optimizer`` — receives
      ``model_provider`` as its first argument; QAD wraps it to load the
      teacher model alongside the student.
    * ``megatron.training.training.train_step`` — receives ``forward_step_func``
      as its first argument; QAD wraps it to compute the KL distillation loss.

    Both wrappers are no-ops when ``--qad-enable`` is off, so non-QAD training
    is unaffected.
    """

    @staticmethod
    def patched_setup_model_and_optimizer_wrapper(original_setup_model_and_optimizer):
        """Wrap setup_model_and_optimizer to inject the QAD model_provider.

        The QAD model_provider builds the student model via the original
        provider, then loads a frozen BF16 teacher (created from the same
        provider) for KL distillation.
        """

        @wraps(original_setup_model_and_optimizer)
        def patched_setup_model_and_optimizer(model_provider, *args, **kwargs):
            # NOTE: do NOT name this `args` — it would shadow the *args tuple and
            # break `*args` unpacking below (argparse.Namespace is not iterable).
            ms_args = get_args()
            if not getattr(ms_args, 'qad_enable', False):
                return original_setup_model_and_optimizer(model_provider, *args, **kwargs)

            def qad_model_provider(pre_process=True, post_process=True):
                student_model = model_provider(pre_process, post_process)
                logger.info("[QAD] Loading teacher model for distillation")
                # Create a NEW teacher instance via model_provider (not the student)
                get_teacher_manager().load_teacher(lambda: model_provider(pre_process=True, post_process=True))
                logger.info("[QAD] Teacher model loaded successfully")
                return student_model

            return original_setup_model_and_optimizer(qad_model_provider, *args, **kwargs)

        return patched_setup_model_and_optimizer

    @staticmethod
    def patched_train_step_wrapper(original_train_step):
        """Wrap train_step to inject the QAD forward_step (KL distillation loss).

        QAD uses KL-only distillation: the total loss is ``α · KL``. CE loss is
        computed (under no_grad) for logging only — it does not contribute to
        gradients. CE+KL mode was removed because it required a double student
        forward pass (producing inconsistent logits under dropout) and
        error-prone num_tokens scaling. For RL-trained models, the teacher
        distribution is the target — CE on hard labels is unnecessary.
        """

        @wraps(original_train_step)
        def patched_train_step(forward_step_func, *args, **kwargs):
            # NOTE: do NOT name this `args` — it would shadow the *args tuple and
            # break `*args` unpacking below (argparse.Namespace is not iterable).
            ms_args = get_args()
            if not getattr(ms_args, 'qad_enable', False):
                return original_train_step(forward_step_func, *args, **kwargs)

            teacher_manager = get_teacher_manager()
            if not teacher_manager.is_loaded():
                logger.warning("[QAD] teacher not loaded, falling back to original")
                return original_train_step(forward_step_func, *args, **kwargs)

            def qad_forward_step(data_iterator, model):
                """QAD-patched forward_step (KL-only distillation)."""
                is_validating = not model.training

                # Wrap the iterator so the first batch is replayed once.
                # forward_step_func (CE) and _compute_kl_loss (KL) both call
                # get_batch(data_iterator); without replay they would consume
                # two different batches, mixing CE/KL gradients from
                # different data.
                cached_iterator = _BatchCacheIterator(data_iterator)

                # --- CE forward (logging only, no gradients) ---
                # CE is computed under no_grad for logging purposes only.
                # KL drives all gradients.
                with torch.no_grad():
                    output_tensor, loss_func_partial = forward_step_func(cached_iterator, model)

                # --- KL loss (drives all gradients) ---
                # Student logits forward WITH gradients, teacher logits
                # under no_grad (frozen). KL loss is masked and reduced.
                # cached_iterator replays the same batch consumed by CE above.
                kl_loss_value = _compute_kl_loss(ms_args, cached_iterator, model, teacher_manager)
                if kl_loss_value is None:
                    # get_batch unavailable: cannot compute KL loss — fail fast
                    # rather than returning (None, None), which causes a TypeError
                    # in the Megatron scheduler.
                    raise RuntimeError(
                        "[QAD] KL-only mode requires get_batch from pretrain_gpt, "
                        "but it could not be imported. Cannot compute KL loss."
                    )

                return output_tensor, partial(
                    _qad_loss_func,
                    original_loss_func_partial=loss_func_partial,
                    kl_loss_value=kl_loss_value,
                    is_validating=is_validating,
                )

            return original_train_step(qad_forward_step, *args, **kwargs)

        return patched_train_step


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def _qad_loss_func(output_tensor, original_loss_func_partial=None, kl_loss_value=None, is_validating=False):
    """Compute KL-only loss for the Megatron pipeline schedule.

    QAD uses KL-only distillation: the total loss is ``α · KL``.
    CE loss is computed (under no_grad in qad_forward_step) for logging
    only — it does not contribute to gradients.

    Since KL is the sole gradient source, num_tokens = 1 so Megatron's
    division by num_tokens is a no-op and KL is returned as-is.
    """
    args = get_args()
    kl_weight = getattr(args, 'kl_loss_weight', 1.0)
    kl_loss_for_logging = _all_reduce_kl_for_logging(kl_loss_value)

    # CE loss (for logging only — no gradient contribution)
    if original_loss_func_partial is not None and output_tensor is not None:
        ce_loss, ce_num_tokens, metrics = original_loss_func_partial(output_tensor)
    else:
        ce_loss = torch.tensor(0.0, device=kl_loss_value.device)
        ce_num_tokens = torch.tensor(1, dtype=torch.int, device=kl_loss_value.device)
        metrics = {'lm loss': (ce_loss, ce_num_tokens)}

    # KL-only: KL drives all gradients. num_tokens = 1 so Megatron's
    # division is a no-op.
    num_tokens = torch.tensor(1, dtype=torch.int, device=kl_loss_value.device)
    total_loss = kl_weight * kl_loss_value

    metrics['kl loss'] = (kl_loss_for_logging, 1)

    ce_mean = ce_loss / (ce_num_tokens + 1e-12)
    mode = "validation" if is_validating else "training"
    logger.debug(
        "[QAD] KL-only %s. CE: %.6f, KL: %.6f, Total: %.6f",
        mode,
        ce_mean.item(),
        kl_loss_for_logging.item(),
        total_loss.item(),
    )

    return total_loss, num_tokens, metrics
