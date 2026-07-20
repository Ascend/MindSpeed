# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Teacher model manager for Quantization-Aware Distillation (QAD).

Manages the teacher model lifecycle: loading, inference, and state management.
The teacher model runs in frozen mode (no_grad, eval) to guide the student
model training via KL divergence loss.
"""

import logging
from typing import Optional

import torch
from megatron.training import get_args

logger = logging.getLogger(__name__)


class TeacherModelManager:
    """Manages the teacher model lifecycle for QAD training.

    The teacher model is a full-precision model (precision handled by Megatron
    autocast) that provides target output distributions for the student model.

    Key properties:
    - Teacher runs in torch.no_grad() mode
    - Teacher parameters are frozen (requires_grad=False)
    - Teacher is in eval() mode
    - Teacher checkpoint is loaded independently from student
    """

    def __init__(self, config, student_model=None):
        """Initialize the TeacherModelManager.

        Args:
            config: QADConfig instance containing QAD configuration
            student_model: Optional reference to the student model for
                          architecture matching
        """
        self.config = config
        self.student_model = student_model
        self.teacher_model: Optional[torch.nn.Module] = None
        self._is_loaded = False
        # Cache the device the teacher model resides on to avoid calling
        # .to(device) on every get_logits() iteration. .to() is a no-op when
        # the device matches, but it still traverses all parameters/buffers
        # to check, which is wasteful at scale.
        self._teacher_device = None

    def load_teacher(self, model_provider_func=None):
        """Load the teacher model from checkpoint.

        The teacher model precision is handled by Megatron's autocast mechanism,
        matching the NVIDIA Model-Optimizer approach. When --bf16 is set, autocast
        uses mixed precision (BF16 matmul + FP32 sensitive ops like norm/softmax).
        When --fp32 is set, the teacher runs in full FP32.

        If model_provider_func is provided, it will be called to create
        a fresh model instance for the teacher. Otherwise, the student
        model architecture is used as reference.

        Args:
            model_provider_func: Optional function to create teacher model.
                                If None, uses student model architecture.

        Returns:
            torch.nn.Module: The loaded teacher model

        Raises:
            RuntimeError: If teacher model cannot be loaded
        """
        if self._is_loaded and self.teacher_model is not None:
            logger.info("[QAD] Teacher model already loaded, skipping")
            return self.teacher_model

        logger.info("[QAD] Loading teacher model from: %s", self.config.teacher_checkpoint_path)

        if model_provider_func is not None:
            # Create a new model instance for the teacher
            # Teacher uses no quantization (precision handled by autocast)
            self.teacher_model = model_provider_func()
        elif self.student_model is not None:
            # Clone student model architecture for teacher
            self.teacher_model = self._create_teacher_from_student()
        else:
            raise RuntimeError("Either model_provider_func or student_model must be provided")

        if self.teacher_model is None:
            raise RuntimeError("Failed to create teacher model instance")

        # Load checkpoint
        self._load_checkpoint(self.config.teacher_checkpoint_path)

        # Freeze teacher parameters (always — this is a QAD requirement)
        # Teacher must be frozen to provide a stable target distribution.
        # This is a mathematical requirement of knowledge distillation,
        # not a tunable hyperparameter.
        self._freeze_teacher()

        # Set to eval mode (always — disables dropout, uses running stats)
        self.teacher_model.eval()

        self._is_loaded = True
        logger.info("[QAD] Teacher model loaded successfully")

        return self.teacher_model

    def _create_teacher_from_student(self):
        """Create a teacher model with the same architecture as the student.

        Returns:
            torch.nn.Module: New teacher model instance (precision handled by autocast)
        """
        args = get_args()

        # Temporarily disable QAT scheme when creating teacher model.
        # Teacher must not be quantized, while student uses QAT quantization.
        # Precision is handled by Megatron's autocast (not explicit .bfloat16()).
        original_qat_scheme = getattr(args, 'qat_scheme', None)
        args.qat_scheme = None

        try:
            # Import the appropriate model class
            from megatron.core.models.gpt import GPTModel
            from megatron.training.arguments import core_transformer_config_from_args
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_local_spec,
                get_gpt_layer_with_transformer_engine_spec,
            )

            use_te = args.transformer_impl == "transformer_engine"

            if not args.use_legacy_models:
                if args.spec is not None:
                    from megatron.core.transformer.spec_utils import import_module

                    transformer_layer_spec = import_module(args.spec)
                else:
                    if use_te:
                        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                            args.num_experts, args.moe_grouped_gemm
                        )
                    else:
                        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

                config = core_transformer_config_from_args(args)

                teacher_model = GPTModel(
                    config=config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=args.padded_vocab_size,
                    max_sequence_length=args.max_position_embeddings,
                    pre_process=True,
                    post_process=True,
                    fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                    parallel_output=True,
                    share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                    position_embedding_type=args.position_embedding_type,
                    rotary_percent=args.rotary_percent,
                    rotary_base=args.rotary_base,
                    rope_scaling=args.use_rope_scaling,
                    mtp_block_spec=None,
                )
            else:
                import megatron.legacy.model

                config = core_transformer_config_from_args(args)
                teacher_model = megatron.legacy.model.GPTModel(
                    config, num_tokentypes=0, parallel_output=True, pre_process=True, post_process=True
                )

            # NOTE: Do NOT explicitly cast teacher to BF16 here.
            # Megatron's autocast (enabled by --bf16 flag) handles precision
            # using mixed precision: BF16 for matmul, FP32 for norm/softmax.
            # This matches the NVIDIA Model-Optimizer approach and provides
            # better accuracy than forcing all ops to BF16 via .bfloat16().

        finally:
            # Restore QAT scheme for student model
            args.qat_scheme = original_qat_scheme

        return teacher_model

    def _load_checkpoint(self, checkpoint_path):
        """Load teacher model checkpoint.

        Raises FileNotFoundError if the path doesn't exist, and RuntimeError if
        no checkpoint is found, the checkpoint's tensor-parallel size does not
        match the current training run, or loading fails. A randomly-initialized
        teacher produces meaningless KL targets, so we fail fast rather than
        silently training against noise.

        Args:
            checkpoint_path: Path to the teacher checkpoint directory

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist
            RuntimeError: If no checkpoint is found, the checkpoint TP size
                does not match the current training TP size, or checkpoint
                loading fails
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint path not found: {checkpoint_path}")

        from megatron.training.checkpointing import _load_base_checkpoint
        from megatron.training.utils import unwrap_model

        args = get_args()

        # Load the base checkpoint state dict directly.
        # CRITICAL: Use rank0=False so each TP rank loads its OWN shard.
        # With rank0=True, all ranks load mp_rank_00's shard, which means
        # ranks 1-7 get WRONG output layer weights (rank 0's partition).
        # This causes teacher and student to have different logits on
        # non-rank-0 TP ranks, resulting in KL ≈ 2.2 even when both
        # load from the same checkpoint.
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(checkpoint_path, args, rank0=False)

        if state_dict is None:
            # Fail fast instead of silently using a random teacher.
            # A random teacher produces meaningless KL targets.
            raise RuntimeError(
                f"No checkpoint found in {checkpoint_path}. "
                "A randomly-initialized teacher provides meaningless KL targets. "
                "Please verify --qad-teacher-load points to a valid checkpoint directory."
            )

        # Validate that the checkpoint was saved with the same tensor-parallel
        # size as the current training run. With rank0=False each TP rank loads
        # its own shard, so a TP mismatch means every rank loads the WRONG
        # weight partition. strict=False below would silently accept this,
        # producing a teacher whose logits are meaningless -> silent KL
        # corruption. Fail fast instead.
        saved_args = state_dict.get('args')
        if saved_args is not None:
            # saved_args may be an argparse.Namespace (standard) or a plain dict
            # (some checkpoint converters). Handle both so the check is not
            # silently skipped for dict-format args.
            if isinstance(saved_args, dict):
                saved_tp_size = saved_args.get('tensor_model_parallel_size')
            else:
                saved_tp_size = getattr(saved_args, 'tensor_model_parallel_size', None)
            current_tp_size = getattr(args, 'tensor_model_parallel_size', None)
            if saved_tp_size is not None and current_tp_size is not None and saved_tp_size != current_tp_size:
                raise RuntimeError(
                    f"Teacher checkpoint TP size mismatch: checkpoint was saved with "
                    f"tensor_model_parallel_size={saved_tp_size}, but current training "
                    f"uses tensor_model_parallel_size={current_tp_size}. "
                    f"Each TP rank loads its own shard (rank0=False), so a TP mismatch "
                    f"means every rank loads the wrong weight partition, producing "
                    f"meaningless KL targets. Re-convert the teacher checkpoint to "
                    f"TP={current_tp_size} or run training with TP={saved_tp_size}."
                )

        # Load model state dict into teacher model
        teacher_model = unwrap_model(self.teacher_model)
        if isinstance(teacher_model, list):
            teacher_model = teacher_model[0]

        # The state dict may have 'model' key (Megatron format)
        if 'model' in state_dict:
            model_state_dict = state_dict['model']
        else:
            model_state_dict = state_dict

        # Use strict=False to handle shape mismatches (e.g., test config vs full model)
        missing_keys, unexpected_keys = teacher_model.load_state_dict(model_state_dict, strict=False)

        if missing_keys:
            logger.warning("[QAD] Missing keys in teacher checkpoint: %s", missing_keys[:5])
        if unexpected_keys:
            logger.warning("[QAD] Unexpected keys in teacher checkpoint: %s", unexpected_keys[:5])

        logger.info("[QAD] Teacher checkpoint loaded from: %s", checkpoint_path)

    def _freeze_teacher(self):
        """Freeze all teacher model parameters."""
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Also ensure parameters are not in any optimizer
        logger.info("[QAD] Teacher parameters frozen")

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get teacher model logits for distillation.

        Runs teacher forward pass in torch.no_grad() mode.

        Args:
            input_ids: Input token IDs [seq_len, batch_size]
            attention_mask: Attention mask [seq_len, batch_size]
            position_ids: Position IDs [seq_len, batch_size]

        Returns:
            torch.Tensor: Teacher logits [seq_len, batch_size, vocab_size]
        """
        if not self._is_loaded or self.teacher_model is None:
            raise RuntimeError("Teacher model not loaded. Call load_teacher() first.")

        # Only move teacher to the input device when the device actually
        # changes. Calling .to(device) every iteration traverses all
        # parameters/buffers even when it is a no-op, which is wasteful
        # at scale. We cache the current device and skip the move when it
        # matches the input device.
        device = input_ids.device
        if self._teacher_device != device:
            self.teacher_model = self.teacher_model.to(device)
            self._teacher_device = device

        # Always run in no_grad mode for teacher
        # Teacher produces logits for KL divergence — no labels needed
        with torch.no_grad():
            logits = self.teacher_model(input_ids, position_ids, attention_mask)

        return logits

    def is_loaded(self) -> bool:
        """Check if teacher model is loaded.

        Returns:
            bool: True if teacher model is loaded
        """
        return self._is_loaded and self.teacher_model is not None

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics for the teacher model.

        Returns:
            dict: Memory usage statistics
        """
        if not self._is_loaded or self.teacher_model is None:
            return {"loaded": False}

        total_params = 0
        total_grads = 0
        for param in self.teacher_model.parameters():
            total_params += param.numel()
            if param.grad is not None:
                total_grads += param.grad.numel()

        return {
            "loaded": True,
            "total_params": total_params,
            "total_grads": total_grads,
            "requires_grad_all_false": all(not p.requires_grad for p in self.teacher_model.parameters()),
        }

    def reset(self):
        """Reset the teacher model manager.

        Unloads the teacher model and clears all state.
        """
        if self.teacher_model is not None:
            del self.teacher_model
            self.teacher_model = None
            # Clear the cached device so the next load re-detects it.
            self._teacher_device = None
            # Support both CUDA and NPU for cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.empty_cache()
            self._is_loaded = False
            logger.info("[QAD] Teacher model reset")
