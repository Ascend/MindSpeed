# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""KL divergence loss for Quantization-Aware Distillation (QAD).

Implements the KL divergence loss between teacher and student logits,
with support for temperature scaling and tensor parallelism.
"""

import torch
import torch.distributed
import torch.distributed.nn.functional as dist_nn_func
import torch.nn.functional as F
from megatron.core import parallel_state


class LogitsKLLoss(torch.nn.Module):
    """Calculates KL-Divergence loss between teacher and student logits.

    This loss function computes:
        L_KL = D_KL(p_teacher || p_student)
             = Σ_y p_teacher(y|x) * log(p_teacher(y|x) / p_student(y|x))

    Supports tensor parallelism (TP>1) via TP-aware softmax:
        - All-reduce MAX for numerical stability
        - All-reduce SUM for global softmax denominator
        - Uses torch.distributed.nn.functional.all_reduce to preserve gradients

    Temperature scaling:
        - Both teacher and student logits are divided by temperature T
        - Loss is multiplied by T² to compensate for gradient scaling
    """

    def __init__(self, temperature=1.0, reduction="mean"):
        """Initialize LogitsKLLoss.

        Args:
            temperature: Temperature for logit scaling (default: 1.0)
            reduction: Reduction method ("mean", "sum", or "none")
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            student_logits: Student model logits [seq_len, batch_size, vocab_size]
            teacher_logits: Teacher model logits [seq_len, batch_size, vocab_size]

        Returns:
            torch.Tensor: KL loss scalar or per-token loss
        """
        # Validate inputs
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Student and teacher logits must have same shape. "
                f"Got {student_logits.shape} and {teacher_logits.shape}"
            )

        # Step 1: Temperature scaling (divide by temperature)
        # This must happen before finding max for numerical stability
        output_teacher = teacher_logits.float() / self.temperature
        output_student = student_logits.float() / self.temperature

        # Step 2: Compute log probabilities (log softmax)
        # For TP>1, use TP-aware softmax with all_reduce for global probability distribution.
        # For TP=1, use standard F.log_softmax (no communication overhead).
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if tp_size > 1:
            teacher_log_prob, student_log_prob = self._tp_aware_log_softmax(output_teacher, output_student)
        else:
            teacher_log_prob = F.log_softmax(output_teacher, dim=-1)
            student_log_prob = F.log_softmax(output_student, dim=-1)

        # Step 3: Compute KL divergence
        # F.kl_div(input, target, log_target=True) computes:
        #   target_exp * (log_target - input)
        # = p_teacher * (log p_teacher - log p_student)
        # This is exactly D_KL(p_teacher || p_student)
        loss = F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True)

        # Sum over vocabulary dimension
        loss = torch.sum(loss, dim=-1)

        # Step 4: Temperature scaling compensation
        # Multiply by T² to maintain gradient magnitude
        loss = loss * (self.temperature**2)

        # Step 5: Apply reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass  # per-token loss, no reduction
        else:
            raise ValueError(f"Invalid reduction '{self.reduction}'. Must be 'mean', 'sum', or 'none'.")

        return loss

    def _tp_aware_log_softmax(self, output_teacher: torch.Tensor, output_student: torch.Tensor):
        """Compute TP-aware log softmax for KL divergence.

        When TP>1, each rank only has a partition of the vocabulary.
        Standard softmax over partial vocabulary is incorrect.
        This method computes global softmax using all-reduce operations:
        1. All-reduce MAX for numerical stability (prevent exp overflow)
        2. All-reduce SUM for global softmax denominator
        3. Compute log_softmax manually: log_prob = output - log(global_denom)

        Uses torch.distributed.nn.functional.all_reduce for denominator
        to preserve gradient flow through TP communication.

        Args:
            output_teacher: Teacher logits after temperature scaling [s, b, v_local]
            output_student: Student logits after temperature scaling [s, b, v_local]

        Returns:
            Tuple of (teacher_log_prob, student_log_prob) with global softmax
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()

        # Step 2a: Subtract global max for numerical stability
        # Each rank finds local max, then all-reduce to get global max.
        # This prevents exp() overflow when computing softmax denominator.
        teacher_logits_max, _ = torch.max(output_teacher, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            teacher_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
        output_teacher = output_teacher - teacher_logits_max

        student_logits_max, _ = torch.max(output_student, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            student_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
        # Detach student max to prevent gradient noise from max operation.
        # The max is only used for numerical stability (shifting), not for
        # probability computation. Following NVIDIA's design pattern.
        output_student = output_student - student_logits_max.detach()

        # Step 2b: Compute global softmax denominator
        # Sum of exp over local vocab partition, then all-reduce to get global sum.
        # Use torch.distributed.nn.functional.all_reduce to preserve gradients
        # through the TP communication (unlike torch.distributed.all_reduce
        # which breaks gradient flow).
        denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1, keepdim=True)
        denom_teacher = dist_nn_func.all_reduce(denom_teacher, group=tp_group)

        denom_student = torch.sum(torch.exp(output_student), dim=-1, keepdim=True)
        denom_student = dist_nn_func.all_reduce(denom_student, group=tp_group)

        # Step 2c: Compute log softmax with global denominator
        # log_prob = output - log(global_denom)
        # = log(exp(output) / Σ exp(output))  over FULL vocabulary
        teacher_log_prob = output_teacher - torch.log(denom_teacher)
        student_log_prob = output_student - torch.log(denom_student)

        return teacher_log_prob, student_log_prob
