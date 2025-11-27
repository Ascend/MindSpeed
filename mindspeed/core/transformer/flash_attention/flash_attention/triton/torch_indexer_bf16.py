import math
import logging
import time
import torch


def index_bf16(
        q: torch.Tensor,
        q_s: torch.Tensor, 
        k: torch.Tensor,
        k_s: torch.Tensor
    ) -> torch.Tensor:
    
    """
    Pytorch bf16 implementation of fp8_index(TileLang impl).
    Compute index scores using a bfloat16-accelerated attention-like kernel.
    This function implements a PyTorch bfloat16 variant of the "fp8_index" (TileLang)
    operation. It performs a batched matmul between query and key tensors (after
    casting to bfloat16), applies a ReLU nonlinearity, scales by per-query and
    per-key scale tensors, and reduces across the head dimension to produce an
    index score per (batch, sequence, memory) position.
    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (B, S, H, D). Expected to contain values suitable
        for bfloat16 computation. Input must be contiguous. Will be cast to
        torch.bfloat16 for the matmul step.
    q_s : torch.Tensor
        Per-query scale tensor of shape (B, S, H). Expected dtype float32 and
        contiguous.
    k : torch.Tensor
        Key tensor of shape (B, M, D). Expected to contain values suitable for
        bfloat16 computation and be contiguous.
    k_s : torch.Tensor
        Per-key scale tensor of shape (B, M). Expected dtype float32 and
        contiguous.
    Returns
    -------
    torch.Tensor
        Index score tensor of shape (B, S, M) with dtype torch.float32. Computed as:
        1) logits = matmul(q_bf16, k_bf16_transposed) -> cast to float32
        2) logits = ReLU(logits)
        3) logits = logits * q_s[..., None]
        4) logits_sum = sum(logits, dim=2)  # reduce over H
        5) index_score = logits_sum * k_s[:, None, :]
    Raises
    ------
    ValueError
        If any of q, q_s, k or k_s is not contiguous.
    Notes
    -----
    - The heavy linear algebra (matmul) is performed in bfloat16 for performance,
        while scale factors and reductions are performed in float32 to retain
        numerical stability.
    - The ReLU nonlinearity sets negative raw dot-product logits to zero prior to
        scaling and reduction.
    - The caller must ensure the shapes are consistent (q.D == k.D, q.B == k.B,
        q.B == q_s.B == k_s.B) and that the provided scale tensors correspond to the
        intended quantization scales.
    - Computational complexity is O(B * S * H * D * M) for the matmul-dominated step.
    Example (shapes only)
    ---------------------
    q:   (B, S, H, D), dtype ~bfloat16-capable
    q_s: (B, S, H),    dtype=float32
    k:   (B, M, D),    dtype ~bfloat16-capable
    k_s: (B, M),       dtype=float32
    Return: (B, S, M), dtype=float32
    """

    if not q.is_contiguous():
        raise ValueError("Input tensors q must be contiguous")
    if not k.is_contiguous():
        raise ValueError("Input tensors k must be contiguous")
    if not q_s.is_contiguous():
        raise ValueError("Input tensors q_s must be contiguous")
    if not k_s.is_contiguous():
        raise ValueError("Input tensors k_s must be contiguous")

    query = q.to(torch.bfloat16) # [b, s, h, d]
    query_scale = q_s.unsqueeze(-1).to(torch.float32) # [b, s, h, 1]

    key = k.to(torch.bfloat16) 
    key = key.transpose(-2, -1).unsqueeze(1) # [b, 1, d, m]
    key_scale = k_s.unsqueeze(1).to(torch.float32) # [b, 1, m]

    logits = torch.matmul(query, key).to(torch.float32) # [b, s, h, m]

    logits = torch.nn.functional.relu(logits) # [b, s, h, m]

    logits = logits * query_scale  # [b, s, h, m]

    logits_sum = torch.sum(logits, dim=2) # [b, s, m]

    index_score = logits_sum * key_scale # [b, s, m]

    return index_score
