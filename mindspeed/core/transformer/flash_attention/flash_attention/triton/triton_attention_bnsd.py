# need new version CANN(>8.3) & triton_npu(test with 3.2.0.dev20250929)
import os
import logging
import math

import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
logger = logging.getLogger(__name__)


@triton.jit
def _triton_flash_attention_forward(
        query, key, value, M, L, output, sm_scale,
        stride_B, stride_N, stride_S, stride_D,
        B: tl.constexpr, 
        N: tl.constexpr,
        S: tl.constexpr, 
        D: tl.constexpr,
        Bc: tl.constexpr,
        Br: tl.constexpr,
        causal: tl.constexpr
    ):
    """
        Performs the forward pass of Flash Attention using Triton kernels.
        This function computes the attention output for a batch of queries, keys, and values using a block-wise,
        numerically stable softmax computation. It supports both causal and non-causal attention, and is optimized
        for high performance on GPUs using Triton.
        Args:
            query (tl.tensor): The input query tensor of shape [B, N, S, D].
            key (tl.tensor): The input key tensor of shape [B, N, S, D].
            value (tl.tensor): The input value tensor of shape [B, N, S, D].
            M (tl.tensor): Output tensor to store logsumexp values for backward pass, shape [B, N, S].
            L (tl.tensor): Unused in this function, reserved for compatibility.
            output (tl.tensor): Output tensor to store the attention results, shape [B, N, S, D].
            sm_scale (float): Scaling factor applied to the attention logits (typically 1/sqrt(D)).
            stride_B (int): Stride for batch dimension.
            stride_N (int): Stride for head dimension.
            stride_S (int): Stride for sequence dimension.
            stride_D (int): Stride for feature dimension.
            B (tl.constexpr): Batch size.
            N (tl.constexpr): Number of attention heads.
            S (tl.constexpr): Sequence length.
            D (tl.constexpr): Feature dimension (head size).
            Bc (tl.constexpr): Block size for keys/values.
            Br (tl.constexpr): Block size for queries.
            causal (tl.constexpr): Whether to apply causal masking (True for autoregressive models).
        Returns:
            None. The results are written in-place to the `output` and `M` tensors.
        Notes:
            - The logsumexp values are stored in `M` for use in the backward pass.
            - The function supports both causal and non-causal attention modes.
    """
    
    # Current M-dimension block index
    start_r = tl.program_id(0)

    # Loop through all (B * N) attention groups
    for off_BN in range(0, B * N):
        # Compute batch and head index
        off_B = off_BN // N
        off_N = off_BN % N

        # Offset for current batch and head
        qkv_offset = off_B.to(tl.int64) * stride_B + off_N.to(tl.int64) * stride_N

        # Construct Q block pointer (BLOCK_M, HEAD_DIM)
        query_block_ptr = tl.make_block_ptr(
            base=query + qkv_offset, 
            shape=(S, D), 
            strides=(stride_S, stride_D), 
            offsets=(start_r * Br, 0), 
            block_shape=(Br, D), 
            order=(1, 0), 
        )

        # Construct V block pointer (starts from 0,0; advanced in inner)
        value_block_ptr = tl.make_block_ptr(
            base=value + qkv_offset, 
            shape=(S, D), 
            strides=(stride_S, stride_D), 
            offsets=(0, 0), 
            block_shape=(Bc, D), 
            order=(1, 0), 
        )

        # Construct K block pointer
        key_block_ptr = tl.make_block_ptr(
            base=key + qkv_offset, 
            shape=(S, D), 
            strides=(stride_S, stride_D), 
            offsets=(0, 0), 
            block_shape=(Bc, D), 
            order=(1, 0), 
        )

        # Construct output block pointer
        O_block_ptr = tl.make_block_ptr(
            base=output + qkv_offset, 
            shape=(S, D), 
            strides=(stride_S, stride_D), 
            offsets=(start_r * Br, 0), 
            block_shape=(Br, D), 
            order=(1, 0), 
        )

        # offset indices for q&o
        offs_r = start_r * Br + tl.arange(0, Br)
        # offset indices for k&v
        offs_c = tl.arange(0, Bc)

        # m_i and l_i for online softmax 
        m_i = tl.zeros([Br], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([Br], dtype=tl.float32) + 1.0
        # accumulate for block output
        acc = tl.zeros([Br, D], dtype=tl.float32)

        # Apply softmax scale and convert to log2 base
        # change e^(x) to 2^(x):
        #   exp(x) = e^x = (2^(log2(e)))^x = 2^(1/ln2)^x = 2^(x/ln2)
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1 / log(2)

        # Load current Q block
        q = tl.load(query_block_ptr)

        # snot causal, all block attention computation
        # causal, [0 ~ Diagonal block)
        acc, l_i, m_i = _triton_flash_attention_forward_inner(
                            acc=acc, l_i=l_i, m_i=m_i, q=q, 
                            key_block_ptr=key_block_ptr, 
                            value_block_ptr=value_block_ptr, 
                            start_r=start_r, qk_scale=qk_scale, 
                            Br=Br, Bc=Bc, 
                            causal=causal, 
                            causal_diagonal_block=False, 
                            offs_r=offs_r, 
                            offs_c=offs_c, 
                            S=S
                            )
        
        if causal:
            # for causal, Diagonal block attention computation
            acc, l_i, m_i = _triton_flash_attention_forward_inner(
                                acc=acc, l_i=l_i, m_i=m_i, q=q, 
                                key_block_ptr=key_block_ptr, 
                                value_block_ptr=value_block_ptr, 
                                start_r=start_r, qk_scale=qk_scale, 
                                Br=Br, Bc=Bc, 
                                causal=causal, 
                                causal_diagonal_block=True, 
                                offs_r=offs_r, 
                                offs_c=offs_c, 
                                S=S
                                )

        # compute logsumexp for backward
        # m_i means
        #     = m_i + log2(sum(exp(x_i - m_i)) ) 
        #     = log2(m_i * sum(exp(x_i - m_i)) )
        #     = log2-sum-exp(x_i) # for backward
        m_i = m_i + tl.math.log2(l_i)  
        # normalize output
        acc = acc / l_i[:, None]

        # Store logsumexp to M 
        m_ptrs = M + off_BN * S + offs_r
        tl.store(m_ptrs, m_i)
        # Store final output to output
        tl.store(O_block_ptr, acc.to(output.type.element_ty))


@triton.jit
def _triton_flash_attention_forward_inner(
        acc, l_i, m_i, q, # Accumulator, local l, local m, query vector
        key_block_ptr, value_block_ptr, # Key and value block pointers for current stage
        start_r, # Starting row position of current query block, 
        qk_scale: tl.constexpr, 
        Br: tl.constexpr, 
        Bc: tl.constexpr, 
        causal: tl.constexpr, 
        causal_diagonal_block: tl.constexpr, 
        offs_r: tl.constexpr, 
        offs_c: tl.constexpr, # Current stage flag, m and n offset indices
        S: tl.constexpr # Total context length, whether to enable FP8 for value precision
        ): 
    """
        Performs the inner loop of the Triton-based Flash Attention forward pass for a single query block.
        This function computes the attention output for a block of queries using block-wise matrix multiplication,
        softmax normalization, and optional causal masking. It processes the attention in blocks to optimize memory
        access and computational efficiency on GPUs.
        Args:
            acc: Accumulator tensor for the output, shape [Br, D].
            l_i: Softmax denominator for each query row, shape [Br].
            m_i: Maximum value for numerical stability in softmax, shape [Br].
            q: Query block, shape [Br, D].
            key_block_ptr: Pointer to the current key block, shape [S, D].
            value_block_ptr: Pointer to the current value block, shape [S, D].
            start_r: Starting row index of the current query block.
            qk_scale (tl.constexpr): Scaling factor for the QK^T product.
            Br (tl.constexpr): Block size for rows (number of queries per block).
            Bc (tl.constexpr): Block size for columns (number of keys/values per block).
            causal (tl.constexpr): Whether to apply causal masking (prevents attending to future positions).
            causal_diagonal_block (tl.constexpr): Whether the current block is on the causal diagonal.
            offs_r (tl.constexpr): Row offsets for the current block.
            offs_c (tl.constexpr): Column offsets for the current block.
            S (tl.constexpr): Total sequence/context length.
        Returns:
            Tuple of (acc, l_i, m_i):
                acc: Updated accumulator with the attention output for the block, shape [Br, D].
                l_i: Updated softmax denominator for each query row, shape [Br].
                m_i: Updated maximum value for numerical stability in softmax, shape [Br].
        Notes:
            - The function supports both causal and non-causal attention.
            - Uses block-wise computation for efficiency.
            - Applies numerical stabilization for softmax using the max trick.
            - Designed for use within a Triton kernel for high-performance attention computation.
    """

    if causal:
        if not causal_diagonal_block:
            lo, hi = 0, start_r * Br
        if causal_diagonal_block:
            lo, hi = start_r * Br, (start_r + 1) * Br
            lo = tl.multiple_of(lo, Br)  # Align starting position
    else:
        if not causal_diagonal_block:
            lo, hi = 0, S
        else:
            raise RuntimeError("Should not enter here.")

    # Adjust K and V block pointers to the starting position `lo`
    key_block_ptr = tl.advance(key_block_ptr, (lo, 0))  # K is [D, S], shift along the second dim by lo
    value_block_ptr = tl.advance(value_block_ptr, (lo, 0))  # V is [S, D], shift along the first dim by lo

    # row fixed as [start_r, start_r + Br), 
    # loop for column [lo, hi] step by Bc
    for start_c in range(lo, hi, Bc):
        start_c = tl.multiple_of(start_c, Bc) # Align column start position
        # -- Compute qk ----
        k = tl.load(key_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        qk = qk * qk_scale
        # Apply causal mask for causal_diagonal_block
        if causal and causal_diagonal_block:
            mask = offs_r[:, None] >= (start_c + offs_c[None, :]) # Construct upper triangular mask
            qk = tl.where(mask, qk, -1.0e6) # Set invalid positions to -∞
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]  # Stabilize
        # Softmax weights middle value: p = exp2(qk - m_ij), div m_i in the end
        p = tl.math.exp2(qk)  # Use base-2 exponent for better numerical stability
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)

        # Update factor: exp difference between old and new max
        alpha = tl.math.exp2(m_i - m_ij)  
        # Update softmax denominator
        l_i = l_i * alpha + l_ij
        # Scale accumulator by alpha to maintain consistency
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(value_block_ptr)  
        p = p.to(tl.float16)
        # Multiply softmax weights with V and accumulate to acc
        acc = tl.dot(p, v, acc)
        # Update current block max
        m_i = m_ij
        # Advance V and K block pointers to next Bc range
        key_block_ptr = tl.advance(key_block_ptr, (Bc, 0))
        value_block_ptr = tl.advance(value_block_ptr, (Bc, 0))
    return acc, l_i, m_i  # Return accumulated output acc, softmax denominator l_i, and max value m_i


@triton.jit
def _triton_flash_attention_backward_preprocess(
        output, 
        d_output, 
        Delta, 
        S: tl.constexpr, 
        D: tl.constexpr, 
        Br: tl.constexpr,  
        ):
    off_r = tl.program_id(0) * Br + tl.arange(0, Br)
    off_bn = tl.program_id(1)
    off_c = tl.arange(0, D)
    # load
    output = tl.load(output + off_bn * S * D + off_r[:, None] * D + off_c[None, :])
    d_output = tl.load(d_output + off_bn * S * D + off_r[:, None] * D + off_c[None, :]).to(tl.float32)
    delta = tl.sum(output * d_output, axis=1)
    # write-back
    tl.store(Delta + off_bn * S + off_r, delta)


@triton.jit
def _backward_for_dk_dv(
        dk, dv, 
        Q, k, v, 
        DO, 
        M, Delta, 
        # shared by Q/K/V/DO.
        stride_S, stride_D, 
        BLOCK_R1: tl.constexpr, 
        BLOCK_C1: tl.constexpr, 
        D: tl.constexpr, 
        # Filled in by the wrapper.
        start_c, start_r, num_steps, 
        causal):
    """
    Performs the backward pass computation for the gradients of the key (dk) and value (dv) tensors in a flash attention mechanism.
    This function iterates over blocks of the input sequence and computes the gradients with respect to the key and value tensors using the provided query (Q), key (k), value (v), and output gradient (DO) tensors. It supports optional autoregressive masking and is designed for use with Triton kernels for efficient GPU execution.
    Args:
        dk: Output tensor for the gradient with respect to the key tensor.
        dv: Output tensor for the gradient with respect to the value tensor.
        Q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        DO: Gradient of the output tensor.
        M: Tensor containing the log-sum-exp values for numerical stability.
        Delta: Tensor containing precomputed delta values for scaling.
        stride_S: Stride for the sequence dimension.
        stride_D: Stride for the feature dimension.
        BLOCK_R1 (tl.constexpr): Block size for the row dimension.
        BLOCK_C1 (tl.constexpr): Block size for the column dimension.
        D (tl.constexpr): Feature dimension size.
        start_c: Starting column index for the block.
        start_r: Starting row index for the block.
        num_steps: Number of block steps to process.
        causal: Boolean flag indicating whether to apply autoregressive masking.
    Returns:
        Tuple of (dk, dv): The updated gradients with respect to the key and value tensors.
    """
    offs_r = start_r + tl.arange(0, BLOCK_R1)
    offs_c = start_c + tl.arange(0, BLOCK_C1)
    offs_k = tl.arange(0, D)
    q_ptrs = Q + offs_r[:, None] * stride_S + offs_k[None, :] * stride_D
    do_ptrs = DO + offs_r[:, None] * stride_S + offs_k[None, :] * stride_D
    # BLOCK_C1 must be a multiple of BLOCK_R1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_C1 % BLOCK_R1 == 0)
    curr_r = start_r
    step_r = BLOCK_R1
    for _ in range(num_steps):
        q = tl.load(q_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_r = curr_r + tl.arange(0, BLOCK_R1)
        m = tl.load(M + offs_r)
        qk_T = tl.dot(k, tl.trans(q))
        # m means m_i + tl.math.log2(l_i), aka, logsumexp of P_row
        # p_T means
        #    = exp2(qKT - logsumexp(x))
        #    = exp2(qKT) / sum-exp(x)
        p_T = tl.math.exp2(qk_T - m[None, :])
        # Autoregressive masking.
        if causal:
            mask = (offs_r[None, :] >= offs_c[:, None])
            p_T = tl.where(mask, p_T, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        pp_T = p_T
        pp_T = pp_T.to(tl.float16)
        dv = dv + tl.dot(pp_T, do) 
        # D ( = delta) is pre-divided by ds_scale.
        Di = tl.load(Delta + offs_r)
        # Compute dP and dS.
        dp_T = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = p_T * (dp_T - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, q)
        # Increment pointers.
        curr_r += step_r
        q_ptrs += step_r * stride_S
        do_ptrs += step_r * stride_S
    return dk, dv


@triton.jit
def _backward_for_dq(
        dq, q, K, V, 
        do, m, Delta, 
        # shared by Q/K/V/DO.
        stride_S, stride_D, 
        BLOCK_R2: tl.constexpr, 
        BLOCK_C2: tl.constexpr, 
        D: tl.constexpr, 
        start_r, start_c, num_steps, 
        causal: tl.constexpr):
    """
    Performs the backward pass computation for the gradient with respect to the query (dq) in a block-wise attention mechanism, optimized for Triton kernels.
    Args:
        dq: The gradient tensor with respect to the query, to be updated in-place.
        q: The query tensor.
        K: The key tensor.
        V: The value tensor.
        do: The gradient tensor with respect to the output.
        m: The maximum logits tensor used for numerical stability.
        Delta: The precomputed delta tensor (pre-divided by ds_scale).
        stride_S: The stride for the sequence dimension.
        stride_D: The stride for the feature dimension.
        BLOCK_R2 (tl.constexpr): Block size for the row dimension.
        BLOCK_C2 (tl.constexpr): Block size for the column dimension.
        D (tl.constexpr): Feature dimension size.
        start_r: Starting index for the row block.
        start_c: Starting index for the column block.
        num_steps: Number of block steps to iterate over.
        causal (tl.constexpr): Whether to apply causal masking (autoregressive).
    Returns:
        Updated dq tensor after accumulating the gradients with respect to the query.
    Notes:
        - Assumes BLOCK_R2 is a multiple of BLOCK_C2.
        - Designed for use within Triton kernels for efficient block-wise computation.
        - Applies causal masking if specified.
    """

    offs_r = start_r + tl.arange(0, BLOCK_R2)
    offs_c = start_c + tl.arange(0, BLOCK_C2)
    offs_k = tl.arange(0, D)
    k_ptrs = K + offs_c[:, None] * stride_S + offs_k[None, :] * stride_D
    v_ptrs = V + offs_c[:, None] * stride_S + offs_k[None, :] * stride_D
    # D ( = delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_r)
    # BLOCK_R2 must be a multiple of BLOCK_C2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_R2 % BLOCK_C2 == 0)
    curr_c = start_c
    step_c = BLOCK_C2
    for _ in range(num_steps):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k))
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if causal:
            offs_c = curr_c + tl.arange(0, BLOCK_C2)
            mask = (offs_r[:, None] >= offs_c[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        v = tl.load(v_ptrs)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, k)
        # Increment pointers.
        curr_c += step_c
        k_ptrs += step_c * stride_S
        v_ptrs += step_c * stride_S
    return dq


@triton.jit
def _triton_flash_attention_backward(
        Q, K, V, 
        sm_scale, 
        DO, 
        DQ, DK, DV, 
        M, # logsumexp
        Delta, 
        # shared by Q/K/V/DO.
        stride_B: tl.constexpr, 
        stride_N: tl.constexpr, 
        stride_S: tl.constexpr, 
        stride_D: tl.constexpr, 
        N: tl.constexpr, 
        S: tl.constexpr, 
        D: tl.constexpr, 
        BLOCK_R1: tl.constexpr, 
        BLOCK_C1: tl.constexpr, 
        BLOCK_R2: tl.constexpr, 
        BLOCK_C2: tl.constexpr, 
        BLOCK_SLICE_FACTOR: tl.constexpr, 
        causal: tl.constexpr
        ):
    """
    Computes the backward pass for the Flash Attention mechanism using Triton kernels.
    This function calculates the gradients with respect to the query (Q), key (K), and value (V) tensors,
    as well as the output gradients (DO), for a block-sparse attention pattern. It supports both causal and
    non-causal attention, and is optimized for efficient memory access and parallelism.
    Args:
        Q: Pointer to the query tensor of shape [B, N, S, D].
        K: Pointer to the key tensor of shape [B, N, S, D].
        V: Pointer to the value tensor of shape [B, N, S, D].
        sm_scale: Scaling factor applied to the attention logits.
        DO: Pointer to the output gradient tensor of shape [B, N, S, D].
        DQ: Pointer to the output tensor for the gradient with respect to Q.
        DK: Pointer to the output tensor for the gradient with respect to K.
        DV: Pointer to the output tensor for the gradient with respect to V.
        M: Pointer to the logsumexp tensor (used for numerical stability).
        Delta: Pointer to the auxiliary tensor used in the backward computation.
        stride_B (tl.constexpr): Stride between batches.
        stride_N (tl.constexpr): Stride between attention heads.
        stride_S (tl.constexpr): Stride between sequence positions.
        stride_D (tl.constexpr): Stride between feature dimensions.
        N (tl.constexpr): Number of attention heads.
        S (tl.constexpr): Sequence length.
        D (tl.constexpr): Feature dimension.
        BLOCK_R1 (tl.constexpr): Block size for rows for d_value and d_key.
        BLOCK_C1 (tl.constexpr): Block size for columns for d_value and d_key.
        BLOCK_R2 (tl.constexpr): Block size for rows for d_query.
        BLOCK_C2 (tl.constexpr): Block size for columns for d_query.
        BLOCK_SLICE_FACTOR (tl.constexpr): Factor for slicing blocks.
        causal (tl.constexpr): Whether to apply causal masking (True for causal attention).
    Returns:
        None. The gradients are written in-place to the provided DQ, DK, and DV pointers.
    Notes:
        - This function is intended to be called as a Triton kernel.
        - It assumes the input tensors are properly laid out in memory and that the strides are set correctly.
        - The function uses block-wise computation for efficiency and supports both causal and non-causal attention.
    """
    # GRID is set as (S // BLOCK_C1, 1, B * N)
    BN_id = tl.program_id(2)
    off_chz = (BN_id * S).to(tl.int64)
    offset = ((BN_id // N) * stride_B + (BN_id % N) * stride_N).to(tl.int64)
    Bc_id = tl.program_id(0)

    # offset pointers for batch/head
    Q += offset
    K += offset
    V += offset
    DO += offset
    DQ += offset
    DK += offset
    DV += offset
    M += off_chz
    Delta += off_chz

    dv = tl.zeros([BLOCK_C1, D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_C1, D], dtype=tl.float32)
    
    # load scales
    offs_k = tl.arange(0, D)

    start_c = Bc_id * BLOCK_C1
    start_r = start_c if causal else 0

    MASK_BLOCK_R1: tl.constexpr = BLOCK_R1 // BLOCK_SLICE_FACTOR
    offs_c = start_c + tl.arange(0, BLOCK_C1)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_c[:, None] * stride_S + offs_k[None, :] * stride_D)
    v = tl.load(V + offs_c[:, None] * stride_S + offs_k[None, :] * stride_D)
    
    if causal:
        num_steps = BLOCK_C1 // MASK_BLOCK_R1
        dk, dv = _backward_for_dk_dv(
                    dk=dk, 
                    dv=dv, 
                    Q=Q, 
                    k=k, 
                    v=v, 
                    DO=DO, 
                    M=M, 
                    D=D, 
                    Delta=Delta, 
                    stride_S=stride_S, 
                    stride_D=stride_D, 
                    BLOCK_R1=MASK_BLOCK_R1, 
                    BLOCK_C1=BLOCK_C1, 
                    start_c=start_c, 
                    start_r=start_r, # start_r=start_c if causal
                    num_steps=num_steps, 
                    causal=causal
                    )
        start_r += num_steps * MASK_BLOCK_R1
        
    num_steps = (S - start_r) // BLOCK_R1
    # Compute dK and dV for non-masked blocks.
    dk, dv = _backward_for_dk_dv(
                dk=dk, dv=dv, 
                Q=Q, k=k, v=v, 
                DO=DO, 
                M=M, 
                D=D, 
                Delta=Delta, 
                stride_S=stride_S, 
                stride_D=stride_D, 
                BLOCK_R1=BLOCK_R1, 
                BLOCK_C1=BLOCK_C1, 
                start_c=start_c, 
                start_r=start_r, 
                num_steps=num_steps, 
                causal=False
                )

    dv_ptrs = DV + offs_c[:, None] * stride_S + offs_k[None, :] * stride_D
    tl.store(dv_ptrs, dv)
    dv = None

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_c[:, None] * stride_S + offs_k[None, :] * stride_D
    tl.store(dk_ptrs, dk)
    dk = None

    # THIS BLOCK DOES DQ:
    start_r = Bc_id * BLOCK_R2
    end_n = (start_r + BLOCK_R2) if causal else (S)

    MASK_BLOCK_C2: tl.constexpr = BLOCK_C2 // BLOCK_SLICE_FACTOR
    offs_r = start_r + tl.arange(0, BLOCK_R2)

    q = tl.load(Q + offs_r[:, None] * stride_S + offs_k[None, :] * stride_D)
    dq = tl.zeros([BLOCK_R2, D], dtype=tl.float32)
    do = tl.load(DO + offs_r[:, None] * stride_S + offs_k[None, :] * stride_D)

    m = tl.load(M + offs_r)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _backward_for_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    if causal:
        num_steps = BLOCK_R2 // MASK_BLOCK_C2
        dq = _backward_for_dq(
                dq=dq, q=q, K=K, V=V, 
                do=do, m=m, Delta=Delta, 
                stride_S=stride_S, stride_D=stride_D, 
                BLOCK_R2=BLOCK_R2, 
                BLOCK_C2=MASK_BLOCK_C2, 
                D=D, 
                start_r=start_r, 
                start_c=end_n - num_steps * MASK_BLOCK_C2, 
                num_steps=num_steps, 
                causal=causal
                )
        end_n -= num_steps * MASK_BLOCK_C2
    # stage 2
    num_steps = end_n // BLOCK_C2
    dq = _backward_for_dq(
            dq=dq, q=q, K=K, V=V, 
            do=do, m=m, Delta=Delta, 
            stride_S=stride_S, stride_D=stride_D, 
            BLOCK_R2=BLOCK_R2, 
            BLOCK_C2=MASK_BLOCK_C2, 
            D=D, 
            start_r=start_r, 
            start_c=end_n - num_steps * BLOCK_C2, 
            num_steps=num_steps, 
            causal=False
            )
    # Write back dQ.
    dq_ptrs = DQ + offs_r[:, None] * stride_S + offs_k[None, :] * stride_D
    LN2 = math.log(2)
    dq *= LN2
    tl.store(dq_ptrs, dq)


def forward_ub_memory_used_simulation(Br, Bc, B, N, S, D, causal):
    '''
    calculate best of Br and Bc, 
    for A2, UB 192KB, L2 192MB
    
    HBM memory:
        M       [B, N, S]
        L       [B, N, S]
        output  [B, N, S, D]
    '''
    ub_men = (
        # for row
        + Br * D * 4 # acc
        + Br * D * 4 # q block
            # for block
            + Br * Bc * 4 # qk block
            + max(
                Bc * D * 4, # k block
                Bc * D * 4, # v block
                ((Br * Bc * 4 * 2) if causal else 0) # qk mask
            )
        )
    return ub_men


def backward_ub_memory_used_simulation(R1, C1, R2, C2, B, N, S, D, causal):
    ub_men_base = (
        # for row
        + C1 * D * 4 # k
        + C1 * D * 4 # v
    )
    ub_mem_dvdv = (
        + C1 * D * 4 # dk
        + C1 * D * 4 # dv
            # dk*dv
            + R1 * 4 # m
            + R1 * 4 # Di
            + R1 * D * 4 # q
            + R1 * C1 * 4 # p_T
            + max(
                R1 * D * 4, # do
                ((R1 * C1 * 4 * 1) if causal else 0) # p_T mask
            )
    )
    
    ub_mem_dq = (
        + R2 * D * 4 # q
        + R2 * D * 4 # dq
        + R2 * D * 4 # do
        + R2 * 4 # m
            # dq
            + R2 * 4 # Di
            + R2 * C2 * 4 # q
            + max(
                C2 * D * 4, # k
                C2 * D * 4, # v
                ((R2 * C2 * 4 * 1) if causal else 0) # p_T mask
            )
    )
    ub_men = ub_men_base + max(ub_mem_dvdv, ub_mem_dq)
    return ub_men


class TritonFlashAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query, key, value, causal, sm_scale):
        # batch_size
        B = query.shape[0] 
        # number of head
        N = query.shape[1] 
        # sequence length
        S = query.shape[2] 
        # dimension of each head
        D = query.shape[3] 
        # stride of BNSD
        stride_B = query.stride(0)
        stride_N = query.stride(1) 
        stride_S = query.stride(2) 
        stride_D = query.stride(3)
        
        # assert contiguous 
        if not query.is_contiguous():
            raise ValueError("query tensor is not contiguous")
        if not key.is_contiguous():
            raise ValueError("key tensor is not contiguous")
        if not value.is_contiguous():
            raise ValueError("value tensor is not contiguous")
        
        # assert same dim
        if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
            raise ValueError("The last dimension of query, key, and value tensors must be the same")

        # same stride assert.
        for i in range(0, 4):
            if query.stride(i) != key.stride(i) or key.stride(i) != value.stride(i):
                raise ValueError(f"The stride at dimension {i} of query, key, and value tensors must be the same")
        
        # block of r (for query and output) Br in paper
        Br = 64
        # block of c (for key and value) Bc in paper
        Bc = 32
        if S % Br != 0:
            raise ValueError(f"S ({S}) must be divisible by Br ({Br})")
        if S % Bc != 0:
            raise ValueError(f"S ({S}) must be divisible by Bc ({Bc})")
        
        Brs = [16, 32, 64, 128]
        Bcs = [16, 32, 64, 128]
        MAX_UB_MEM = 192 * 1024
        UB_MEM_WATERMARK_RATIO = 0.3
        UB_MEM_LIMIT = MAX_UB_MEM * UB_MEM_WATERMARK_RATIO
        max_Br_mul_Bc = Br * Bc
        min_Br_minus_Bc = abs(Br - Bc)
        best_perf_ub_mem = forward_ub_memory_used_simulation(Br, Bc, B, N, S, D, causal)
        for one_Br in Brs:
            if one_Br < S and S % one_Br == 0:
                for one_Bc in Bcs:
                    if (one_Bc < S and S % one_Bc == 0 
                        and (one_Br * one_Bc >= max_Br_mul_Bc # bigger block
                            or 
                                # more square block
                                (one_Br * one_Bc == max_Br_mul_Bc 
                                    and min_Br_minus_Bc > abs(one_Br - one_Bc))
                            )
                        and (one_Br % one_Bc == 0)
                        ):
                        ub_men_simulated = forward_ub_memory_used_simulation(Br, Bc, B, N, S, D, causal)
                        if ub_men_simulated <= UB_MEM_LIMIT:
                            max_Br_mul_Bc = one_Br * one_Bc
                            min_Br_minus_Bc = abs(one_Br - one_Bc)
                            best_perf_ub_mem = ub_men_simulated
                            Br = one_Br
                            Bc = one_Bc

        grid = (triton.cdiv(S, Br), 1, 1)
        logger.info(f"triton FA forward B:{B}, N:{N}, S:{S}, D:{D}, Br:{Br}, Bc:{Bc}, grid:{grid}, ub_mem_simulated: {best_perf_ub_mem}<{UB_MEM_LIMIT:.1f}")

        # M, L in FlashAttentionV2 paper.
        # in each block compute, m_i save max of m_ij
        # finally, m_i save log2-sum-exp(x_i) for backward
        M = torch.empty((B, N, S), device=query.device, dtype=torch.float32)
        # here L means sum(exp(x_i - m_i))
        L = torch.empty((B, N, S), device=query.device, dtype=torch.float32)
        
        # output of FA
        output = torch.empty_like(query)
        
        # triton FA forward
        _triton_flash_attention_forward[grid](
            query=query, 
            key=key, 
            value=value, 
            M=M, 
            L=L, 
            output=output, 
            sm_scale=sm_scale, 
            stride_B=stride_B, 
            stride_N=stride_N, 
            stride_S=stride_S, 
            stride_D=stride_D, 
            B=B, 
            N=N, 
            S=S, 
            D=D, 
            Br=Br, # autotune
            Bc=Bc, # autotune
            causal=causal, 
        )
        '''
            multibuffer=True, 
            unit_flag=True, 
            limit_auto_multi_buffer_only_for_local_buffer=False, 
            tile_mix_vector_loop=4, 
            tile_mix_cube_loop=4, 
            set_workspace_multibuffer=4, 
            )
        '''

        ctx.save_for_backward(query, key, value, output, M)
        ctx.sm_scale = sm_scale
        ctx.D = D
        ctx.causal = causal
        return output
    
    @staticmethod
    def backward(ctx, d_output):
        query, key, value, output, M = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale
        # batch_size
        B = query.shape[0] 
        # number of head
        N = query.shape[1] 
        # sequence length
        S = query.shape[2] 
        # dimension of each head
        D = query.shape[3] 
        # stride of BNSD
        stride_B = query.stride(0)
        stride_N = query.stride(1)
        stride_S = query.stride(2)
        stride_D = query.stride(3)
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        BLOCK_1 = 64
        BLOCK_2 = 64
        BLOCK_R1 = min(BLOCK_1, S) # block size of row for d_value、d_key
        BLOCK_C1 = min(BLOCK_1, S) # block size of column for d_value、d_key
        BLOCK_R2 = min(BLOCK_2, S) # block size of row for d_query
        BLOCK_C2 = min(BLOCK_2, S) # block size of column for d_query
        if S % BLOCK_R1 != 0:
            raise ValueError(f"S ({S}) must be divisible by BLOCK_R1 ({BLOCK_R1})")
        if S % BLOCK_C1 != 0:
            raise ValueError(f"S ({S}) must be divisible by BLOCK_C1 ({BLOCK_C1})")
        if S % BLOCK_R2 != 0:
            raise ValueError(f"S ({S}) must be divisible by BLOCK_R2 ({BLOCK_R2})")
        if S % BLOCK_C2 != 0:
            raise ValueError(f"S ({S}) must be divisible by BLOCK_C2 ({BLOCK_C2})")
        if BLOCK_C1 % BLOCK_R1 != 0:
            raise ValueError(f"BLOCK_C1 ({BLOCK_C1}) must be divisible by BLOCK_R1 ({BLOCK_R1})")
        
        R1s = [16, 32, 64, 128]
        C1s = [16, 32, 64, 128]
        R2s = [16, 32, 64, 128]
        C2s = [16, 32, 64, 128]
        MAX_UB_MEM = 192 * 1024
        UB_MEM_WATERMARK_RATIO = 0.7
        UB_MEM_LIMIT = MAX_UB_MEM * UB_MEM_WATERMARK_RATIO
        max_R1_mul_C1 = BLOCK_R1 * BLOCK_C1
        min_R1_minus_C1 = abs(BLOCK_R1 - BLOCK_C1)
        max_R2_mul_C2 = BLOCK_R2 * BLOCK_C2
        min_R2_minus_C2 = abs(BLOCK_R2 - BLOCK_C2)
        best_perf_ub_mem = backward_ub_memory_used_simulation(BLOCK_R1, BLOCK_C1, BLOCK_R2, BLOCK_C2, B, N, S, D, causal)
        for one_R1 in R1s:
            if one_R1 < S and S % one_R1 == 0:
                for one_C1 in C1s:
                    if (one_C1 < S and S % one_C1 == 0 
                    and (one_R1 * one_C1 >= max_R1_mul_C1 # bigger block
                        or 
                            # more square block
                            (one_R1 * one_C1 == max_R1_mul_C1 
                                and min_R1_minus_C1 > abs(one_R1 - one_C1))
                        )
                    and (one_R1 % one_C1 == 0)):
                        for one_R2 in R2s:
                            if one_R2 < S and S % one_R2 == 0:
                                for one_C2 in C2s:
                                    if (one_C2 < S and S % one_C2 == 0 
                                    and (one_R2 * one_C2 >= max_R2_mul_C2 # bigger block
                                        or 
                                            # more square block
                                            (one_R2 * one_C2 == max_R2_mul_C2
                                                and min_R1_minus_C2 > abs(one_R2 - one_C2))
                                        )
                                    and (one_R2 % one_C2 == 0)):
                                        ub_men_simulated = backward_ub_memory_used_simulation(BLOCK_R1, BLOCK_C1, BLOCK_R2, BLOCK_C2, B, N, S, D, causal)
                                        if ub_men_simulated <= UB_MEM_LIMIT:
                                            max_R2_mul_C2 = one_R2 * one_C2
                                            min_R1_minus_C2 = abs(one_R2 - one_C2)
                                            max_R1_mul_C1 = one_R1 * one_C1
                                            min_R1_minus_C1 = abs(one_R1 - one_C1)
                                            best_perf_ub_mem = ub_men_simulated
                                            BLOCK_R1 = one_R1
                                            BLOCK_C1 = one_C1
                                            BLOCK_R2 = one_R2
                                            BLOCK_C2 = one_C2
        BLOCK_SLICE_FACTOR = 1
        RCP_LN2 = 1.44269504
        scaled_key = key
        qk_scale = sm_scale * RCP_LN2
        scaled_key = scaled_key * qk_scale
        PRE_BLOCK = 128
        PRE_BLOCK = min(S, PRE_BLOCK)
        if S % PRE_BLOCK != 0:
            raise ValueError(f"S ({S}) must be divisible by PRE_BLOCK ({PRE_BLOCK})")
        pre_grid = (S // PRE_BLOCK, B * N)
        grid = (S // BLOCK_C1, 1, B * N)
        delta = torch.empty_like(M)
        logger.info(
            f"backward B:{B}, N:{N}, S:{S}, D:{D}, "
            f"BLOCK_R1:{BLOCK_R1}, BLOCK_C1:{BLOCK_C1}, BLOCK_R2:{BLOCK_R2}, BLOCK_C2:{BLOCK_C2}, "
            f"pre_grid:{pre_grid},grid:{grid},BLOCK_SLICE_FACTOR:{BLOCK_SLICE_FACTOR}, "
            f"stride: {(query.stride(0), query.stride(1), query.stride(2), query.stride(3))}, "
            f"ub_mem_simulated: {best_perf_ub_mem}<{UB_MEM_LIMIT:.1f}"
            )
        _triton_flash_attention_backward_preprocess[pre_grid](
            output=output, 
            d_output=d_output, 
            Delta=delta, 
            S=S, 
            D=D, 
            Br=PRE_BLOCK, 
        )
        _triton_flash_attention_backward[grid](
            Q=query, 
            K=scaled_key, 
            V=value, 
            sm_scale=sm_scale, 
            DO=d_output, 
            DQ=d_query, 
            DK=d_key, 
            DV=d_value, 
            M=M, # logsumexp
            Delta=delta,   
            stride_B=stride_B,  
            stride_N=stride_N, 
            stride_S=stride_S, 
            stride_D=stride_D,  
            N=N, 
            S=S, 
            D=D, 
            BLOCK_R1=BLOCK_R1, # autotune
            BLOCK_C1=BLOCK_C1, # autotune
            BLOCK_R2=BLOCK_R2, # autotune
            BLOCK_C2=BLOCK_C2, # autotune
            BLOCK_SLICE_FACTOR=BLOCK_SLICE_FACTOR, 
            causal=causal
        )
        '''
            multibuffer=True, 
            unit_flag=True, 
            limit_auto_multi_buffer_only_for_local_buffer=False, 
            tile_mix_vector_loop=4, 
            tile_mix_cube_loop=4, 
            set_workspace_multibuffer=4, 
            
        )
        '''

        return d_query, d_key, d_value, None, None, None, None


attention = TritonFlashAttentionFunction.apply


def test_op(B, N, S, D, causal, dtype):
    import time
    DEVICE = "npu"
    torch.manual_seed(20)
    q = (torch.empty((B, N, S, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=math.sqrt(1 / S)).requires_grad_())
    k = (torch.empty((B, N, S, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=math.sqrt(1 / S)).requires_grad_())
    v = (torch.empty((B, N, S, D), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=math.sqrt(1 / S)).requires_grad_())
    sm_scale = 1
    dout = torch.randn_like(q)

    logger.info(f"Start Test Triton-Attention with ({B},{N},{S},{D}), causal = {causal}, dtype = {dtype}")
    
    # first run for compile
    torch.npu.synchronize()
    start = time.perf_counter()
    tri_out = attention(q, k, v, causal, sm_scale)
    tri_out = tri_out.half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone().cpu(), None
    tri_dk, k.grad = k.grad.clone().cpu(), None
    tri_dq, q.grad = q.grad.clone().cpu(), None
    torch.npu.synchronize()
    end = time.perf_counter()
    triton_first_run_time_cost = (end - start) / 1
    tri_out = None
    tri_dv = None
    tri_dk = None
    tri_dq = None
    torch_npu.npu.empty_cache()
    
    # run for time cost
    torch.npu.synchronize()
    start = time.perf_counter()
    tri_out = attention(q, k, v, causal, sm_scale)
    tri_out = tri_out.half()
    torch.npu.synchronize()
    forward_time = time.perf_counter()
    triton_forward_time_cost = (forward_time - start) / 1
    tri_out.backward(dout)
    torch.npu.synchronize()
    end = time.perf_counter()
    triton_backward_time_cost = (end - forward_time) / 1
    triton_time_cost = (end - start) / 1
    tri_out = tri_out.cpu()
    tri_dv, v.grad = v.grad.clone().cpu(), None
    tri_dk, k.grad = k.grad.clone().cpu(), None
    tri_dq, q.grad = q.grad.clone().cpu(), None

    Mask = torch.tril(torch.ones((S, S), dtype=torch.uint8, device=DEVICE))
    torch.npu.synchronize()
    start = time.perf_counter()
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    qk = p
    if causal:
        p[:, :, Mask == 0] = float("-inf")
    p = torch.softmax(p, dim=-1).half()
    torch_out = torch.matmul(p, v)
    torch_out.backward(dout)
    end = time.perf_counter()
    torch_time_cost = (end - start) / 1
    torch_out = torch_out.cpu()
    torch_dv, v.grad = v.grad.clone().cpu(), None
    torch_dk, k.grad = k.grad.clone().cpu(), None
    torch_dq, q.grad = q.grad.clone().cpu(), None
    
    def block_attention(Q, K, V, block_size=2 * 1024, causal=False, Mask=None):
        batch_size, num_heads, seq_len, d_k = Q.shape
        block_size = min(block_size, seq_len)
        output = torch.zeros_like(V, dtype=torch.float32)  # 最终输出
        if seq_len % block_size != 0:
            raise RuntimeError("序列长度必须能被块大小整除")
        
        # 遍历Q的每个块（按查询位置分块）
        for i in range(0, seq_len, block_size):
            q_tile = Q[:, :, i:i + block_size, :] # (B, H, BQ, d_k)，BQ为当前Q块长度
            bq_len = q_tile.shape[2]
            
            # 初始化累加器：存储当前Q块的输出结果
            out_tile = torch.zeros(batch_size, num_heads, bq_len, d_k, device=Q.device, dtype=torch.float32)
            # 初始化softmax的中间变量（用于数值稳定）
            max_score = torch.full((batch_size, num_heads, bq_len, 1), -torch.inf, device=Q.device, dtype=torch.float32)
            sum_exp = torch.zeros(batch_size, num_heads, bq_len, 1, device=Q.device, dtype=torch.float32)
            
            # 遍历K、V的每个块（按键值位置分块）
            for j in range(0, seq_len, block_size):
                k_tile = K[:, :, j:j + block_size, :]# (B, H, BK, d_k)
                v_tile = V[:, :, j:j + block_size, :] # (B, H, BK, d_k)
                bk_len = k_tile.shape[2]
                    
                # 1. 计算当前Q块与K块的注意力分数 (B, H, BQ, BK)
                scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)).to(torch.float32)
                
                # 2. 更新全局max（用于softmax数值稳定）
                current_max = scores.max(dim=-1, keepdim=True).values  # (B, H, BQ, 1)
                new_max = torch.max(max_score, current_max)
                
                # 3. 修正历史sum_exp（因max更新，需重新缩放）
                # 公式：exp(x - new_max) = exp(x - old_max) * exp(old_max - new_max)
                exp_diff = torch.exp(max_score - new_max)
                sum_exp = sum_exp * exp_diff + torch.exp(scores - new_max).sum(dim=-1, keepdim=True)
                
                # 4. 更新max_score为新的全局max
                max_score = new_max
                
                # 5. 计算当前块的注意力权重，并累加至输出
                # 权重 = exp(scores - max_score) / sum_exp（包含历史块的累积sum_exp）
                attn_weights = torch.exp(scores - max_score) / sum_exp
                out_tile += torch.matmul(attn_weights, v_tile)  # 累加V的贡献
            
            # 将当前Q块的结果写入输出
            output[:, :, i:i + block_size, :] = out_tile
        
        return output
    torch.npu.synchronize()
    start = time.perf_counter()
    torch_ba_out = block_attention(q, k, v).half()
    torch_ba_out.backward(dout)
    end = time.perf_counter()
    torch_ba_time_cost = (end - start) / 1
    torch_ba_out = torch_ba_out.cpu()
    torch_ba_dv, v.grad = v.grad.clone().cpu(), None
    torch_ba_dk, k.grad = k.grad.clone().cpu(), None
    torch_ba_dq, q.grad = q.grad.clone().cpu(), None
    
    atten_mask = 1 - Mask
    if not causal:
        atten_mask = torch.zeros_like(atten_mask)
    atten2_masks = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1))
    atten2_mask = atten2_masks.to(torch.float16).bool().npu()
    # first run avoid overhead time
    npu_op_out = torch_npu.npu_fusion_attention(
        query=q, 
        key=k, 
        value=v, 
        head_num=N, 
        input_layout="BNSD", 
        scale=sm_scale, 
        sparse_mode=2 if causal else 0, 
        next_tockens=0, 
        pre_tockens=S, 
        atten_mask=atten2_mask if causal else None 
    )[0].half()
    npu_op_out.backward(dout)
    npu_op_dv, v.grad = v.grad.clone().cpu(), None
    npu_op_dk, k.grad = k.grad.clone().cpu(), None
    npu_op_dq, q.grad = q.grad.clone().cpu(), None
    # real time cost run
    torch.npu.synchronize()
    start = time.perf_counter()
    npu_op_out = torch_npu.npu_fusion_attention(
        query=q, 
        key=k, 
        value=v, 
        head_num=N, 
        input_layout="BNSD", 
        scale=sm_scale, 
        sparse_mode=2 if causal else 0, 
        next_tockens=0, 
        pre_tockens=S, 
        atten_mask=atten2_mask if causal else None 
    )[0].half()
    npu_op_out.backward(dout)
    torch.npu.synchronize()
    end = time.perf_counter()
    npu_op_time_cost = (end - start) / 1
    npu_op_out = npu_op_out.cpu()
    npu_op_dv, v.grad = v.grad.clone().cpu(), None
    npu_op_dk, k.grad = k.grad.clone().cpu(), None
    npu_op_dq, q.grad = q.grad.clone().cpu(), None

    logger.info(f"Performance Test Triton-Attention with ({B},{N},{S},{D}), causal = {causal}, dtype = {dtype}, "
                f"torch_ref: {torch_time_cost:.6f}s, torch_ba_ref: {torch_ba_time_cost:.6f}s, triton: {triton_time_cost:.6f}s, npu_op: {npu_op_time_cost:.6f}s, "
                f"triton_first_more_time: {(triton_first_run_time_cost - triton_time_cost):.3f}s, "
                f"triton_forward_time_cost:triton_backward_time_cost : {triton_forward_time_cost:.6f}s:{triton_backward_time_cost:.6f}s, ({triton_forward_time_cost/triton_time_cost:.3f}:{triton_backward_time_cost/triton_time_cost:.3f}) "
                f"pytorch: pytorch_ba: triton: npu_op speed: {npu_op_time_cost/torch_time_cost:.3f}:{npu_op_time_cost/torch_ba_time_cost:.3f}:{npu_op_time_cost/triton_time_cost:.3f}:1, "
                )
    
    atol = 1e-2 if S * D < 1e5 else 1e-2
    rtol = 0 if S * D < 1e5 else 0
    torch.testing.assert_close(torch_out, npu_op_out, atol=atol, rtol=rtol)
    rtol = 1e-2 if S * D < 1e5 else 1e-2
    torch.testing.assert_close(torch_dv, npu_op_dv, atol=atol, rtol=rtol)
    torch.testing.assert_close(torch_dk, npu_op_dk, atol=atol, rtol=rtol)
    torch.testing.assert_close(torch_dq, npu_op_dq, atol=atol, rtol=rtol)

    if not causal:
        atol = 1e-1 if S * D < 1e5 else 1e-1
        rtol = 0 if S * D < 1e5 else 0
        torch.testing.assert_close(torch_out, torch_ba_out, atol=atol, rtol=rtol)
        rtol = 1e-0 if S * D < 1e5 else 1e-0
        torch.testing.assert_close(torch_dv, torch_ba_dv, atol=atol, rtol=rtol)
        torch.testing.assert_close(torch_dk, torch_ba_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(torch_dq, torch_ba_dq, atol=atol, rtol=rtol)
    
    atol = 1e-2 if S * D < 1e5 else 1e-2
    rtol = 0 if S * D < 1e5 else 0
    torch.testing.assert_close(npu_op_out, tri_out, atol=atol, rtol=rtol)
    rtol = 1e-2 if S * D < 1e5 else 1e-2
    torch.testing.assert_close(npu_op_dv, tri_dv, atol=atol, rtol=rtol)
    torch.testing.assert_close(npu_op_dk, tri_dk, atol=atol, rtol=rtol)
    torch.testing.assert_close(npu_op_dq, tri_dq, atol=atol, rtol=rtol)
    
    logger.info(f"Finish Test Triton-Attention with ({B},{N},{S},{D}), causal = {causal}, dtype = {dtype}, ")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  
        format="%(asctime)s %(name)s %(levelname)s %(message)s" 
    )
    test_op(B=1, N=64, S=4096, D=128, causal=True, dtype=torch.float16) # deepseek v3.2 exp
    test_op(B=1, N=64, S=4096, D=128, causal=False, dtype=torch.float16) # deepseek v3.2 exp
    exit()
    Bs = [1, 2]
    Ns = [1, 2]
    Ss = [64, 512]
    Ds = [32]
    causals = [False, True]
    for B in Bs:
        for N in Ns:
            for S in Ss:
                for D in Ds:
                    for causal in causals:
                        test_op(B=B, N=N, S=S, D=D, causal=causal, dtype=torch.float16)
