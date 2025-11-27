# need new version CANN(>8.3) & triton_npu(coming stable version) installed for triton npu support.
import math
import os
import logging
import time
import torch
import triton
import torch_npu
import triton.language as tl


class TritonIndexerFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a Triton-backed indexed attention-like kernel.
        This function validates inputs, allocates the output tensor, configures tile/block
        sizes, and launches a Triton kernel to compute a (b, s, m) float32 result from
        per-query and per-key representations and auxiliary scale factors. 
        Parameters
        ----------
        ctx : context
            Autograd context used to save tensors/metadata for backward.
        q : torch.Tensor
            Query tensor with dtype torch.float16 and contiguous memory layout.
            Expected shape: (b, s, h, d)
        q_s : torch.Tensor
            Scaling tensor with dtype torch.float32 and contiguous
            memory layout. Expected shape: (b, s, h)
        k : torch.Tensor
            Key tensor with dtype torch.float16 and contiguous memory layout.
            Expected shape: (b, m, d)
        k_s : torch.Tensor
            Scaling tensor with dtype torch.float32 and contiguous
            memory layout. Expected shape: (b, m)
        Returns
        -------
        torch.Tensor
            A newly allocated tensor of dtype torch.float32 and shape (b, s, m) placed
            on the same device as q. The tensor contains the computed outputs produced
            by the Triton kernel.
        Raises
        ------
        ValueError
            - If q or k are not torch.float16.
            - If q_s or k_s are not torch.float32.
            - If any of q, k, q_s, k_s are not contiguous.
            - If batch sizes or feature dimensions do not match between q and k.
            - If q_s or k_s have unexpected shapes.
            - If sequence length s or key length m are not divisible by the chosen
              block sizes (BLOCK_S = min(s, 8), BLOCK_M = min(m, 8)).
        Notes
        -----
        - Output memory layout and individual strides are passed explicitly to the Triton
          kernel. The grid used to launch the kernel is (b, s // BLOCK_S).
        - BLOCK_S and BLOCK_M are chosen as min(s, 8) and min(m, 8) respectively and
          must evenly divide s and m.
        - The function saves (q, q_s, k, k_s) into ctx and records ctx.dims and
          ctx.block_sizes for use during backward propagation.
        - The Triton kernel is invoked with multibuffer=False and receives decomposed
          stride values and block parameters as kernel arguments.
        """
        # 输入检查
        if q.dtype != torch.float16:
            raise ValueError(f"q must be float16, got {q.dtype}")
        if k.dtype != torch.float16:
            raise ValueError(f"k must be float16, got {k.dtype}")
        if q_s.dtype != torch.float32:
            raise ValueError(f"q_s must be float32, got {q_s.dtype}")
        if k_s.dtype != torch.float32:
            raise ValueError(f"k_s must be float32, got {k_s.dtype}")
        if not q.is_contiguous():
            raise ValueError("q must be contiguous")
        if not k.is_contiguous():
            raise ValueError("k must be contiguous")
        if not q_s.is_contiguous():
            raise ValueError("q_s must be contiguous")
        if not k_s.is_contiguous():
            raise ValueError("k_s must be contiguous")

        # 维度解析
        b, s, h, d = q.shape
        b_k, m, d_k = k.shape
        if b != b_k:
            raise ValueError(f"Batch size mismatch: q={b}, k={b_k}")
        if d != d_k:
            raise ValueError(f"Feature dim mismatch: q={d}, k={d_k}")
        if q_s.shape != (b, s, h):
            raise ValueError(f"q_s shape mismatch: expected {(b,s,h)}, got {q_s.shape}")
        if k_s.shape != (b, m):
            raise ValueError(f"k_s shape mismatch: expected {(b,m)}, got {k_s.shape}")
        
        # 输出张量初始化
        output = torch.empty((b, s, m), dtype=torch.float32, device=q.device)

        # 核函数配置 - 以B和S作为grid，对M进行切分，D不切分
        BLOCK_S = min(s, 8)
        BLOCK_M = min(m, 8)
        if s % BLOCK_S != 0:
            raise ValueError(f"s must be divisible by BLOCK_S={BLOCK_S}")
        if m % BLOCK_M != 0:
            raise ValueError(f"m must be divisible by BLOCK_M={BLOCK_M}")
        
        # grid改为(batch, s//BLOCK_S)
        grid = (b, s // BLOCK_S)

        # 启动Triton核
        _forward_kernel[grid](
            q_ptr=q, q_s_ptr=q_s, k_ptr=k, k_s_ptr=k_s,
            output_ptr=output,
            b=b, s=s, h=h, d=d, m=m,
            # 拆分stride元组为独立参数
            q_stride_b=q.stride(0), 
            q_stride_s=q.stride(1), 
            q_stride_h=q.stride(2), 
            q_stride_d=q.stride(3),
            q_s_stride_b=q_s.stride(0), 
            q_s_stride_s=q_s.stride(1), 
            q_s_stride_h=q_s.stride(2),
            k_stride_b=k.stride(0), 
            k_stride_m=k.stride(1), 
            k_stride_d=k.stride(2),
            k_s_stride_b=k_s.stride(0),
            k_s_stride_m=k_s.stride(1),
            output_stride_b=output.stride(0), 
            output_stride_s=output.stride(1), 
            output_stride_m=output.stride(2),
            # block参数 - S和M
            BLOCK_S=BLOCK_S,
            BLOCK_M=BLOCK_M,
            multibuffer=False,
        )

        # 保存反向传播所需数据
        ctx.save_for_backward(q, q_s, k, k_s)
        ctx.dims = (b, s, h, d, m)
        ctx.block_sizes = (BLOCK_S, BLOCK_M)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Compute gradients for the custom flash-attention forward pass.
        This backward implementation expects that the forward stored the following on ctx:
        - saved_tensors: (q, q_s, k, k_s)
        - dims: (b, s, h, d, m)
        - block_sizes: (BLOCK_S, BLOCK_M)
        Behavior:
        - Allocates zero-filled gradient tensors matching the corresponding saved inputs:
            grad_q, grad_q_s, grad_k, grad_k_s. grad_q and grad_k are created as float32
            to match the accumulator precision used by the kernels; grad_q_s and grad_k_s
            preserve the dtype/shape of q_s and k_s respectively.
        - Launches two Triton kernels to compute the gradients in blocked fashion:
            1) _backward_q_kernel is launched with grid (b, s // BLOCK_S) to compute
                 gradients with respect to q and q_s. The kernel is provided pointers to
                 grad_output, q, q_s, k, k_s, and the output pointers grad_q and grad_q_s,
                 plus all relevant dimension and per-dimension stride information and the
                 block sizes.
            2) _backward_k_kernel is launched with grid (b, m // BLOCK_M) to compute
                 gradients with respect to k and k_s. It receives the analogous set of
                 pointers, dims and stride arguments.
        Inputs:
        - ctx: autograd context containing saved tensors and configuration (see above).
        - grad_output (torch.Tensor): gradient of the loss w.r.t. the forward output;
            expected shape and strides are passed into the kernels.
        Outputs:
        - Returns a tuple of gradients in the order:
            (grad_q, grad_q_s, grad_k, grad_k_s).
            Each gradient has the same shape as its corresponding input tensor.
        Notes:
        - The function relies on the forward having populated ctx.saved_tensors, ctx.dims,
            and ctx.block_sizes correctly; incorrect shapes or strides will lead to kernel
            launch errors or incorrect gradients.
        - Strides for every tensor are explicitly passed to the kernels so they can
            index arbitrary memory layouts.
        - Grid partitioning uses sequence blocks (BLOCK_S) across the s dimension for q
            and mask blocks (BLOCK_M) across the m dimension for k.
        """
        q, q_s, k, k_s = ctx.saved_tensors
        b, s, h, d, m = ctx.dims
        BLOCK_S, BLOCK_M = ctx.block_sizes

        # 初始化梯度张量
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_q_s = torch.zeros_like(q_s)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_k_s = torch.zeros_like(k_s)

        # 计算q和q_s的梯度 - grid使用(b, s//BLOCK_S)
        grid_q = (b, s // BLOCK_S)
        _backward_q_kernel[grid_q](
            grad_output_ptr=grad_output,
            q_ptr=q, q_s_ptr=q_s, k_ptr=k, k_s_ptr=k_s,
            grad_q_ptr=grad_q, grad_q_s_ptr=grad_q_s,
            b=b, s=s, h=h, d=d, m=m,
            # 拆分stride参数
            q_stride_b=q.stride(0), 
            q_stride_s=q.stride(1), 
            q_stride_h=q.stride(2), 
            q_stride_d=q.stride(3),
            q_s_stride_b=q_s.stride(0), 
            q_s_stride_s=q_s.stride(1), 
            q_s_stride_h=q_s.stride(2),
            k_stride_b=k.stride(0), 
            k_stride_m=k.stride(1), 
            k_stride_d=k.stride(2),
            k_s_stride_b=k_s.stride(0),
            k_s_stride_m=k_s.stride(1),
            grad_output_stride_b=grad_output.stride(0), 
            grad_output_stride_s=grad_output.stride(1), 
            grad_output_stride_m=grad_output.stride(2),
            BLOCK_S=BLOCK_S,
            BLOCK_M=BLOCK_M,
            multibuffer=False,
        )

        # 计算k和k_s的梯度 - grid使用(b, m//BLOCK_M)
        grid_k = (b, m // BLOCK_M)
        _backward_k_kernel[grid_k](
            grad_output_ptr=grad_output,
            q_ptr=q, q_s_ptr=q_s, k_ptr=k, k_s_ptr=k_s,
            grad_k_ptr=grad_k, grad_k_s_ptr=grad_k_s,
            b=b, s=s, h=h, d=d, m=m,
            # 拆分stride参数
            q_stride_b=q.stride(0),
            q_stride_s=q.stride(1), 
            q_stride_h=q.stride(2),
            q_stride_d=q.stride(3),
            q_s_stride_b=q_s.stride(0),
            q_s_stride_s=q_s.stride(1), 
            q_s_stride_h=q_s.stride(2),
            k_stride_b=k.stride(0),
            k_stride_m=k.stride(1), 
            k_stride_d=k.stride(2),
            k_s_stride_b=k_s.stride(0),
            k_s_stride_m=k_s.stride(1),
            grad_output_stride_b=grad_output.stride(0), 
            grad_output_stride_s=grad_output.stride(1), 
            grad_output_stride_m=grad_output.stride(2),
            BLOCK_S=BLOCK_S,
            BLOCK_M=BLOCK_M,
            multibuffer=False,
        )
        return (
            grad_q,
            grad_q_s,
            grad_k,
            grad_k_s,
        )


@triton.jit
def _forward_kernel(
    q_ptr, q_s_ptr, k_ptr, k_s_ptr, output_ptr,
    b: tl.constexpr,
    s: tl.constexpr, 
    h: tl.constexpr, 
    d: tl.constexpr, 
    m: tl.constexpr, 
    # 单独的stride参数
    q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    q_s_stride_b, q_s_stride_s, q_s_stride_h,
    k_stride_b, k_stride_m, k_stride_d,
    k_s_stride_b, k_s_stride_m,
    output_stride_b, output_stride_s, output_stride_m,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # grid: b, s//BLOCK_S
    batch_idx = tl.program_id(0)
    s_block_idx = tl.program_id(1)

    # 计算当前block的S范围
    s_offset = s_block_idx * BLOCK_S
    s_idx = tl.arange(0, BLOCK_S) + s_offset

    # 计算基础偏移
    q_b = batch_idx * q_stride_b
    q_base = q_b + s_idx[:, None, None] * q_stride_s  # [BLOCK_S, 1, 1]

    q_s_b = batch_idx * q_s_stride_b
    q_s_base = q_s_b + s_idx[:, None] * q_s_stride_s  # [BLOCK_S, 1]

    k_base = batch_idx * k_stride_b  # [1]

    output_b = batch_idx * output_stride_b
    output_base = output_b + s_idx[:, None] * output_stride_s  # [BLOCK_S, 1]
    
    # 处理所有头和完整的D维度
    h_idx = tl.arange(0, h)  # [h]
    d_idx = tl.arange(0, d)  # [d] - D不切分
    
    # 加载所有头的q_s [BLOCK_S, h]
    q_s_idx = q_s_base + h_idx[None, :] * q_s_stride_h
    q_s_val = tl.load(q_s_ptr + q_s_idx).to(tl.float32) # b, [BLOCK_S, h]
    
    # 加载q [BLOCK_S, h, d]
    q_idx = q_base + h_idx[None, :, None] * q_stride_h + d_idx[None, None, :] * q_stride_d
    q_val = tl.load(q_ptr + q_idx).to(tl.float32) # b, [BLOCK_S, h, d]
    
    # 循环处理M维度（切分）
    for m_off in range(0, m, BLOCK_M):
        m_idx = m_off + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        
        # 加载key [BLOCK_M, d]
        k_idx = k_base + m_idx[:, None] * k_stride_m + d_idx[None, :] * k_stride_d
        k_val = tl.load(k_ptr + k_idx).to(tl.float32) # b, [m, d]
        
        # 将k_val转置为[BLOCK_M, d] -> [d, BLOCK_M]，再扩展维度为[h, d, BLOCK_M]（与q_val的h维度匹配）
        k_val_transposed = tl.trans(k_val, (1, 0))[None, :, :].broadcast_to((BLOCK_S, d, BLOCK_M))
        
        # 确保q_val (BLOCK_S, h, d) 与 k_val_transposed (h, d, BLOCK_M) 首维度兼容
        logits = tl.dot(q_val, k_val_transposed).to(tl.float32)  # 结果维度为 (BLOCK_S, h, BLOCK_M)
        
        # ReLU激活 + q_s缩放
        logits = tl.maximum(logits, 0.0)
        logits = logits * q_s_val[:, :, None] # (BLOCK_S, h, BLOCK_M)
        
        # 累加到头维度求和结果中 [BLOCK_S, BLOCK_M]
        logits_sum = tl.sum(logits, axis=1) # (BLOCK_S, BLOCK_M)
        
        # 加载k_s并应用缩放 [BLOCK_M]
        k_s_idx = batch_idx * k_s_stride_b + m_idx * k_s_stride_m
        k_s_val = tl.load(k_s_ptr + k_s_idx).to(tl.float32) # [BLOCK_M]
        output_val = logits_sum * k_s_val[None, :] # (BLOCK_S, BLOCK_M)
        
        # 存储结果
        output_idx = output_base + m_idx[None, :] * output_stride_m
        tl.store(output_ptr + output_idx, output_val) # (BLOCK_S, BLOCK_M)


@triton.jit
def _backward_q_kernel(
    grad_output_ptr,
    q_ptr, q_s_ptr, k_ptr, k_s_ptr,
    grad_q_ptr, grad_q_s_ptr,
    b: tl.constexpr, 
    s: tl.constexpr, 
    h: tl.constexpr, 
    d: tl.constexpr, 
    m: tl.constexpr, 
    # 单独的stride参数
    q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    q_s_stride_b, q_s_stride_s, q_s_stride_h,
    k_stride_b, k_stride_m, k_stride_d,
    k_s_stride_b, k_s_stride_m,
    grad_output_stride_b, grad_output_stride_s, grad_output_stride_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # 解析program ID
    batch_idx = tl.program_id(0)
    s_block_idx = tl.program_id(1)
    
    # 计算当前block的S范围
    s_offset = s_block_idx * BLOCK_S
    s_idx = tl.arange(0, BLOCK_S) + s_offset
    
    # 基础偏移计算
    q_b = batch_idx * q_stride_b
    q_base = q_b + s_idx[:, None, None] * q_stride_s  # [BLOCK_S, 1, 1]
    
    q_s_b = batch_idx * q_s_stride_b
    q_s_base = q_s_b + s_idx[:, None] * q_s_stride_s  # [BLOCK_S, 1]
    
    k_base = batch_idx * k_stride_b
    
    grad_out_b = batch_idx * grad_output_stride_b
    grad_out_base = grad_out_b + s_idx[:, None] * grad_output_stride_s  # [BLOCK_S, 1]
    
    # 处理所有头和完整的D维度
    h_idx = tl.arange(0, h)  # [h]
    d_idx = tl.arange(0, d)  # [d] - D不切分
    
    # 加载q和q_s
    q_idx = q_base + h_idx[None, :, None] * q_stride_h + d_idx[None, None, :] * q_stride_d
    q_val = tl.load(q_ptr + q_idx).to(tl.float32) # [BLOCK_S, h, d]
    
    q_s_idx = q_s_base + h_idx[None, :] * q_s_stride_h
    q_s_val = tl.load(q_s_ptr + q_s_idx).to(tl.float32) # [BLOCK_S, h]
    
    # 初始化梯度累加器
    grad_q_accum = tl.zeros((BLOCK_S, h, d), dtype=tl.float32)
    grad_q_s_accum = tl.zeros((BLOCK_S, h), dtype=tl.float32)
    
    # 循环处理M维度
    for m_off in range(0, m, BLOCK_M):
        m_idx = m_off + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        
        # 加载k和k_s
        k_idx = k_base + m_idx[:, None] * k_stride_m + d_idx[None, :] * k_stride_d
        k_val = tl.load(k_ptr + k_idx).to(tl.float32) # [BLOCK_M, d]
        
        k_s_idx = batch_idx * k_s_stride_b + m_idx * k_s_stride_m 
        k_s_val = tl.load(k_s_ptr + k_s_idx).to(tl.float32) # [BLOCK_M]
        
        # 加载梯度输出
        grad_out_idx = grad_out_base + m_idx[None, :] * grad_output_stride_m
        grad_out_val = tl.load(grad_output_ptr + grad_out_idx).to(tl.float32) # [BLOCK_S, BLOCK_M]
        
        # 正向计算重现        
        # 将k_val转置为[BLOCK_M, d] -> [d, BLOCK_M]，再扩展维度为[h, d, BLOCK_M]（与q_val的h维度匹配）
        k_val_transposed = tl.trans(k_val, (1, 0))[None, :, :].broadcast_to((BLOCK_S, d, BLOCK_M))
        # 确保q_val (BLOCK_S, h, d) 与 k_val_transposed (BLOCK_S, d, BLOCK_M) 首维度兼容
        logits = tl.dot(q_val, k_val_transposed)  # 结果维度为 (BLOCK_S, h, BLOCK_M)
        
        relu_mask = logits > 0.0
        logits_relu = tl.maximum(logits, 0.0)
        
        # 梯度链计算
        grad_logits = grad_out_val[:, None, :] * k_s_val[None, None, :] * q_s_val[:, :, None] # [BLOCK_S, h, BLOCK_M]
        grad_logits = grad_logits * relu_mask  # ReLU梯度 
        
        k_val_3d = k_val[None, :, :].broadcast_to((BLOCK_S, BLOCK_M, d)) 
        
        grad_logits = tl.reshape(grad_logits, (BLOCK_S, h, BLOCK_M)) 
        k_val_3d = tl.reshape(k_val_3d, (BLOCK_S, BLOCK_M, d)) 
        # 计算q的梯度 (BLOCK_S, h, BLOCK_M) x (BLOCK_S, BLOCK_M, d) = (BLOCK_S, h, d)
        grad_q = tl.dot(grad_logits, k_val_3d) # (BLOCK_S, h, d)
        
        grad_q_accum += grad_q # (BLOCK_S, h, d)
        
        # 计算q_s的梯度
        grad_q_s = tl.sum(grad_logits * logits_relu, axis=2)
        grad_q_s_accum += grad_q_s
    
    # 存储q的梯度
    tl.store(grad_q_ptr + q_idx, grad_q_accum) 
    
    # 存储q_s的梯度
    tl.store(grad_q_s_ptr + q_s_idx, grad_q_s_accum)


@triton.jit
def _backward_k_kernel(
    grad_output_ptr,
    q_ptr, q_s_ptr, k_ptr, k_s_ptr,
    grad_k_ptr, grad_k_s_ptr,
    b: tl.constexpr, 
    s: tl.constexpr, 
    h: tl.constexpr, 
    d: tl.constexpr, 
    m: tl.constexpr, 
    # 单独的stride参数
    q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    q_s_stride_b, q_s_stride_s, q_s_stride_h,
    k_stride_b, k_stride_m, k_stride_d,
    k_s_stride_b, k_s_stride_m,
    grad_output_stride_b, grad_output_stride_s, grad_output_stride_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # 解析program ID - grid为(batch, m//BLOCK_M)
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    
    # 计算当前block的M范围
    m_start = m_block_idx * BLOCK_M
    m_idx = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    
    # 基础偏移计算
    k_b = batch_idx * k_stride_b
    k_base = k_b + m_idx[:, None] * k_stride_m  # [BLOCK_M, 1]
    
    k_s_b = batch_idx * k_s_stride_b
    k_s_idx = k_s_b + m_idx * k_s_stride_m  # [BLOCK_M]
    k_s_val = tl.load(k_s_ptr + k_s_idx).to(tl.float32) # [BLOCK_M]
    
    grad_out_b = batch_idx * grad_output_stride_b
    
    # 维度索引准备
    h_vec = tl.arange(0, h)  # 头维度向量索引 [h]
    d_idx = tl.arange(0, d)  # [d]
    
    # 加载当前block的k值 [BLOCK_M, d]
    k_idx = k_base + d_idx[None, :] * k_stride_d
    k_val = tl.load(k_ptr + k_idx).to(tl.float32)
    
    # 初始化梯度累加器
    grad_k_accum = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    grad_k_s_accum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 循环处理S维度
    for s_off in range(0, s, BLOCK_S):
        s_idx = s_off + tl.arange(0, BLOCK_S)  # [BLOCK_S]
        
        # 加载q相关数据 [BLOCK_S, h, d]
        q_idx = (batch_idx * q_stride_b 
                + s_idx[:, None, None] * q_stride_s 
                + h_vec[None, :, None] * q_stride_h 
                + d_idx[None, None, :] * q_stride_d)
        q_val = tl.load(q_ptr + q_idx).to(tl.float32)
        
        # 加载q_s相关数据 [BLOCK_S, h]
        q_s_idx = (batch_idx * q_s_stride_b 
                  + s_idx[:, None] * q_s_stride_s 
                  + h_vec[None, :] * q_s_stride_h)
        q_s_val = tl.load(q_s_ptr + q_s_idx).to(tl.float32)
        
        # 加载梯度输出 [BLOCK_S, BLOCK_M]
        grad_out_idx = (grad_out_b 
                       + s_idx[:, None] * grad_output_stride_s 
                       + m_idx[None, :] * grad_output_stride_m)
        grad_out_val = tl.load(grad_output_ptr + grad_out_idx).to(tl.float32)
        
        # 1. 重现前向logits计算（严格遵循3D tl.dot规则）
        # 调整q_val为 [BLOCK_S*h, 1, d]（合并S和h维度，确保第一维度一致）
        q_3d = tl.reshape(q_val, (BLOCK_S * h, 1, d))  # 关键：合并维度使第一维度唯一
        
        # 调整k_val为 [BLOCK_S*h, d, BLOCK_M]（广播第一维度匹配q_3d）
        k_t = tl.trans(k_val, (1, 0))[None, :, :]  # [1, d, BLOCK_M]
        k_3d = k_t.broadcast_to((BLOCK_S * h, d, BLOCK_M))  # 广播第一维度
        
        # 3D dot运算：[BLOCK_S*h, 1, d] × [BLOCK_S*h, d, BLOCK_M] → [BLOCK_S*h, 1, BLOCK_M]
        logits_3d = tl.dot(q_3d, k_3d)
        
        # 恢复形状为 [BLOCK_S, h, BLOCK_M]
        logits = tl.reshape(logits_3d, (BLOCK_S, h, BLOCK_M))
        
        # ReLU激活
        relu_mask = logits > 0.0
        logits_relu = tl.maximum(logits, 0.0)
        
        # 2. 计算梯度链
        grad_logits = grad_out_val[:, None, :] * q_s_val[:, :, None] * k_s_val[None, None, :] * relu_mask
        
        # 3. 计算k的梯度
        # q_val转置为 [BLOCK_S, d, h]
        q_t = tl.trans(q_val, (0, 2, 1))  # [BLOCK_S, d, h]
        
        # 3D dot计算：[BLOCK_S, d, h] × [BLOCK_S, h, BLOCK_M] → [BLOCK_S, d, BLOCK_M]
        grad_k_inter = tl.dot(q_t, grad_logits)
        
        # 沿S维度累加 → [d, BLOCK_M] → 转置为 [BLOCK_M, d]
        grad_k_block = tl.sum(grad_k_inter, axis=0)  # [d, BLOCK_M]
        grad_k_block = tl.trans(grad_k_block, (1, 0))  # [BLOCK_M, d]
        grad_k_accum += grad_k_block
        
        # 4. 计算k_s的梯度
        # 头维度求和 → [BLOCK_S, BLOCK_M]
        logits_sum = tl.sum(logits_relu * q_s_val[:, :, None], axis=1)
        # 沿S维度累加 → [BLOCK_M]
        grad_k_s_block = tl.sum(grad_out_val * logits_sum, axis=0)
        grad_k_s_accum += grad_k_s_block
    
    # 存储k的梯度
    tl.store(grad_k_ptr + k_idx, grad_k_accum)
    
    # 存储k_s的梯度
    tl.store(grad_k_s_ptr + k_s_idx, grad_k_s_accum)

indexer_function = TritonIndexerFunction.apply


# 便捷调用接口
def triton_indexer(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
    return indexer_function(q, q_s, k, k_s)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply the normalized Hadamard transform (Fast Walsh-Hadamard Transform) to the last dimension of x.
    
    Args:
        x (torch.Tensor): Input tensor of shape [..., N], where N is a power of 2.
        scale (float): Scaling factor applied after the transform (e.g., N**-0.5 for orthonormal transform).
    
    Returns:
        torch.Tensor: Transformed tensor of the same shape as x.
    """
    dtype = x.dtype
    x = x.float()  # FWHT is numerically safer in float32
    n = x.size(-1)
    
    # Check that n is a power of two
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    # Reshape to [..., n] and make contiguous
    original_shape = x.shape
    x = x.view(-1, n)
    h = x

    # Iterative in-place FWHT (butterfly operations)
    h = h.contiguous()
    m = 1
    while m < n:
        for i in range(0, n, m * 2):
            a = h[:, i:i + m]
            b = h[:, i + m:i + 2 * m]
            h[:, i:i + m] = a + b
            h[:, i + m:i + 2 * m] = a - b
        m *= 2

    h = h.view(original_shape)
    h = h * scale
    return h.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


def test_index_bf16(B, S, H, D, M):
    DEVICE = "npu"  
    logging.info("Start case: B=%s, S=%s, H=%s, D=%s, M=%s", B, S, H, D, M)

    q = torch.empty((B, S, H, D), dtype=torch.float16, device=DEVICE).normal_(mean=0.0, std=math.sqrt(1 / ((S + M))))
    q_s = torch.ones((B, S, H), dtype=torch.float32, device=DEVICE)
    k = torch.empty((B, M, D), dtype=torch.float16, device=DEVICE).normal_(mean=0.0, std=math.sqrt(1 / ((S + M))))
    k_s = torch.ones((B, M), dtype=torch.float32, device=DEVICE)
    dout = torch.ones_like(torch.empty((B, S, M), dtype=torch.float32, device=DEVICE)) 
    
    q = rotate_activation(q)
    k = rotate_activation(k)
    q_s = q_s * (D ** -0.5)
    k_s = k_s * (D ** -0.5)
    
    q = q.requires_grad_()
    k = k.requires_grad_()
    q_s = q_s.requires_grad_()
    k_s = k_s.requires_grad_()
    
    # Pytorch实现
    torch_run_time_cost = 0
    from torch_indexer_bf16 import index_bf16
    start = time.perf_counter()
    index_score_torch = index_bf16(q, q_s, k, k_s) 
    index_score_torch.backward(dout, retain_graph=True)
    torch_dq, q.grad = q.grad.clone().cpu(), None
    torch_dqs, q_s.grad = q_s.grad.clone().cpu(), None
    torch_dk, k.grad = k.grad.clone().cpu(), None
    torch_dks, k_s.grad = k_s.grad.clone().cpu(), None
    torch.npu.synchronize()
    end = time.perf_counter()
    torch_run_time_cost = (end - start) / 1
    
    index_score_torch2 = index_bf16(q, q_s, k, k_s) 
    index_score_torch2.backward(dout, retain_graph=True)
    torch_dq, q.grad = q.grad.clone().cpu(), None
    torch_dqs, q_s.grad = q_s.grad.clone().cpu(), None
    torch_dk, k.grad = k.grad.clone().cpu(), None
    torch_dks, k_s.grad = k_s.grad.clone().cpu(), None
    
    # 调用Triton实现
    index_score_triton2 = triton_indexer(q, q_s, k, k_s) # 跳过首次编译
    index_score_triton2.backward(dout, retain_graph=True)
    triton_dq, q.grad = q.grad.clone().cpu(), None
    triton_dqs, q_s.grad = q_s.grad.clone().cpu(), None
    triton_dk, k.grad = k.grad.clone().cpu(), None
    triton_dks, k_s.grad = k_s.grad.clone().cpu(), None
    logging.info("First triton compile&run")
    torch.npu.synchronize()
    start = time.perf_counter()
    index_score_triton = triton_indexer(q, q_s, k, k_s)
    torch.npu.synchronize()
    end = time.perf_counter()
    triton_forward_run_time_cost = end - start
    start = time.perf_counter()
    index_score_triton.backward(dout, retain_graph=True)
    triton_dq, q.grad = q.grad.clone().cpu(), None
    triton_dqs, q_s.grad = q_s.grad.clone().cpu(), None
    triton_dk, k.grad = k.grad.clone().cpu(), None
    triton_dks, k_s.grad = k_s.grad.clone().cpu(), None
    torch.npu.synchronize()
    end = time.perf_counter()
    triton_backward_run_time_cost = end - start
    triton_run_time_cost = triton_forward_run_time_cost + triton_backward_run_time_cost
    
    
    logging.info("index_score.shape: %s, time_cost torch:triton: %.6fs,%.6fs(%.6fs,%.6fs),"
                " speed-up triton vs torch: %.3fx", 
                index_score_triton.shape, torch_run_time_cost, triton_run_time_cost,
                triton_forward_run_time_cost, triton_backward_run_time_cost,
                (torch_run_time_cost / triton_run_time_cost))
    atol = 1e-1
    rtol = 1e-1
    torch.testing.assert_close(index_score_torch, index_score_triton, atol=atol, rtol=rtol)
    torch.testing.assert_close(torch_dq, triton_dq, atol=atol, rtol=rtol)
    torch.testing.assert_close(torch_dk, triton_dk, atol=atol, rtol=rtol)
    
    logging.info("Finish case: B=%s, S=%s, H=%s, D=%s, M=%s", B, S, H, D, M)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s" 
    )
    test_index_bf16(B=1, S=2, H=2, D=16, M=8)
    test_index_bf16(B=1, S=128, H=2, D=16, M=256)
    test_index_bf16(B=1, S=128, H=2, D=128, M=4096)