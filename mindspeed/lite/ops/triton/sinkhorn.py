import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.runtime.driver as driver


def get_npu_properties():
    """Get NPU device properties and return the number of vector cores"""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit
def _hc_split_sinkhorn_kernel_part1(
    # Input/output tensor pointers
    mixes_ptr, hc_scale_ptr, hc_base_ptr,
    pre_ptr, post_ptr,
    # Dimension parameters
    batch_seq_size,
    # Constant parameters
    eps: tl.constexpr,
    feat_dim: tl.constexpr,
    # Block size (compile-time constant)
    hc_mult: tl.constexpr,
    group: tl.constexpr,
):
    """
    Triton Kernel: Core computation for HC-Split Sinkhorn (Pre/Post components)
    
    Removes all CUDA-related validations and compatible with older Triton versions 
    (without keepdim parameter support). Each thread block processes one (batch, seq) sample.
    
    Args:
        mixes_ptr: Pointer to input tensor mixes [batch_seq_size, feat_dim]
        hc_scale_ptr: Pointer to scale tensor [3]
        hc_base_ptr: Pointer to base tensor [(2+hc_mult)*hc_mult]
        pre_ptr: Pointer to output pre tensor [batch_seq_size, hc_mult]
        post_ptr: Pointer to output post tensor [batch_seq_size, hc_mult]
        batch_seq_size: Total number of (batch, seq) samples (b*s)
        hc_mult: HC dimension size (typically 4)
        sinkhorn_iters: Number of Sinkhorn iterations (not used in this kernel)
        eps: Small constant to avoid division by zero
        feat_dim: Total feature dimension (2+hc_mult)*hc_mult
        BLOCK_HC: Compile-time constant for HC block size (equal to hc_mult)
    """
    # program handles GROUP batch_seq entries
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)  
    pid_mask = pids < batch_seq_size

    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)

    ar4 = tl.arange(0, hc_mult)
    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)

    # offsets for each pid
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult
 
    # mixes_pre/post: shape (G,4)
    mixes_pre = tl.load(
        mixes_ptr + pid_feat_off + ar4[None, :],
        mask=pid_mask[:, None],
        other=0.0
    )
    mixes_post = tl.load(
        mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :],
        mask=pid_mask[:, None],
        other=0.0
    )

    # pre/post compute
    pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
    post = 2.0 * tl.sigmoid(mixes_post * scale_post + base_post[None, :])

    # store pre/post (contiguous, mask only for pid out-of-range)
    tl.store(
        pre_ptr + pid_hc_off + ar4[None, :],
        pre,
        mask=pid_mask[:, None]
    )
    tl.store(
        post_ptr + pid_hc_off + ar4[None, :],
        post,
        mask=pid_mask[:, None]
    )


@triton.jit
def _hc_split_sinkhorn_kernel_part2(
    # Input/output tensor pointers
    mixes_ptr, hc_scale_ptr, hc_base_ptr,
    comb_ptr,
    # Dimension parameters
    batch_seq_size, 
    hc_mult: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    # Constant parameters
    eps: tl.constexpr,
    group: tl.constexpr,
    BLOCK_ALIGN: tl.constexpr
):
    """
    Triton Kernel: Core computation for HC-Split Sinkhorn (Comb component)
    
    Implements Comb tensor calculation with Sinkhorn normalization iterations.
    Removes all CUDA-related validations and compatible with older Triton versions 
    (without keepdim parameter support). Each thread block processes one (batch, seq) sample.
    
    Args:
        mixes_ptr: Pointer to padded mixes tensor [batch_seq_size, BLOCK_HC*BLOCK_ALIGN]
        hc_scale_ptr: Pointer to scale tensor [3]
        hc_base_ptr: Pointer to padded base tensor [BLOCK_HC*BLOCK_ALIGN]
        comb_ptr: Pointer to output comb tensor [batch_seq_size, BLOCK_HC*BLOCK_ALIGN]
        batch_seq_size: Total number of (batch, seq) samples (b*s)
        hc_mult: HC dimension size (typically 4)
        sinkhorn_iters: Number of Sinkhorn normalization iterations
        eps: Small constant to avoid division by zero
        BLOCK_HC: Compile-time constant for HC block size (equal to hc_mult)
        BLOCK_ALIGN: Compile-time constant for alignment (typically 8)
    """
    lin = tl.arange(0, hc_mult * BLOCK_ALIGN)
    # program handles GROUP batch_seq entries
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size
    # ---------------- scales (3) ----------------
    scale_comb = tl.load(hc_scale_ptr + 2)
    # ---------------- base (load once per program) ----------------
    # comb base: 4x8 padded (mask columns)
    r = tl.arange(0, hc_mult)[:, None]
    c = tl.arange(0, BLOCK_ALIGN)[None, :]
    col_mask = c < hc_mult
    arange_val = tl.arange(0, hc_mult * BLOCK_ALIGN)

    base_comb = tl.load(hc_base_ptr + arange_val)
    # ---------------- mixes load (GROUP of pids) ----------------
    # offsets for each pid
    pid_feat_off = pids[:, None] * (hc_mult * BLOCK_ALIGN)
    pid_comb_off = pids[:, None] * (hc_mult * BLOCK_ALIGN)

    mixes_comb = tl.load(
        mixes_ptr + pid_feat_off[:, :, None] + arange_val[None, :, :],
        mask=pid_mask[:, None, None]
    )

    # comb logits
    comb = mixes_comb * scale_comb + base_comb[None, :, :]

    comb = comb.reshape(group, hc_mult, BLOCK_ALIGN)

    # ---------------- row softmax (stable, only valid cols) ----------------
    very_neg = -1.0e20
    comb_for_max = tl.where(col_mask[None, :, :], comb, very_neg)
    row_max = tl.max(comb_for_max, axis=2)
    comb = tl.exp(comb - row_max[:, :, None])

    # padding cols to 0 (once)
    comb = tl.where(col_mask[None, :, :], comb, 0.0)

    # ---------------- Sinkhorn iters (no mask/where inside loop) ----------------
    # relies on padding cols staying 0: 0/(x+eps)=0
    for _ in range(sinkhorn_iters):
        row_sum = tl.sum(comb, axis=2)
        comb = comb / (row_sum[:, :, None] + eps)

        col_sum = tl.sum(comb, axis=1)
        comb = comb / (col_sum[:, None, :] + eps)

    # ---------------- store comb (G,32) contiguous ----------------
    comb_flat = tl.reshape(comb, (group, hc_mult * BLOCK_ALIGN))
    tl.store(
        comb_ptr + pid_comb_off + lin[None, :],
        comb_flat,
        mask=pid_mask[:, None]
    )

 
def hc_split_sinkhorn_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton implementation of HC-Split Sinkhorn, fully aligned with PyTorch precision benchmark

    Args:
        mixes: Input tensor with shape [batch_size, seq_len, (2+hc_mult)*hc_mult]
        hc_scale: Scale tensor with shape [3] (pre/post/comb scales)
        hc_base: Base tensor with shape [(2+hc_mult)*hc_mult] (pre/post/comb bases)
        hc_mult: HC dimension size (only 4 supported in current implementation), default=4
        sinkhorn_iters: Number of Sinkhorn normalization iterations, default=20
        eps: Small constant to prevent division by zero, default=1e-6
    
    Returns:
        tuple: (pre, post, comb)
            - pre: Output tensor with shape [batch_size, seq_len, hc_mult]
            - post: Output tensor with shape [batch_size, seq_len, hc_mult]
            - comb: Output tensor with shape [batch_size, seq_len, hc_mult, hc_mult]
    """
    # Save original dtype and convert to float32 for stable computation
    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)

    # 1. Dimension calculation and flattening (align with PyTorch benchmark)
    b, s, _ = mixes.shape
    feat_dim = (2 + hc_mult) * hc_mult
    batch_seq_size = b * s  # Merge batch and sequence dimensions

    # 2. Flatten input tensor ([b, s, feat_dim] → [b*s, feat_dim])
    mixes_flat = mixes.view(-1, feat_dim).contiguous()

    # 3. Create output tensors (use input device/dtype, no CUDA enforcement)
    pre_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)

    # 4. Configure Triton kernel parameters
    BLOCK_ALIGN = 8
    group = 16

    grid = (triton.cdiv(batch_seq_size, group),)
    # 5. Launch Triton kernel for Pre/Post computation
    _hc_split_sinkhorn_kernel_part1[grid](
        # Input/output pointers
        mixes_flat, hc_scale, hc_base,
        pre_flat, post_flat,
        # Dimension parameters
        batch_seq_size,
        # Constant parameters
        eps, feat_dim, hc_mult,
        group
    )

    # Process Comb component: extract and pad to BLOCK_ALIGN
    mixes_flat_slice = mixes_flat[:, 2 * hc_mult:]
    mixes_flat_slice = mixes_flat_slice.view(-1, hc_mult, hc_mult)
    
    pad_num = BLOCK_ALIGN - hc_mult
    mixes_flat_padded = F.pad(mixes_flat_slice, pad=(0, pad_num), mode="constant", value=0.0)
    
    # Pad hc_base for comb component
    hc_base_slice = hc_base[2 * hc_mult:].view(hc_mult, hc_mult)
    hc_base_padded = F.pad(hc_base_slice, pad=(0, pad_num), mode="constant", value=0.0)

    # Create padded comb output tensor
    comb_flat_padded = torch.empty((batch_seq_size, hc_mult * BLOCK_ALIGN), dtype=mixes.dtype, device=mixes.device)
    
    # 5. Launch Triton kernel for Comb computation
    _hc_split_sinkhorn_kernel_part2[grid](
        # Input/output pointers
        mixes_flat_padded, hc_scale, hc_base_padded,
        comb_flat_padded,
        # Dimension parameters
        batch_seq_size, hc_mult, sinkhorn_iters,
        eps, group,
        # Block size
        BLOCK_ALIGN=BLOCK_ALIGN,
    )

    # 6. Reshape output tensors (inverse of flattening)
    pre = pre_flat.view(b, s, hc_mult)
    post = post_flat.view(b, s, hc_mult)
    comb = comb_flat_padded.view(b, s, hc_mult, BLOCK_ALIGN)
    # Truncate padding to original hc_mult dimension
    comb = comb[:, :, :, :hc_mult]

    # Restore original dtype
    pre = pre.to(dtype=origin_dtype)
    post = post.to(dtype=origin_dtype)
    comb = comb.to(dtype=origin_dtype)

    return pre, post, comb


@triton.jit
def hc_split_sinkhorn_backward_kernel_part1(
    # Input gradient pointers
    grad_pre_ptr,
    grad_post_ptr,
    # Forward input pointers
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    # Output gradient pointers
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    hc_mult: tl.constexpr = 4,
):
    """
    Triton Kernel: Compute gradients for Pre/Post components of HC-Split Sinkhorn

    Each thread block processes one (batch_seq) sample.
    Calculates gradients for sigmoid-transformed Pre/Post tensors and updates
    gradients for mixes, hc_scale, and hc_base.
    """
    # Single thread block processes one (b*s) sample
    idx = tl.program_id(0)
    total_dim = (2 + hc_mult) * hc_mult

    # ------------------- 1. Gradient calculation for Pre component -------------------
    # Load pre slice: idx*total_dim + [0, hc_mult)
    pre_slice = tl.load(mixes_ptr + idx * total_dim + tl.arange(0, hc_mult))
    scale_pre = tl.load(hc_scale_ptr + 0)
    base_pre = tl.load(hc_base_ptr + tl.arange(0, hc_mult))

    # Forward calculation (reconstruct)
    pre_input = pre_slice * scale_pre + base_pre
    sigmoid_pre = tl.sigmoid(pre_input)

    # Load gradient and compute sigmoid derivative
    grad_pre = tl.load(grad_pre_ptr + idx * hc_mult + tl.arange(0, hc_mult))
    sigmoid_deriv = sigmoid_pre * (1.0 - sigmoid_pre)
    grad_pre_input = grad_pre * sigmoid_deriv

    # Update gradients for pre slice [0, hc_mult)
    tl.store(
        grad_mixes_ptr + idx * total_dim + tl.arange(0, hc_mult),
        grad_pre_input * scale_pre,
    )
    tl.atomic_add(grad_hc_scale_ptr + 0, tl.sum(grad_pre_input * pre_slice))
    tl.atomic_add(grad_hc_base_ptr + tl.arange(0, hc_mult), grad_pre_input)

    # ------------------- 2. Gradient calculation for Post component -------------------
    # Load post slice: idx*total_dim + [hc_mult, 2*hc_mult)
    post_slice = tl.load(mixes_ptr + idx * total_dim + hc_mult + tl.arange(0, hc_mult))
    scale_post = tl.load(hc_scale_ptr + 1)
    base_post = tl.load(hc_base_ptr + hc_mult + tl.arange(0, hc_mult))

    # Forward calculation (reconstruct)
    post_input = post_slice * scale_post + base_post
    sigmoid_post = tl.sigmoid(post_input)

    # Load gradient and compute sigmoid derivative
    grad_post = tl.load(grad_post_ptr + idx * hc_mult + tl.arange(0, hc_mult))
    sigmoid_deriv_post = sigmoid_post * (1.0 - sigmoid_post)
    grad_post_input = grad_post * 2.0 * sigmoid_deriv_post

    # Update gradients for post slice [hc_mult, 2*hc_mult)
    tl.store(
        grad_mixes_ptr + idx * total_dim + hc_mult + tl.arange(0, hc_mult),
        grad_post_input * scale_post,
    )
    tl.atomic_add(grad_hc_scale_ptr + 1, tl.sum(grad_post_input * post_slice))
    tl.atomic_add(grad_hc_base_ptr + hc_mult + tl.arange(0, hc_mult), grad_post_input)


@triton.jit
def hc_split_sinkhorn_backward_kernel_part2(
    # Input gradient pointer
    grad_comb_ptr,
    # Forward input pointers
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    # Output gradient pointers
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    # Constant parameters (compile-time)
    batch_seq_size: tl.constexpr,
    hc_mult: tl.constexpr = 4,
    sinkhorn_iters: tl.constexpr = 20,
    eps: tl.constexpr = 1e-6,
    BLOCK_ALIGN: tl.constexpr = 8,
):
    """
    Triton Kernel: Compute gradients for Comb component of HC-Split Sinkhorn

    Each thread block processes one (batch_seq) sample.
    Reconstructs forward Sinkhorn iterations and backpropagates gradients
    through the normalization process.
    """
    # Single thread block processes one (b*s) sample
    idx = tl.program_id(0)
    if idx >= batch_seq_size:
        return

    # Define constants
    EPS = eps
    feat_dim = hc_mult * BLOCK_ALIGN

    # ------------------- Gradient calculation for Comb component -------------------
    # Load comb slice and parameters
    comb_slice_flat = tl.load(
        mixes_ptr + idx * feat_dim + tl.arange(0, hc_mult * BLOCK_ALIGN)
    )
    comb_slice = comb_slice_flat.reshape(hc_mult, BLOCK_ALIGN)

    scale_comb = tl.load(hc_scale_ptr + 2)
    base_comb_flat = tl.load(hc_base_ptr + tl.arange(0, hc_mult * BLOCK_ALIGN))
    base_comb = base_comb_flat.reshape(hc_mult, BLOCK_ALIGN)

    # Reconstruct forward computation (without keepdim)
    comb_init = comb_slice * scale_comb + base_comb

    # Subtract row max for numerical stability (simulate keepdim=True)
    row_max_compressed = tl.max(comb_init, axis=1)
    row_max = row_max_compressed.reshape(hc_mult, 1)
    exp_comb = tl.exp(comb_init - row_max)

    # Apply column mask (truncate beyond hc_mult)
    mask = tl.arange(0, hc_mult * BLOCK_ALIGN) % BLOCK_ALIGN < hc_mult
    comb = tl.reshape(exp_comb, (hc_mult * BLOCK_ALIGN))
    comb = tl.where(mask, comb, 0.0)
    exp_comb = tl.reshape(comb, (hc_mult, BLOCK_ALIGN))

    # Initialize K and save row/column sums for backprop
    K = exp_comb
    row_sum_list = tl.full((sinkhorn_iters, hc_mult, 1), 0.0, dtype=tl.float32)
    col_sum_list = tl.full((sinkhorn_iters, 1, BLOCK_ALIGN), 0.0, dtype=tl.float32)

    # Reconstruct forward Sinkhorn iterations
    for i in range(sinkhorn_iters):
        # Row normalization
        row_sum_compressed = tl.sum(K, axis=1)
        row_sum = row_sum_compressed.reshape(hc_mult, 1)
        K_row = K / (row_sum + EPS)

        # Column normalization
        col_sum_compressed = tl.sum(K_row, axis=0)
        col_sum = col_sum_compressed.reshape(1, BLOCK_ALIGN)
        K_col = K_row / (col_sum + EPS)

        # Save row/column sums
        row_sum_list = tl.insert_slice(
            ful=row_sum_list,
            sub=row_sum[None, :, :],
            offsets=[i, 0, 0],
            sizes=[1, hc_mult, 1],
            strides=[1, 1, 1],
        )
        col_sum_list = tl.insert_slice(
            ful=col_sum_list,
            sub=col_sum[None, :, :],
            offsets=[i, 0, 0],
            sizes=[1, 1, BLOCK_ALIGN],
            strides=[1, 1, 1],
        )

        K = K_col

    # 3.3 Backpropagate through Sinkhorn iterations
    grad_comb_flat = tl.load(
        grad_comb_ptr
        + idx * hc_mult * BLOCK_ALIGN
        + tl.arange(0, hc_mult * BLOCK_ALIGN)
    )
    dK = grad_comb_flat.reshape(hc_mult, BLOCK_ALIGN)

    # Reverse iteration for backprop
    for j in range(sinkhorn_iters):
        i = sinkhorn_iters - j - 1

        # Extract saved row/column sums
        row_sum = tl.extract_slice(
            row_sum_list,
            [i, 0, 0],
            [1, hc_mult, 1],
            [1, 1, 1],
        )
        col_sum = tl.extract_slice(
            col_sum_list,
            [i, 0, 0],
            [1, 1, BLOCK_ALIGN],
            [1, 1, 1],
        )

        row_sum = row_sum.reshape(hc_mult, 1)
        col_sum = col_sum.reshape(1, BLOCK_ALIGN)

        # Step 1: Reconstruct K_col
        K_col = K * (col_sum + EPS)
        # Step 2: Backprop column normalization
        grad_direct = dK / (col_sum + EPS)
        d_col_sum_compressed = -tl.sum(
            dK * K_col / ((col_sum + EPS) * (col_sum + EPS)), axis=-2
        )
        d_col_sum = d_col_sum_compressed.reshape(1, BLOCK_ALIGN)
        dK_row = grad_direct + d_col_sum

        # Step 3: Reconstruct K_row
        K_row = K_col * (row_sum + EPS)

        # Step 4: Update K to K_row
        K = K_row

        # Step 5: Backprop row normalization
        grad_direct_row = dK_row / (row_sum + EPS)
        d_row_sum_compressed = -tl.sum(
            dK_row * K_row / ((row_sum + EPS) * (row_sum + EPS)), axis=-1
        )
        d_row_sum = d_row_sum_compressed.reshape(hc_mult, 1)
        dK = grad_direct_row + d_row_sum

        # Apply mask to gradient
        tmp = tl.reshape(dK, (hc_mult * BLOCK_ALIGN))
        tmp = tl.where(mask, tmp, 0.0)
        dK = tl.reshape(tmp, (hc_mult, BLOCK_ALIGN))

    # Backprop through exp and row max subtraction
    d_exp_comb = dK
    d_comb_before_exp = d_exp_comb * exp_comb

    # Handle gradient of row max subtraction
    max_mask = tl.where(comb_init == row_max, 1.0, 0.0)
    max_count_compressed = tl.sum(max_mask, axis=-1)
    max_count = max_count_compressed.reshape(hc_mult, 1) + EPS

    row_sum_d_before_exp_compressed = tl.sum(d_comb_before_exp, axis=-1)
    row_sum_d_before_exp = row_sum_d_before_exp_compressed.reshape(hc_mult, 1)

    d_comb_init = d_comb_before_exp - (row_sum_d_before_exp * max_mask / max_count)

    # Backprop through linear transformation
    grad_comb_slice_flat = d_comb_init * scale_comb

    # Update gradients
    tl.store(
        grad_mixes_ptr + idx * feat_dim + tl.arange(0, hc_mult * BLOCK_ALIGN),
        grad_comb_slice_flat.reshape(hc_mult * BLOCK_ALIGN),
    )
    tl.atomic_add(grad_hc_scale_ptr + 2, tl.sum(d_comb_init * comb_slice))
    tl.atomic_add(
        grad_hc_base_ptr + tl.arange(0, hc_mult * BLOCK_ALIGN),
        d_comb_init.reshape(hc_mult * BLOCK_ALIGN),
    )


def hc_split_sinkhorn_triton_backward(
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for HC-Split Sinkhorn (Triton implementation)

    Computes gradients for input tensors (mixes, hc_scale, hc_base) using
    Triton kernels optimized for NPU execution.

    Args:
        grad_pre: Gradient of loss w.r.t. pre output, shape [b, s, hc_mult]
        grad_post: Gradient of loss w.r.t. post output, shape [b, s, hc_mult]
        grad_comb: Gradient of loss w.r.t. comb output, shape [b, s, hc_mult, hc_mult]
        mixes: Input tensor from forward pass, shape [b, s, (2+hc_mult)*hc_mult]
        hc_scale: Scale tensor from forward pass, shape [3]
        hc_base: Base tensor from forward pass, shape [(2+hc_mult)*hc_mult]
        hc_mult: HC dimension size (only 4 supported), default=4
        sinkhorn_iters: Number of Sinkhorn iterations, default=20
        eps: Small constant to avoid division by zero, default=1e-6

    Returns:
        tuple: (grad_mixes, grad_hc_scale, grad_hc_base)
            - grad_mixes: Gradient w.r.t. mixes, shape [b, s, (2+hc_mult)*hc_mult]
            - grad_hc_scale: Gradient w.r.t. hc_scale, shape [3]
            - grad_hc_base: Gradient w.r.t. hc_base, shape [(2+hc_mult)*hc_mult]
    """
    # Input validation
    b, s, total_dim = mixes.shape
    batch_seq_size = b * s

    # Save original dtype and convert to float32 for stable computation
    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)
    grad_pre = grad_pre.to(dtype=torch.float32)
    grad_post = grad_post.to(dtype=torch.float32)
    grad_comb = grad_comb.to(dtype=torch.float32)

    # Initialize output gradients (zero-initialized)
    grad_mixes = torch.zeros_like(mixes, device=mixes.device)
    grad_hc_scale = torch.zeros_like(hc_scale, device=hc_scale.device)
    grad_hc_base = torch.zeros_like(hc_base, device=hc_base.device)

    # Flatten input gradients (adapt to Triton kernel input shape)
    grad_pre_flat = grad_pre.reshape(-1, hc_mult)
    grad_post_flat = grad_post.reshape(-1, hc_mult)

    # Configure Triton grid (one thread block per batch_seq sample)
    grid = (batch_seq_size,)

    # Launch Part1 kernel (Pre/Post gradients)
    hc_split_sinkhorn_backward_kernel_part1[grid](
        # Input gradients
        grad_pre_flat,
        grad_post_flat,
        # Forward inputs
        mixes,
        hc_scale,
        hc_base,
        # Output gradients
        grad_mixes,
        grad_hc_scale,
        grad_hc_base,
        hc_mult=hc_mult,
    )

    # Process Comb component (padding for BLOCK_ALIGN)
    mixes = mixes.view(-1, (2 + hc_mult) * hc_mult)
    mixes_slice = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)
    hc_base = hc_base.view((2 + hc_mult) * hc_mult)
    hc_base_slice = hc_base[2 * hc_mult:].view(hc_mult, hc_mult)

    # Apply padding to match BLOCK_ALIGN (8)
    BLOCK_ALIGN = 8
    mixes_pad = F.pad(mixes_slice, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0)
    hc_base_pad = F.pad(
        hc_base_slice, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0
    )

    # Initialize padded gradient tensors
    grad_mixes_pad = torch.zeros(
        (batch_seq_size, hc_mult, BLOCK_ALIGN),
        dtype=grad_mixes.dtype,
        device=grad_mixes.device,
    )
    grad_hc_base_pad = torch.zeros(
        (hc_mult, BLOCK_ALIGN), dtype=grad_hc_base.dtype, device=grad_hc_base.device
    )

    # Pad comb gradient
    grad_comb_flat = grad_comb.reshape(-1, hc_mult, hc_mult)
    grad_comb_flat_pad = F.pad(
        grad_comb_flat, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0
    )

    # Launch Part2 kernel (Comb gradients)
    hc_split_sinkhorn_backward_kernel_part2[grid](
        # Input gradient
        grad_comb_flat_pad,
        # Forward inputs
        mixes_pad,
        hc_scale,
        hc_base_pad,
        # Output gradients
        grad_mixes_pad,
        grad_hc_scale,
        grad_hc_base_pad,
        # Compile-time constants
        batch_seq_size=batch_seq_size,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        BLOCK_ALIGN=BLOCK_ALIGN,
    )

    # Merge padded gradients back to original shape
    grad_mixes_slice = grad_mixes_pad[:, :, :hc_mult].reshape(b, s, hc_mult * hc_mult)
    grad_hc_base_slice = grad_hc_base_pad[:, :hc_mult].reshape(hc_mult * hc_mult)

    # Update final gradients
    grad_mixes[:, :, 2 * hc_mult:] = grad_mixes_slice
    grad_hc_base[2 * hc_mult:] = grad_hc_base_slice

    # Restore original dtype
    grad_mixes = grad_mixes.to(dtype=origin_dtype)
    grad_hc_scale = grad_hc_scale.to(dtype=origin_dtype)
    grad_hc_base = grad_hc_base.to(dtype=origin_dtype)

    return grad_mixes, grad_hc_scale, grad_hc_base
