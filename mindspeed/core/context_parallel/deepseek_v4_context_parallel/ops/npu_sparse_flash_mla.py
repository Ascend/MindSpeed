# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=too-many-lines

import torch
import torch.nn.functional as F

__all__ = [
    "npu_sparse_flash_mla_from_smla_inputs",
    "npu_sparse_flash_mla_with_indexer_loss_from_smla_inputs",
    "set_sparse_flash_mla_indexer_loss_scale",
]

_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3
_ORI_WIN_LEFT = 127
_ORI_WIN_RIGHT = 0
_TOPK_VALUE_MODE = 1


_forward_op = None
_grad_op = None
_indexer_loss_grad_op = None


class _OfficialSparseFlashMlaOps:
    def __init__(self, metadata_fn, forward_fn):
        self._metadata_fn = metadata_fn
        self._forward_fn = forward_fn

    def npu_sparse_flash_mla_metadata(self, *args):
        return self._metadata_fn(*args)

    def npu_sparse_flash_mla(self, *args):
        return self._forward_fn(*args)


class _OfficialSparseFlashMlaGradOps:
    def __init__(self, grad_fn):
        self._grad_fn = grad_fn

    def npu_sparse_flash_mla_grad(self, *args):
        return self._grad_fn(*args)


class _OfficialSparseLightningIndexerKLLossGradOps:
    def __init__(self, metadata_fn, grad_fn):
        self._metadata_fn = metadata_fn
        self._grad_fn = grad_fn

    def sparse_lightning_indexer_kl_loss_grad_metadata(self, *args, **kwargs):
        return self._metadata_fn(*args, **kwargs)

    def sparse_lightning_indexer_kl_loss_grad(self, *args, **kwargs):
        return self._grad_fn(*args, **kwargs)


def _load_forward_op():
    global _forward_op
    if _forward_op is None:
        try:
            from cann_ops_transformer.ops import sparse_flash_mla
            from cann_ops_transformer.ops import sparse_flash_mla_metadata
        except ImportError as err:
            raise RuntimeError(
                "npu_sparse_flash_mla requires the official cann_ops_transformer SparseFlashMla "
                "PyTorch extension. Install cann_ops_transformer with torch_npu/CANN support in "
                "the runtime environment."
            ) from err
        _forward_op = _OfficialSparseFlashMlaOps(sparse_flash_mla_metadata, sparse_flash_mla)
    return _forward_op


def _load_grad_op():
    global _grad_op
    if _grad_op is None:
        try:
            from cann_ops_transformer.ops import sparse_flash_mla_grad
        except ImportError as err:
            raise RuntimeError(
                "npu_sparse_flash_mla backward requires the official cann_ops_transformer "
                "SparseFlashMlaGrad PyTorch extension with torch_npu/CANN support."
            ) from err
        _grad_op = _OfficialSparseFlashMlaGradOps(sparse_flash_mla_grad)
    return _grad_op


def _load_indexer_loss_grad_op():
    global _indexer_loss_grad_op
    if _indexer_loss_grad_op is None:
        try:
            from cann_ops_transformer.ops import sparse_lightning_indexer_kl_loss_grad
            from cann_ops_transformer.ops import sparse_lightning_indexer_kl_loss_grad_metadata
        except ImportError as err:
            raise RuntimeError(
                "npu_sparse_flash_mla_with_indexer_loss requires the official "
                "cann_ops_transformer sparse_lightning_indexer_kl_loss_grad extension."
            ) from err
        _indexer_loss_grad_op = _OfficialSparseLightningIndexerKLLossGradOps(
            sparse_lightning_indexer_kl_loss_grad_metadata,
            sparse_lightning_indexer_kl_loss_grad,
        )
    return _indexer_loss_grad_op


def validate_sparse_flash_mla_inputs(
    q,
    ori_kv=None,
    cmp_kv=None,
    cmp_sparse_indices=None,
    layout_q="BSND",
    layout_kv="BSND",
):
    """Check CANN hardware constraints that MindSpeed does not assert.

    Mode/layout/C4A presence, sinks dtype, and cu_seqlens requirements belong
    to ``DeepSeekV4CPContextParallel`` / ``build_deepseek_v4_cp_smla_inputs``.
    """
    # NOTE: This function is invoked on the production hot path by
    # ``npu_sparse_flash_mla_forward``. Upstream MindSpeed wrappers already
    # guarantee layout/dim and most dtypes, so we keep only the "hard"
    # hardware invariants that are cheap to check and hard to reason about.

    _validate_supported_dtype("q", q)
    num_heads_q, head_dim = _get_q_heads_and_dim(q, layout_q)
    if head_dim != 512:
        raise ValueError("SparseFlashMla only supports head_dim=512.")
    if num_heads_q < 1 or num_heads_q > 128 or not _is_power_of_two(num_heads_q):
        raise ValueError("num_heads_q must be a power of two in [1, 128].")

    if _get_kv_heads(ori_kv, layout_kv) != 1:
        raise ValueError("SparseFlashMla only supports num_heads_kv=1.")

    if cmp_kv is not None and _get_kv_heads(cmp_kv, layout_kv) != 1:
        raise ValueError("SparseFlashMla only supports cmp_kv num_heads_kv=1.")

    # C4A: keep topk validation because production sometimes skips the
    # sparse-index python validation when identity prefix + causal indices.
    if cmp_sparse_indices is not None:
        if cmp_sparse_indices.shape[-1] not in (512, 1024):
            raise ValueError("C4A/CSA cmp_sparse_indices topk must be 512 or 1024.")


def infer_sparse_flash_mla_metadata_args(
    q,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_ratio=1,
    layout_q="BSND",
    layout_kv="BSND",
):
    num_heads_q, head_dim = _get_q_heads_and_dim(q, layout_q)
    kv_for_shape = ori_kv if ori_kv is not None else cmp_kv
    num_heads_kv = _get_kv_heads(kv_for_shape, layout_kv) if kv_for_shape is not None else 1
    batch_size = _infer_batch_size(q, layout_q, cu_seqlens_q)
    max_seqlen_q = _infer_max_seqlen(q, layout_q, cu_seqlens_q, seqused_q)
    max_seqlen_ori_kv = _infer_max_seqlen(ori_kv, layout_kv, cu_seqlens_ori_kv, seqused_ori_kv)
    max_seqlen_cmp_kv = _infer_max_seqlen(cmp_kv, layout_kv, cu_seqlens_cmp_kv, seqused_cmp_kv)
    ori_topk = 0 if ori_sparse_indices is None else int(ori_sparse_indices.shape[-1])
    cmp_topk = 0 if cmp_sparse_indices is None else int(cmp_sparse_indices.shape[-1])
    return {
        "num_heads_q": int(num_heads_q),
        "num_heads_kv": int(num_heads_kv),
        "head_dim": int(head_dim),
        "batch_size": int(batch_size),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_ori_kv": int(max_seqlen_ori_kv),
        "max_seqlen_cmp_kv": int(max_seqlen_cmp_kv),
        "ori_topk": int(ori_topk),
        "cmp_topk": int(cmp_topk),
        "cmp_ratio": int(cmp_ratio),
        "ori_mask_mode": _ORI_MASK_MODE,
        "cmp_mask_mode": _CMP_MASK_MODE,
        "ori_win_left": _ORI_WIN_LEFT,
        "ori_win_right": _ORI_WIN_RIGHT,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "has_ori_kv": ori_kv is not None,
        "has_cmp_kv": cmp_kv is not None,
    }


def npu_sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_ori_kv=0,
    max_seqlen_cmp_kv=0,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True,
):
    op = _load_forward_op()
    return op.npu_sparse_flash_mla_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        batch_size,
        max_seqlen_q,
        max_seqlen_ori_kv,
        max_seqlen_cmp_kv,
        ori_topk,
        cmp_topk,
        cmp_ratio,
        _ORI_MASK_MODE,
        _CMP_MASK_MODE,
        _ORI_WIN_LEFT,
        _ORI_WIN_RIGHT,
        layout_q,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )


def npu_sparse_flash_mla_forward(
    q,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=None,
    cmp_ratio=1,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
):
    softmax_scale = _resolve_softmax_scale(softmax_scale)
    validate_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    metadata = _ensure_metadata(
        metadata,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        cmp_ratio,
        layout_q,
        layout_kv,
    )
    op = _load_forward_op()
    return op.npu_sparse_flash_mla(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        _ORI_MASK_MODE,
        _CMP_MASK_MODE,
        _ORI_WIN_LEFT,
        _ORI_WIN_RIGHT,
        layout_q,
        layout_kv,
        _TOPK_VALUE_MODE,
        return_softmax_lse,
    )


def npu_sparse_flash_mla_grad(
    q,
    dout,
    attn_out,
    softmax_lse,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=None,
    cmp_ratio=1,
    layout_q="BSND",
    layout_kv="BSND",
):
    softmax_scale = _resolve_softmax_scale(softmax_scale)
    op = _load_grad_op()
    return op.npu_sparse_flash_mla_grad(
        q,
        dout,
        attn_out,
        softmax_lse,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        _ORI_MASK_MODE,
        _CMP_MASK_MODE,
        _ORI_WIN_LEFT,
        _ORI_WIN_RIGHT,
        layout_q,
        layout_kv,
    )


def _npu_sparse_flash_mla(
    q,
    ori_kv=None,
    cmp_kv=None,
    cmp_sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    cmp_residual_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=None,
    cmp_ratio=1,
    layout_q="TND",
    layout_kv="TND",
):
    (
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        _,
        _,
        _,
        cmp_residual_kv,
        metadata,
    ) = _compact_tnd_empty_batches(
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        None,
        None,
        None,
        cmp_residual_kv,
        metadata,
        layout_q,
        layout_kv,
    )

    return _SparseFlashMlaFunction.apply(
        q,
        ori_kv,
        cmp_kv,
        None,
        cmp_sparse_indices,
        None,
        None,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        None,
        None,
        None,
        cmp_residual_kv,
        None,
        None,
        sinks,
        metadata,
        _resolve_softmax_scale(softmax_scale),
        cmp_ratio,
        _ORI_MASK_MODE,
        _CMP_MASK_MODE,
        _ORI_WIN_LEFT,
        _ORI_WIN_RIGHT,
        layout_q,
        layout_kv,
    )


def npu_sparse_flash_mla_from_smla_inputs(
    smla_inputs,
    softmax_scale=None,
    sinks=None,
    cmp_ratio=None,
    layout_q="TND",
    layout_kv="TND",
):
    if cmp_ratio is None:
        raise ValueError("cmp_ratio is required.")
    cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv = _get_operator_cu_seqlens_from_smla_inputs(
        smla_inputs,
        layout_q,
        layout_kv,
    )
    return _npu_sparse_flash_mla(
        smla_inputs.q,
        ori_kv=smla_inputs.ori_kv,
        cmp_kv=smla_inputs.cmp_kv,
        cmp_sparse_indices=smla_inputs.cmp_sparse_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        cmp_residual_kv=smla_inputs.cmp_residual_kv,
        sinks=sinks,
        metadata=smla_inputs.metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )


def npu_sparse_flash_mla_with_indexer_loss_from_smla_inputs(
    smla_inputs,
    query_index,
    key_index,
    weights,
    softmax_scale=None,
    sinks=None,
    cmp_ratio=None,
    layout_q="TND",
    layout_kv="TND",
    loss_tracker=None,
    loss_coeff=1.0,
):
    if cmp_ratio is None:
        raise ValueError("cmp_ratio is required.")
    cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv = _get_operator_cu_seqlens_from_smla_inputs(
        smla_inputs,
        layout_q,
        layout_kv,
    )
    return _npu_sparse_flash_mla_with_indexer_loss(
        smla_inputs.q,
        ori_kv=smla_inputs.ori_kv,
        cmp_kv=smla_inputs.cmp_kv,
        cmp_sparse_indices=smla_inputs.cmp_sparse_indices,
        query_index=query_index,
        key_index=key_index,
        weights=weights,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        cmp_residual_kv=smla_inputs.cmp_residual_kv,
        sinks=sinks,
        metadata=smla_inputs.metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
        loss_tracker=loss_tracker,
        loss_coeff=loss_coeff,
    )


def _get_operator_cu_seqlens_from_smla_inputs(smla_inputs, layout_q, layout_kv):
    """Keep fixed-BSND boundaries internal; the operator infers them from tensor shapes."""
    return (
        None if layout_q == "BSND" else smla_inputs.cu_seqlens_q,
        None if layout_kv == "BSND" else smla_inputs.cu_seqlens_ori_kv,
        None if layout_kv == "BSND" else smla_inputs.cu_seqlens_cmp_kv,
    )


def _npu_sparse_flash_mla_with_indexer_loss(
    q,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    query_index,
    key_index,
    weights,
    *,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=None,
    cmp_ratio=4,
    layout_q="TND",
    layout_kv="TND",
    loss_tracker=None,
    loss_coeff=1.0,
):
    if query_index is None or key_index is None or weights is None:
        raise ValueError("SMLA with indexer loss requires query_index, key_index, and weights.")
    (
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        metadata,
    ) = _compact_tnd_empty_batches(
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        metadata,
        layout_q,
        layout_kv,
    )

    return _SparseFlashMlaWithIndexerLossFunction.apply(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        query_index,
        key_index,
        weights,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        metadata,
        _resolve_softmax_scale(softmax_scale),
        cmp_ratio,
        _ORI_MASK_MODE,
        _CMP_MASK_MODE,
        _ORI_WIN_LEFT,
        _ORI_WIN_RIGHT,
        layout_q,
        layout_kv,
        loss_tracker,
        loss_coeff,
    )


class _SparseFlashMlaWithIndexerLossFunction(torch.autograd.Function):
    indexer_grad_scale = None

    @staticmethod
    def set_loss_scale(scale):
        if _SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale is None:
            _SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale = scale
        else:
            _SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale.copy_(scale)

    @staticmethod
    def forward(
        ctx,
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        query_index,
        key_index,
        weights,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        loss_tracker,
        loss_coeff,
    ):
        metadata = _ensure_metadata(
            metadata,
            q,
            ori_kv,
            cmp_kv,
            None,
            cmp_sparse_indices,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            seqused_q,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            None,
            None,
            cmp_ratio,
            layout_q,
            layout_kv,
        )
        attn_out, softmax_lse = npu_sparse_flash_mla_forward(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            layout_kv=layout_kv,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            q,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            query_index,
            key_index,
            weights,
            _optional_tensor(cu_seqlens_q, q),
            _optional_tensor(cu_seqlens_ori_kv, q),
            _optional_tensor(cu_seqlens_cmp_kv, q),
            _optional_tensor(cmp_residual_kv, q),
            _optional_tensor(sinks, q),
            attn_out,
            softmax_lse,
        )
        ctx.has_cu_seqlens_q = cu_seqlens_q is not None
        ctx.has_cu_seqlens_ori_kv = cu_seqlens_ori_kv is not None
        ctx.has_cu_seqlens_cmp_kv = cu_seqlens_cmp_kv is not None
        ctx.has_cmp_residual_kv = cmp_residual_kv is not None
        ctx.has_sinks = sinks is not None
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        ctx.layout_kv = layout_kv
        ctx.loss_tracker = loss_tracker
        ctx.loss_coeff = float(loss_coeff)
        ctx.batch_size = _infer_batch_size(q, layout_q, cu_seqlens_q)
        ctx.max_seqlen_q = _infer_max_seqlen(q, layout_q, cu_seqlens_q, seqused_q)
        ctx.max_seqlen_cmp_kv = _infer_max_seqlen(cmp_kv, layout_kv, cu_seqlens_cmp_kv, seqused_cmp_kv)
        ctx.num_heads_q, _ = _get_q_heads_and_dim(q, layout_q)
        ctx.num_heads_kv = _get_kv_heads(cmp_kv, layout_kv)
        ctx.topk = int(cmp_sparse_indices.shape[-1])
        ctx.mark_non_differentiable(softmax_lse)
        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q = saved[0]
        ori_kv = saved[1]
        cmp_kv = saved[2]
        cmp_sparse_indices = saved[3]
        query_index = saved[4]
        key_index = saved[5]
        weights = saved[6]
        cu_seqlens_q = saved[7] if ctx.has_cu_seqlens_q else None
        cu_seqlens_ori_kv = saved[8] if ctx.has_cu_seqlens_ori_kv else None
        cu_seqlens_cmp_kv = saved[9] if ctx.has_cu_seqlens_cmp_kv else None
        cmp_residual_kv = saved[10] if ctx.has_cmp_residual_kv else None
        sinks = saved[11] if ctx.has_sinks else None
        attn_out = saved[12]
        softmax_lse = saved[13]

        dq, dori_kv, dcmp_kv, dsinks, _, cmp_softmax_l1 = npu_sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            attn_out,
            softmax_lse,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=None,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            layout_q=ctx.layout_q,
            layout_kv=ctx.layout_kv,
        )
        if not torch.is_tensor(cmp_softmax_l1) or cmp_softmax_l1.numel() == 0:
            raise RuntimeError("SparseFlashMlaGrad did not return cmp_softmax_l1 for indexer loss.")

        indexer_op = _load_indexer_loss_grad_op()
        slig_metadata = indexer_op.sparse_lightning_indexer_kl_loss_grad_metadata(
            ctx.num_heads_q,
            ctx.num_heads_kv,
            query_index.shape[-1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_cmp_kv,
            cmp_residual_k=cmp_residual_kv,
            batch_size=ctx.batch_size,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_cmp_kv,
            topk=ctx.topk,
            layout_q=ctx.layout_q,
            layout_k=ctx.layout_kv,
            mask_mode=ctx.cmp_mask_mode,
            cmp_ratio=ctx.cmp_ratio,
        )
        d_query_index, d_key_index, d_weights, indexer_softmax_out = indexer_op.sparse_lightning_indexer_kl_loss_grad(
            query_index,
            key_index,
            weights,
            cmp_sparse_indices,
            cmp_softmax_l1,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_cmp_kv,
            cmp_residual_k=cmp_residual_kv,
            metadata=slig_metadata,
            layout_q=ctx.layout_q,
            layout_k=ctx.layout_kv,
            mask_mode=ctx.cmp_mask_mode,
            cmp_ratio=ctx.cmp_ratio,
        )

        grad_scale = _SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale
        if grad_scale is None:
            _SparseFlashMlaWithIndexerLossFunction.set_loss_scale(
                torch.tensor(1.0, dtype=torch.float32, device=d_query_index.device)
            )
            grad_scale = _SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale
        num_seqs = query_index.shape[0] if ctx.layout_q == "TND" else ctx.batch_size * ctx.max_seqlen_q
        indexer_grad_factor = grad_scale * ctx.loss_coeff / num_seqs
        d_query_index = d_query_index * indexer_grad_factor
        d_key_index = d_key_index * indexer_grad_factor
        d_weights = d_weights * indexer_grad_factor

        if ctx.loss_tracker is not None:
            loss = _compute_indexer_loss(F.normalize(cmp_softmax_l1, p=1, dim=-1), indexer_softmax_out)
            ctx.loss_tracker(loss * ctx.loss_coeff)

        grads = [None] * 28
        grads[0] = dq
        grads[1] = dori_kv
        grads[2] = dcmp_kv
        grads[4] = d_query_index
        grads[5] = d_key_index
        grads[6] = d_weights
        grads[16] = dsinks if ctx.has_sinks else None
        return tuple(grads)


def set_sparse_flash_mla_indexer_loss_scale(scale):
    """Match the fused Indexer-loss gradient scale to the main training loss."""
    _SparseFlashMlaWithIndexerLossFunction.set_loss_scale(scale)


def _compute_indexer_loss(attn_softmax_out, indexer_softmax_out, eps=1e-9):
    target = attn_softmax_out
    indexer = indexer_softmax_out
    norm_target = target / (torch.sum(target, dim=-1, keepdim=True) + eps)
    log_target = torch.clamp(norm_target, min=eps).log()
    log_indexer = (indexer + eps).log()
    return ((log_target - log_indexer) * target).sum(dim=-1).mean()


class _SparseFlashMlaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
    ):
        metadata = _ensure_metadata(
            metadata,
            q,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            seqused_q,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            ori_topk_length,
            cmp_topk_length,
            cmp_ratio,
            layout_q,
            layout_kv,
        )
        attn_out, softmax_lse = npu_sparse_flash_mla_forward(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=ori_topk_length,
            cmp_topk_length=cmp_topk_length,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            layout_kv=layout_kv,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            q,
            _optional_tensor(ori_kv, q),
            _optional_tensor(cmp_kv, q),
            _optional_tensor(cmp_sparse_indices, q),
            _optional_tensor(cu_seqlens_q, q),
            _optional_tensor(cu_seqlens_ori_kv, q),
            _optional_tensor(cu_seqlens_cmp_kv, q),
            _optional_tensor(cmp_residual_kv, q),
            _optional_tensor(sinks, q),
            attn_out,
            softmax_lse,
        )
        ctx.has_ori_kv = ori_kv is not None
        ctx.has_cmp_kv = cmp_kv is not None
        ctx.has_cmp_sparse_indices = cmp_sparse_indices is not None
        ctx.has_cu_seqlens_q = cu_seqlens_q is not None
        ctx.has_cu_seqlens_ori_kv = cu_seqlens_ori_kv is not None
        ctx.has_cu_seqlens_cmp_kv = cu_seqlens_cmp_kv is not None
        ctx.has_cmp_residual_kv = cmp_residual_kv is not None
        ctx.has_sinks = sinks is not None
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        ctx.layout_kv = layout_kv
        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q = saved[0]
        ori_kv = saved[1] if ctx.has_ori_kv else None
        cmp_kv = saved[2] if ctx.has_cmp_kv else None
        cmp_sparse_indices = saved[3] if ctx.has_cmp_sparse_indices else None
        cu_seqlens_q = saved[4] if ctx.has_cu_seqlens_q else None
        cu_seqlens_ori_kv = saved[5] if ctx.has_cu_seqlens_ori_kv else None
        cu_seqlens_cmp_kv = saved[6] if ctx.has_cu_seqlens_cmp_kv else None
        cmp_residual_kv = saved[7] if ctx.has_cmp_residual_kv else None
        sinks = saved[8] if ctx.has_sinks else None
        attn_out = saved[9]
        softmax_lse = saved[10]

        dq, dori_kv, dcmp_kv, dsinks, _, _ = npu_sparse_flash_mla_grad(
            q,
            grad_output,
            attn_out,
            softmax_lse,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=None,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            layout_q=ctx.layout_q,
            layout_kv=ctx.layout_kv,
        )

        grads = [None] * 26
        grads[0] = dq
        grads[1] = dori_kv if ctx.has_ori_kv else None
        grads[2] = dcmp_kv if ctx.has_cmp_kv else None
        grads[16] = dsinks if ctx.has_sinks else None
        return tuple(grads)


def _ensure_metadata(
    metadata,
    q,
    ori_kv,
    cmp_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_topk_length,
    cmp_topk_length,
    cmp_ratio,
    layout_q,
    layout_kv,
):
    if metadata is not None:
        return metadata
    metadata_args = infer_sparse_flash_mla_metadata_args(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    return npu_sparse_flash_mla_metadata(
        metadata_args["num_heads_q"],
        metadata_args["num_heads_kv"],
        metadata_args["head_dim"],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        batch_size=metadata_args["batch_size"],
        max_seqlen_q=metadata_args["max_seqlen_q"],
        max_seqlen_ori_kv=metadata_args["max_seqlen_ori_kv"],
        max_seqlen_cmp_kv=metadata_args["max_seqlen_cmp_kv"],
        ori_topk=metadata_args["ori_topk"],
        cmp_topk=metadata_args["cmp_topk"],
        cmp_ratio=metadata_args["cmp_ratio"],
        layout_q=metadata_args["layout_q"],
        layout_kv=metadata_args["layout_kv"],
        has_ori_kv=metadata_args["has_ori_kv"],
        has_cmp_kv=metadata_args["has_cmp_kv"],
    )


def _validate_supported_dtype(name, tensor):
    supported = [torch.float16]
    if hasattr(torch, "bfloat16"):
        supported.append(torch.bfloat16)
    if tensor.dtype not in supported:
        raise ValueError(f"{name} must use float16 or bfloat16, got {tensor.dtype}.")


def _get_q_heads_and_dim(q, layout_q):
    if layout_q == "BSND":
        return q.shape[2], q.shape[3]
    return q.shape[1], q.shape[2]


def _get_kv_heads(kv, layout_kv):
    if layout_kv == "TND":
        return kv.shape[1]
    return kv.shape[2]


def _infer_batch_size(q, layout_q, cu_seqlens_q):
    if layout_q == "BSND":
        return q.shape[0]
    if cu_seqlens_q is None:
        return 0
    return max(0, int(cu_seqlens_q.numel()) - 1)


def _infer_max_seqlen(tensor, layout, cu_seqlens, seqused):
    if tensor is None:
        return 0
    if seqused is not None and seqused.numel() > 0:
        return int(seqused.max().item())
    if layout == "TND":
        return _max_from_cu_seqlens(cu_seqlens)
    if layout == "BSND":
        return tensor.shape[1]
    return 0


def _max_from_cu_seqlens(cu_seqlens):
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        return 0
    lengths = torch.diff(cu_seqlens)
    if lengths.numel() == 0:
        return 0
    return int(lengths.max().item())


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _resolve_softmax_scale(softmax_scale):
    if softmax_scale is None:
        raise ValueError("softmax_scale is required.")
    return float(softmax_scale)


def _optional_tensor(tensor, reference):
    if tensor is not None:
        return tensor
    return reference.new_empty((0,))


def _compact_tnd_empty_batches(
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    metadata,
    layout_q,
    layout_kv,
):
    values = (
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        metadata,
    )
    if layout_q != "TND" or layout_kv != "TND" or cu_seqlens_q is None or cu_seqlens_q.numel() < 2:
        return values

    q_lengths = torch.diff(cu_seqlens_q)
    if bool(torch.any(q_lengths < 0).item()):
        raise ValueError("cu_seqlens_q must be non-decreasing.")

    valid_indices = torch.nonzero(q_lengths > 0, as_tuple=False).reshape(-1)
    batch_size = q_lengths.numel()
    if valid_indices.numel() == batch_size:
        return values
    invalid_indices = torch.nonzero(q_lengths == 0, as_tuple=False).reshape(-1)

    def compact_cu_seqlens(cu_seqlens, name):
        if cu_seqlens is None:
            return None
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() != batch_size + 1:
            raise ValueError(f"{name} must have the same batch cardinality as cu_seqlens_q before compaction.")
        lengths = torch.diff(cu_seqlens)
        invalid_lengths = lengths.index_select(0, invalid_indices)
        if invalid_lengths.numel() > 0 and bool(torch.any(invalid_lengths != 0).item()):
            raise ValueError(
                f"{name} has non-empty KV for a sample whose query length is zero; "
                "the sample cannot be removed without compacting the KV tensor."
            )
        valid_lengths = lengths.index_select(0, valid_indices)
        return torch.cat(
            (
                cu_seqlens.new_zeros((1,)),
                torch.cumsum(valid_lengths, dim=0, dtype=cu_seqlens.dtype),
            ),
            dim=0,
        )

    def compact_per_batch_tensor(tensor, name):
        if tensor is None or tensor.numel() == 0:
            return tensor
        if tensor.dim() == 0:
            raise ValueError(f"{name} must have a batch dimension.")
        if tensor.shape[0] == batch_size:
            return tensor.index_select(0, valid_indices)
        if tensor.shape[0] == valid_indices.numel():
            return tensor
        raise ValueError(
            f"{name} batch dimension must match either the original or compacted batch size, "
            f"got {tensor.shape[0]}, expected {batch_size} or {valid_indices.numel()}."
        )

    return (
        compact_cu_seqlens(cu_seqlens_q, "cu_seqlens_q"),
        compact_cu_seqlens(cu_seqlens_ori_kv, "cu_seqlens_ori_kv"),
        compact_cu_seqlens(cu_seqlens_cmp_kv, "cu_seqlens_cmp_kv"),
        compact_per_batch_tensor(seqused_q, "seqused_q"),
        compact_per_batch_tensor(seqused_ori_kv, "seqused_ori_kv"),
        compact_per_batch_tensor(seqused_cmp_kv, "seqused_cmp_kv"),
        compact_per_batch_tensor(cmp_residual_kv, "cmp_residual_kv"),
        None,
    )
