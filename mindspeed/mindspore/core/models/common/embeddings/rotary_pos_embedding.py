# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.utils import deprecate_inference_params


_ROTATION_MATRIX = None


def get_rotation_matrix(x):
    global _ROTATION_MATRIX
    if _ROTATION_MATRIX is None:
        import numpy as np
        dim = x.shape[-1]
        index1 = np.ones(dim)
        index1[::2] = 0
        index2 = np.zeros(dim)
        index2[::2] = -1
        rotation_matrix = np.eye(dim, k=1) * index1 + np.eye(dim, k=-1) * index2
        _ROTATION_MATRIX = (
            torch.from_numpy(rotation_matrix[None, None, :, :]).to(x.dtype).to(x.device)
        )
    return _ROTATION_MATRIX


def get_rotary_seq_len(
    self,
    inference_context: BaseInferenceContext,
    transformer: TransformerBlock,
    transformer_input: Tensor,
    transformer_config: TransformerConfig,
    packed_seq_params: PackedSeqParams,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
) -> float:
    """Function to get the rotary sequence length.

    Args:
        inference_context : Used during Inference time
        transformer (TransformerBlock): The transformer block (decoder/encoder) used
            by the model
        transformer_input (Tensor): Input tensor to the transformer
        transformer_config (TransformerConfig): Transformer config used by the model
        packed_seq_params (PackedSeqParams): Packed sequence params

    Returns:
        float: The rotary sequence length
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    if packed_seq_params is not None:
        # max_seqlen are the max sequence length in the packed sequence before being divived
        # by the tp and cp size.
        return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
    elif inference_context is not None:
        rotary_seq_len = inference_context.max_sequence_length
    else:
        if transformer is not None and transformer.input_tensor is not None and len(transformer.input_tensor.shape) > 1:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        if transformer_config.sequence_parallel:
            rotary_seq_len *= transformer_config.tensor_model_parallel_size

    rotary_seq_len *= transformer_config.context_parallel_size

    return rotary_seq_len


def local_rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return torch.matmul(x, get_rotation_matrix(x))