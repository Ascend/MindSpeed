# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core.transformer import MegatronModule, ModuleSpec, build_module
from megatron.training import get_args

from mindspeed.core.fusions.fused_rms_norm import RMSNorm

from .deepseek_utils import (
    apply_rotary_emb,
    apply_rotary_emb_tnd,
    rotate_activation,
)
from .linear import LinearNoTP


@dataclass
class CompressorSubmodules:
    wkv: Union[ModuleSpec, type] = None
    wgate: Union[ModuleSpec, type] = None


def get_compressor_spec():
    return ModuleSpec(module=Compressor, submodules=CompressorSubmodules(wkv=LinearNoTP, wgate=LinearNoTP))


class Compressor(MegatronModule):
    """DeepSeek-V4 compressed KV projection and reduction."""

    def __init__(
        self,
        submodules: CompressorSubmodules,
        config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
    ):
        super().__init__(config)
        args = get_args()
        self.dim = args.hidden_size
        self.head_dim = int(head_dim)
        self.rope_head_dim = getattr(args, "qk_pos_emb_head_dim", getattr(args, "rope_head_dim", None))
        if self.rope_head_dim is None:
            raise AttributeError("DeepSeek-V4 compressor requires qk_pos_emb_head_dim.")
        self.compress_ratio = int(compress_ratio)
        self.overlap = self.compress_ratio == 4
        self.rotate = bool(rotate)
        coefficient = 2 if self.overlap else 1
        self.ape = torch.nn.Parameter(
            torch.empty(self.compress_ratio, coefficient * self.head_dim, dtype=torch.float32)
        )
        self.config.init_method(self.ape)
        setattr(self.ape, "sequence_parallel", config.sequence_parallel)

        linear_config = deepcopy(config)
        linear_config.bias = False
        self.wkv = build_module(
            submodules.wkv,
            self.dim,
            coefficient * self.head_dim,
            config=linear_config,
            bias=False,
        )
        self.wgate = build_module(
            submodules.wgate,
            self.dim,
            coefficient * self.head_dim,
            config=linear_config,
            bias=False,
        )
        norm_epsilon = getattr(args, "norm_epsilon", getattr(args, "norm_eps", None))
        if norm_epsilon is None:
            raise AttributeError("DeepSeek-V4 compressor requires norm_epsilon.")
        self.norm = RMSNorm(
            self.head_dim,
            norm_epsilon,
            sequence_parallel=config.sequence_parallel,
            config=config,
        )

    def project_candidate_blocks(self, candidate_blocks: torch.Tensor):
        kv = self.wkv(candidate_blocks)
        score = self.wgate(candidate_blocks)
        ape_shape = (1, self.compress_ratio) + (1,) * (score.dim() - 3) + (score.shape[-1],)
        return kv, score + self.ape.reshape(ape_shape)

    @staticmethod
    def _reduce_candidate_blocks(kv, score, reduction_dim, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=score.device, dtype=torch.bool)
            valid_view = valid_mask.reshape((-1,) + (1,) * (score.dim() - 1))
            score = torch.where(valid_view, score, torch.zeros_like(score))
            probability = score.softmax(dim=reduction_dim) * valid_view.to(score.dtype)
        else:
            probability = score.softmax(dim=reduction_dim)
        result = (kv * probability).sum(dim=reduction_dim)
        if valid_mask is not None:
            result = result * valid_mask.reshape((-1,) + (1,) * (result.dim() - 1)).to(result.dtype)
        return result

    def _postprocess_compressed_kv(self, kv, output_dtype, freqs_cis, *, tnd, valid_mask=None, valid_mask_dim=0):
        kv = self.norm(kv.to(output_dtype))
        valid_shape = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=kv.device, dtype=torch.bool)
            valid_shape = [1] * kv.dim()
            valid_shape[int(valid_mask_dim) % kv.dim()] = -1
            kv = kv * valid_mask.reshape(valid_shape).to(kv.dtype)
        rotary_fn = apply_rotary_emb_tnd if tnd else apply_rotary_emb
        kv[..., -self.rope_head_dim :] = rotary_fn(kv[..., -self.rope_head_dim :], freqs_cis)
        if valid_shape is not None:
            kv = kv * valid_mask.reshape(valid_shape).to(kv.dtype)
        if self.rotate:
            kv = rotate_activation(kv)
            if valid_shape is not None:
                kv = kv * valid_mask.reshape(valid_shape).to(kv.dtype)
        return kv

    def compress_candidate_blocks(
        self,
        projected_kv,
        projected_score,
        freqs_cis,
        output_dtype,
        *,
        valid_mask=None,
        batch_shared_sequence=False,
    ):
        kv = self._reduce_candidate_blocks(projected_kv, projected_score, 1, valid_mask=valid_mask)
        if batch_shared_sequence:
            if kv.dim() == 3:
                kv = kv.unsqueeze(2)
            if kv.dim() != 4 or kv.shape[2] != 1:
                raise ValueError("BSND compressor must return [candidate_count, batch_size, 1, head_dim].")
            kv = kv.transpose(0, 1).contiguous()
            kv = self._postprocess_compressed_kv(
                kv, output_dtype, freqs_cis, tnd=False, valid_mask=valid_mask, valid_mask_dim=1
            )
            return kv.transpose(0, 1).contiguous()
        if kv.dim() == 2:
            kv = kv.unsqueeze(1)
        if kv.dim() != 3 or kv.shape[1] != 1:
            raise ValueError("TND compressor must return [candidate_count, 1, head_dim].")
        return self._postprocess_compressed_kv(kv, output_dtype, freqs_cis, tnd=True, valid_mask=valid_mask)
