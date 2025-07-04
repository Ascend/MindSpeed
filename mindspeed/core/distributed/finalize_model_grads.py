# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import List

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model, get_model_config


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad

            if hasattr(grad, "meta"):
                old_grad_scale = grad.meta.scale.clone()
                new_scale = grad.meta.scale
                torch.distributed.all_reduce(
                    new_scale,
                    op=torch.distributed.ReduceOp.MIN,
                    group=parallel_state.get_embedding_group()
                )
                grad.meta.scale.copy_(new_scale / len(parallel_state._EMBEDDING_GLOBAL_RANKS))
                grad.meta.scale_inv.copy_(1 / grad.meta.scale)
                grad.data.copy_((grad.data.float() / old_grad_scale * grad.meta.scale))

            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )

        if hasattr(grad, "meta"):
            old_grad_scale = grad.meta.scale.clone()
            new_scale = grad.meta.scale
            torch.distributed.all_reduce(
                new_scale,
                op=torch.distributed.ReduceOp.MIN,
                group=parallel_state.get_position_embedding_group()
            )
            grad.meta.scale.copy_(new_scale / len(parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS))
            grad.meta.scale_inv.copy_(1 / grad.meta.scale)
            grad.data.copy_((grad.data.float() / old_grad_scale * grad.meta.scale))

        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):
        grads = []
        scales = []
        metas = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                has_grad = param.requires_grad
                has_sequence_parallel = getattr(param, 'sequence_parallel', False)
                is_layernorm = 'q_layernorm' in name or 'k_layernorm' in name
                if has_grad and (has_sequence_parallel or is_layernorm):
                    grad = param.main_grad
                    grads.append(grad.data)
                    if hasattr(grad, "meta"):
                        scales.append(grad.meta.scale)
                        metas.append(grad.meta)
        if grads:
            if len(scales) > 0:
                scales = torch.cat(scales)
                old_grad_scales = scales.clone()
                torch.distributed.all_reduce(
                    scales,
                    op=torch.distributed.ReduceOp.MIN,
                    group=parallel_state.get_tensor_model_parallel_group()
                )
                for g, new_scale, old_scale, meta in zip(grads, scales, old_grad_scales, metas):
                    meta.scale.copy_(new_scale / parallel_state.get_tensor_model_parallel_world_size())
                    meta.scale_inv.copy_(1 / meta.scale)
                    g.data.copy_((g.float() / old_scale * meta.scale))
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)