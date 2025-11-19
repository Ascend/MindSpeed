import os
from typing import List

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import (
    _get_main_grad_attr,
    _reshard_if_dtensor,
    _unshard_if_dtensor,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias


def _requantize_grad(grad: torch.Tensor, meta, new_scale: torch.Tensor, old_scale: torch.Tensor) -> None:
    """Update quantization metadata and bring grad data into the new scale domain."""
    meta.scale.copy_(new_scale)
    meta.scale_inv.copy_(1 / new_scale)
    adjusted = grad.data.float() / old_scale * new_scale
    grad.data.copy_(adjusted.to(grad.dtype))


def _maybe_adjust_quant_scale(
    grad: torch.Tensor, group: torch.distributed.ProcessGroup, divide_world: bool = True
) -> None:
    """Synchronize quantized gradient scales across a communication group."""
    if grad is None:
        return
    meta = getattr(grad, "meta", None)
    scale = getattr(meta, "scale", None) if meta is not None else None
    scale_inv = getattr(meta, "scale_inv", None) if meta is not None else None
    if meta is None or scale is None or scale_inv is None:
        return

    old_scale = scale.clone()
    new_scale = scale.clone()
    torch.distributed.all_reduce(new_scale, op=torch.distributed.ReduceOp.MIN, group=group)

    world_size = torch.distributed.get_world_size(group)
    requires_division = (
        divide_world
        and world_size > 1
        and getattr(meta, "qtype", None) not in (4,)
    )
    if requires_division:
        new_scale = new_scale / world_size

    _requantize_grad(grad, meta, new_scale, old_scale)


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and torch.distributed.get_world_size(parallel_state.get_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:
            model_module = model[0]

        ddp_config = model_module.ddp_config
        model_module = get_attr_wrapped_model(model_module, "pre_process", return_model_obj=True)

        if model_module.share_embeddings_and_output_weights or getattr(config, "mtp_num_layers", 0):
            weight = model_module.shared_embedding_or_output_weight()
            grad_attr = _get_main_grad_attr(weight, ddp_config.use_custom_fsdp)
            orig_grad = getattr(weight, grad_attr)
            grad = _unshard_if_dtensor(orig_grad)
            _maybe_adjust_quant_scale(grad, parallel_state.get_embedding_group())
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and torch.distributed.get_world_size(parallel_state.get_position_embedding_group()) > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:
            model_module = model[0]

        ddp_config = model_module.ddp_config
        model_module = get_attr_wrapped_model(model_module, "pre_process", return_model_obj=True)
        assert hasattr(model_module, "position_embeddings")

        weight = model_module.position_embeddings.weight
        grad_attr = _get_main_grad_attr(weight, ddp_config.use_custom_fsdp)
        orig_grad = getattr(weight, grad_attr)
        grad = _unshard_if_dtensor(orig_grad)
        _maybe_adjust_quant_scale(grad, parallel_state.get_position_embedding_group())
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)


def _accumulate_quant_grad(target: torch.Tensor, source: torch.Tensor) -> None:
    """Accumulate ``source`` into ``target`` while respecting quant metadata."""
    if target is source:
        return
    target_meta = getattr(target, "meta", None)
    source_meta = getattr(source, "meta", None)
    if target_meta is None and source_meta is None:
        target.add_(source)
        return

    with torch.no_grad():
        target_fp32 = (
            target_meta.dequantization(target.data) if target_meta is not None else target.data.float()
        )
        source_fp32 = (
            source_meta.dequantization(source.data) if source_meta is not None else source.data.float()
        )
        target_fp32.add_(source_fp32)
        if target_meta is not None:
            target.data.copy_(target_meta.quantization(target_fp32))
        else:
            target.data.copy_(target_fp32.to(dtype=target.dtype))


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):
        params = []
        grads = []
        grad_attrs = []
        orig_grads = []
        quant_entries = []
        group = parallel_state.get_tensor_model_parallel_group()
        world_size = parallel_state.get_tensor_model_parallel_world_size()

        for model_chunk in model:
            ddp_config = model_chunk.ddp_config
            for name, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
                if not param.requires_grad:
                    continue
                has_sequence_parallel = getattr(param, "sequence_parallel", False)
                is_layernorm = "q_layernorm" in name or "k_layernorm" in name
                if not (has_sequence_parallel or is_layernorm):
                    continue

                grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
                grad = getattr(param, grad_attr)
                orig_grad = grad
                grad = _unshard_if_dtensor(grad)

                params.append(param)
                grads.append(grad.data)
                grad_attrs.append(grad_attr)
                orig_grads.append(orig_grad)

                meta = getattr(grad, "meta", None)
                scale = getattr(meta, "scale", None) if meta is not None else None
                if meta is not None and scale is not None:
                    quant_entries.append((grad, meta, scale.clone()))

        if grads:
            if quant_entries:
                concat_scales = torch.cat([meta.scale.view(-1) for _, meta, _ in quant_entries])
                torch.distributed.all_reduce(concat_scales, op=torch.distributed.ReduceOp.MIN, group=group)
                if world_size > 1:
                    concat_scales = concat_scales / world_size
                offset = 0
                for grad, meta, old_scale in quant_entries:
                    length = meta.scale.numel()
                    new_scale = concat_scales[offset: offset + length].view_as(meta.scale)
                    _requantize_grad(grad, meta, new_scale, old_scale)
                    offset += length

            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=group)
            for param, grad_attr, orig_grad, buf, synced in zip(
                params,
                grad_attrs,
                orig_grads,
                grads,
                _unflatten_dense_tensors(coalesced, grads),
            ):
                buf.copy_(synced)
                setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def _allreduce_conditional_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig) -> None:
    if parallel_state.get_pipeline_model_parallel_world_size() <= 1 or not getattr(
        config, "has_cond_embedder", False
    ):
        return

    group = parallel_state.get_pipeline_model_parallel_group()
    name_to_entry = {}
    primary_entries = []

    for model_chunk in model:
        ddp_config = getattr(model_chunk, "ddp_config", None)
        for name, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
            if not param.requires_grad or not getattr(param, "pipeline_parallel", False):
                continue
            grad_attr = _get_main_grad_attr(param, getattr(ddp_config, "use_custom_fsdp", False))
            orig_grad = getattr(param, grad_attr, None)
            if orig_grad is None:
                continue
            grad = _unshard_if_dtensor(orig_grad)
            entry = {
                "param": param,
                "grad_attr": grad_attr,
                "orig_grad": orig_grad,
                "grad": grad,
                "meta": getattr(grad, "meta", None),
            }
            if name not in name_to_entry:
                record = {"primary": entry, "extras": []}
                name_to_entry[name] = len(primary_entries)
                primary_entries.append(record)
            else:
                record = primary_entries[name_to_entry[name]]
                _accumulate_quant_grad(record["primary"]["grad"], grad)
                record["extras"].append(entry)

    if not primary_entries:
        return

    grads = []
    for record in primary_entries:
        grad = record["primary"]["grad"]
        _maybe_adjust_quant_scale(grad, group, divide_world=False)
        grads.append(grad.data)

    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=group)

    for record, synced in zip(primary_entries, _unflatten_dense_tensors(coalesced, grads)):
        primary = record["primary"]
        primary["grad"].data.copy_(synced)
        setattr(
            primary["param"],
            primary["grad_attr"],
            _reshard_if_dtensor(primary["grad"], primary["orig_grad"]),
        )

        meta = primary["meta"]
        for extra in record["extras"]:
            if meta is not None and getattr(extra["grad"], "meta", None) is not None:
                extra_meta = extra["grad"].meta
                extra_meta.scale.copy_(meta.scale)
                extra_meta.scale_inv.copy_(meta.scale_inv)
            extra["grad"].data.copy_(primary["grad"].data)
            setattr(
                extra["param"],
                extra["grad_attr"],
                _reshard_if_dtensor(extra["grad"], extra["orig_grad"]),
            )


def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig) -> None:
    if not getattr(config, "moe_router_enable_expert_bias", False):
        return

    tokens_per_expert_list = []
    expert_bias_list = []
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, "modules")():
            if hasattr(module, "expert_bias"):
                tokens_per_expert_list.append(module.local_tokens_per_expert)
                expert_bias_list.append(module.expert_bias)

    if len(expert_bias_list) == 0:
        return

    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert, stacked_expert_bias, config.moe_router_bias_update_rate
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
        tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
    ):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)
