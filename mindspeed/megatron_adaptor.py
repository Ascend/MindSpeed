import os
import sys
import shutil
import argparse
import time
from functools import wraps
from multiprocessing import Lock
import torch
from torch.distributed import all_gather_into_tensor, reduce_scatter_tensor
from torch_npu.contrib import transfer_to_npu
from mindspeed.features_manager import FEATURES_LIST
from .arguments import process_args


_ARGS = None


def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def get_mindspeed_args():
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='MindSpeed Arguments', allow_abbrev=False)
        _ARGS, unknown = process_args(parser).parse_known_args()
        parser_unknown_args(_ARGS, unknown)
    return _ARGS


def dummy_jit(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def lcm(a, b):
    import math
    return (a * b) // math.gcd(a, b)


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        if name == 'transformer-engine':
            return '0.0'
        res = fn(name, *args, **kwargs)
        return res

    return wrapper


# Patch view method to ensure tensor is contiguous before performing view
def ensure_contiguous_wrapper(fn):
    def wrapper(tensor, *args, **kwargs):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return fn(tensor, *args, **kwargs)

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    total_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_norm.clone())
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_norm = grad_norm ** norm_type
    return total_norm ** (1 / norm_type), ret_per_tensor


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'.format(len(tensor_lists[0]),
                                                                                              len(tensor_lists[1])))

    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def te_adaptation(aspm):
    aspm.register_patch('torch.compile', torch.jit.script)
    # Need replace modules before import megatron
    aspm.register_patch('importlib.metadata.version', version_wrapper)
    aspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.common.recipe.DelayedScaling', torch.nn.Module, create_dummy=True)
    aspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)


def apex_adaptation(aspm):
    from .core.fusions.fused_layer_norm import fused_layer_norm_affine
    from .ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
    aspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
    aspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
    aspm.register_patch('fused_layer_norm_cuda', create_dummy=True)
    aspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
    aspm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine,
                        create_dummy=True)
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)


def torch_adaptation(aspm):
    aspm.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
    aspm.register_patch('torch.Tensor.type', type_wrapper)
    aspm.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
    aspm.register_patch('torch.distributed._all_gather_base', all_gather_into_tensor)
    aspm.register_patch('torch.distributed._reduce_scatter_base', reduce_scatter_tensor)
    # lmc is supported python >=3.9
    if sys.version_info < (3, 9):
        aspm.register_patch('math.lcm', lcm, create_dummy=True)


def communication_adaptation(aspm, mindspeed_args):
    if mindspeed_args.disable_gloo_group:
        from mindspeed.optimizer.distrib_optimizer import get_parameter_state_dp_zero_hccl, \
            load_parameter_state_from_dp_zero_hccl
        from mindspeed.core.parallel_state import (get_data_parallel_group_gloo_replace,
                                                   get_data_modulo_expert_parallel_group_gloo_replace,
                                                  new_group_wrapper)
        from mindspeed.utils import check_param_hashes_across_dp_replicas_hccl

        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
                            get_parameter_state_dp_zero_hccl)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_parameter_state_from_dp_zero',
                            load_parameter_state_from_dp_zero_hccl)
        aspm.register_patch('megatron.core.utils.check_param_hashes_across_dp_replicas',
                            check_param_hashes_across_dp_replicas_hccl)

        aspm.register_patch('megatron.core.parallel_state.get_data_parallel_group_gloo',
                            get_data_parallel_group_gloo_replace)
        aspm.register_patch('megatron.core.parallel_state.get_data_modulo_expert_parallel_group_gloo',
                            get_data_modulo_expert_parallel_group_gloo_replace)
        aspm.register_patch('torch.distributed.new_group', new_group_wrapper)


def mcore_models_adaptation_l0(aspm):
    from .core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper
    from .core.parallel_state import get_nccl_options_wrapper
    # Replace FusedLayerNorm with MindSpeed's PTNorm operator in get_gpt-layer
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
                        get_gpt_layer_local_spec_wrapper)
    aspm.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)


def mcore_models_adaptation(aspm, mindspeed_args):
    import megatron.core
    megatron.core.jit.jit_fuser = dummy_jit

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from .core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank, \
        rotary_embedding_init_wrapper
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                        get_pos_emb_on_this_cp_rank)
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec',
                        get_gpt_layer_local_spec)
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
                        rotary_embedding_init_wrapper)
    from .core.models.common.embeddings.language_model_embedding import language_model_embedding_forward_wrapper
    aspm.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
                        language_model_embedding_forward_wrapper)
    from .core.models.common.embeddings.rotary_pos_embedding import rotary_embedding_get_rotary_seq_len_wrapper
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                        rotary_embedding_get_rotary_seq_len_wrapper)
    # Fix DDP scaling factor with Context Parallel
    from .core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_with_cp
    aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                        distributed_data_parallel_init_with_cp)

    if not mindspeed_args.automated_pipeline and mindspeed_args.noop_layers:
        from .core.transformer.transformer_block import _build_layers
        from .core.transformer.moe.moe_utils import track_moe_metrics
        from megatron.core.transformer.transformer_block import TransformerBlock
        from mindspeed.training import num_floating_point_wrapper
        TransformerBlock._build_layers = _build_layers
        aspm.register_patch('megatron.training.training.num_floating_point_operations', num_floating_point_wrapper)
        aspm.register_patch('megatron.core.transformer.moe.moe_utils.track_moe_metrics', track_moe_metrics)


    if mindspeed_args.recompute_norm:
        from .core.models.gpt.gpt_layer_specs import build_norm_recompute_layer_wrapper
        aspm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers', build_norm_recompute_layer_wrapper)
    
    if getattr(mindspeed_args, 'reset_attention_mask', False):
        from .core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids, collate_wrapper
        from .utils import get_batch_on_this_cp_rank_wrapper
        aspm.register_patch('megatron.core.datasets.gpt_dataset._get_ltor_masks_and_position_ids', _get_ltor_masks_and_position_ids)
        aspm.register_patch('torch.utils.data._utils.collate.default_collate', collate_wrapper)
        aspm.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank_wrapper)

        from mindspeed.core.pipeline_parallel.p2p_communication import _p2p_ops_eod 
        aspm.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops', _p2p_ops_eod)
        from mindspeed.core.models.gpt.gpt_model import gpt_forward_wrapper
        aspm.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_forward_wrapper)        
        from .core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_thd
        aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_thd', apply_rotary_pos_emb_thd)
        from .core.transformer.attention import attention_forward
        aspm.register_patch('megatron.core.transformer.attention.Attention.forward', attention_forward)

        from .core.models.common.embeddings.rotary_pos_embedding import rotary_forward
        aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward', rotary_forward)


def mcore_transformer_adaptation_l0(aspm):
    import megatron.core
    from .core.transformer.custom_layers.transformer_engine import PTNorm
    from .core.transformer.dot_product_attention import dot_product_attention_forward_wrapper, \
        dot_product_attention_init
    megatron.core.transformer.transformer_block.LayerNormImpl = PTNorm
    aspm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TENorm', PTNorm)
    # Add cp parameters to dot_deduct_mattention init, and add fusion attention support for alibi in non cp situations
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                        dot_product_attention_init)
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                        dot_product_attention_forward_wrapper)


def mcore_transformer_adaptation(aspm, args):
    from .core.transformer.module import megatron_module_init_wrapper
    from .core.transformer.attention import (attention_init, SelfAttentionSubmodules,
                                             self_attention_init_wrapper, attention_forward_wrapper)
    from .core.transformer.transformer_block import transformer_block_checkpointed_forward_wrapper
    from .core.transformer.transformer import parallel_transformer_layer_init_wrapper
    from .core.transformer.transformer import core_mlp_forward_wrapper
    from .core.transformer.mlp import mlp_init_2d_wrapper
    from .core.transformer.transformer_block import transformer_block_forward_wrapper
    aspm.register_patch('megatron.core.transformer.attention.SelfAttentionSubmodules', SelfAttentionSubmodules)
    aspm.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init_wrapper)
    aspm.register_patch("megatron.core.transformer.attention.Attention.forward", attention_forward_wrapper)
    aspm.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init)
    aspm.register_patch('megatron.core.transformer.module.MegatronModule.__init__', megatron_module_init_wrapper)
    aspm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                        transformer_block_checkpointed_forward_wrapper)
    aspm.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                        parallel_transformer_layer_init_wrapper)
    aspm.register_patch('megatron.core.transformer.mlp.MLP.forward',
                        core_mlp_forward_wrapper)
    aspm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init_2d_wrapper)
    aspm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                        transformer_block_forward_wrapper)
    if hasattr(args, "multi_head_latent_attention") and args.multi_head_latent_attention:
        from mindspeed.core.transformer.attention import self_attention_init_mla_wrapper
        aspm.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init_mla_wrapper)


def mcore_parallel_state_adaptation(aspm):
    from .core.parallel_state import initialize_model_parallel_wrapper
    from .core.parallel_state import destroy_model_parallel_wrapper
    from .core.memory.auto_pipeline.autopipeline_solver import destroy_model_parallel_profiling_wrapper
    from .core.parallel_state import get_context_parallel_group_for_send_recv_overlap
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                        initialize_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_profiling_wrapper)
    aspm.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                        get_context_parallel_group_for_send_recv_overlap)


def mcore_fusions_adaptation(aspm, args):
    from .core.fusions.fused_bias_swiglu import SwiGLUFunction, BiasSwiGLUFunction
    from .core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN
    from .core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from .core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_bshd
    aspm.register_patch('megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction',
                        FusedLayerNormAffineFunction)
    aspm.register_patch('megatron.core.fusions.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                        ScaledUpperTriangMaskedSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledSoftmax', ScaledSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                        is_kernel_available)
    aspm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                        forward_fused_softmax)
    aspm.register_patch('megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction', SwiGLUFunction)
    aspm.register_patch('megatron.core.fusions.fused_bias_swiglu.BiasSwiGLUFunction', BiasSwiGLUFunction)

    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_bshd',
                        apply_rotary_pos_emb_bshd)
    if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute:
        from .core.fusions.npu_moe_token_permute import permute_wrapper
        from .core.fusions.npu_moe_token_unpermute import unpermute_wrapper
        aspm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
        aspm.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_wrapper)
    if args.npu_deterministic:
        from mindspeed.initialize import deter_comp_wrapper
        aspm.register_patch('megatron.training.initialize._set_random_seed', deter_comp_wrapper)


def mcore_optimizer_adapation(aspm, mindspeed_args):
    from .optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
    from .optimizer.optimizer import (step_with_ready_grads, prepare_grads,
                                      reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper)
    from .core.distributed.param_and_grad_buffer import reuse_fp32_param_param_and_grad_buffer_init_wrapper
    # optim relative.
    aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
                        prepare_grads)
    aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step_with_ready_grads',
                        step_with_ready_grads)
    aspm.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                        reuse_fp32_param_init_wrapper)
    aspm.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                        optimizer_config_init_wrapper)
    aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                        reuse_fp32_param_distrib_optimizer_init_wrapper)
    aspm.register_patch('megatron.core.distributed.ParamAndGradBuffer.__init__',
                        reuse_fp32_param_param_and_grad_buffer_init_wrapper)

    if mindspeed_args.param_and_grad_buffer_pad:
        from .core.distributed.param_and_grad_buffer import param_and_grad_buffer_init_pad
        aspm.register_patch('megatron.core.distributed.ParamAndGradBuffer.__init__',
                            param_and_grad_buffer_init_pad)


def mcore_pipeline_parallel_adaptation(aspm, mindspeed_args):
    from .core.pipeline_parallel.schedules import get_tensor_shapes_wrapper, get_forward_backward_func_wrapper
    from .core.performance.auto_pipeline_perf.schedules import get_forward_backward_func_decorator, \
        backward_step_decorator, forward_step_decorator

    aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                        get_forward_backward_func_wrapper)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                        get_forward_backward_func_decorator)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.backward_step',
                        backward_step_decorator)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                        forward_step_decorator)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_tensor_shapes',
                        get_tensor_shapes_wrapper)
    if mindspeed_args.optimize_vpp_send_recv_comm:
        from .core.pipeline_parallel.p2p_communication import _p2p_ops_send_recv_overlap
        aspm.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops',
                            _p2p_ops_send_recv_overlap)
    if mindspeed_args.variable_seq_lengths:
        from .core.pipeline_parallel.p2p_communication import _communicate_shapes, _communicate
        aspm.register_patch('megatron.core.pipeline_parallel.p2p_communication._communicate',
                            _communicate)
        aspm.register_patch('megatron.core.pipeline_parallel.p2p_communication._communicate_shapes',
                            _communicate_shapes)


def mcore_multiparam_pipeline_parallel_adaptation(aspm, mindspeed_args):
    if mindspeed_args.use_multiparameter_pipeline_model_parallel:
        from .core.pipeline_parallel.multiparameter_schedules import get_tensor_shapes_wrapper, forward_step_wrapper, \
            recv_forward_wrapper, recv_backward_wrapper, send_forward_wrapper, send_backward_wrapper, \
            send_forward_recv_backward_wrapper, send_backward_recv_forward_wrapper, backward_step_wrapper

        aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_tensor_shapes',
                            get_tensor_shapes_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                            forward_step_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.backward_step',
                            backward_step_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.recv_forward',
                            recv_forward_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.recv_backward',
                            recv_backward_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.send_forward',
                            send_forward_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.send_backward',
                            send_backward_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.send_forward_recv_backward',
                            send_forward_recv_backward_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.send_backward_recv_forward',
                            send_backward_recv_forward_wrapper)


def mcore_tensor_parallel_adaptation_l0(aspm):
    from .core.tensor_parallel.random import _set_cuda_rng_state
    aspm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)


def mcore_tensor_parallel_adaptation_l1(aspm):
    from .core.tensor_parallel.cross_entropy import calculate_predicted_logits
    # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
    aspm.register_patch('megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
                        calculate_predicted_logits)


def mcore_tensor_parallel_adaptation(aspm, args):
    from .core.tensor_parallel.random import checkpoint_wrapper
    from .core.tensor_parallel.random import checkpoint_function_backward
    from .core.tensor_parallel.layers import vocab_parallel_embedding_forward
    from .core.tensor_parallel.layers import row_parallel_nocomm_optimizer_wrapper
    from .core.tensor_parallel.layers import parallel_linear_init_wrapper

    def has_recomputation_or_swap(args):
        return (args.swap_attention or
                args.recompute_in_bubble or
                args.adaptive_recompute_device_swap or
                args.recompute_in_advance or
                args.adaptive_memory_optimization)

    aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                        checkpoint_function_backward)
    aspm.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                        vocab_parallel_embedding_forward)
    aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                        row_parallel_nocomm_optimizer_wrapper)
    aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.__init__',
                        parallel_linear_init_wrapper)
    aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__',
                        parallel_linear_init_wrapper)
    aspm.register_patch('megatron.core.tensor_parallel.random.checkpoint', checkpoint_wrapper)
    if has_recomputation_or_swap(args):
        from .core.tensor_parallel.layers import linear_forward_main_grad_wrapper, linear_backward_main_grad_wrapper
        aspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                            linear_forward_main_grad_wrapper)
        aspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                            linear_backward_main_grad_wrapper)


def megatron_legacy_adaptation(aspm):
    from .model.language_model import parallel_lm_logits, embedding_forward_wrapper
    from .core.performance.auto_pipeline_perf.data_samplers import build_pretraining_data_loader_decorator
    from .core.performance.auto_pipeline_perf.transformer import get_attention_mask_wrapper
    aspm.register_patch('mindspeed.model.transformer.get_attention_mask', get_attention_mask_wrapper)
    aspm.register_patch('megatron.legacy.data.data_samplers.build_pretraining_data_loader',
                        build_pretraining_data_loader_decorator)
    aspm.register_patch('megatron.legacy.model.language_model.parallel_lm_logits', parallel_lm_logits)
    aspm.register_patch('megatron.legacy.model.language_model.Embedding.forward', embedding_forward_wrapper)


def legacy_model_fusions_adaptation(aspm):
    from .core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN, fused_layer_norm_affine
    from .core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.FusedLayerNormAffineFunction',
                        FusedLayerNormAffineFunction)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                        ScaledUpperTriangMaskedSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledSoftmax', ScaledSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                        is_kernel_available)
    aspm.register_patch('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                        forward_fused_softmax)


def legacy_model_rms_norm_adaptation(aspm):
    from .core.fusions.rms_norm import rms_norm_init_wrapper, rms_norm_forward_wrapper, rms_norm_norm_wrapper
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm.__init__', rms_norm_init_wrapper)
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm.forward', rms_norm_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm._norm', rms_norm_norm_wrapper)


def legacy_model_transformer_l0(aspm):
    from .model.transformer import parallel_mlp_init_wrapper, flash_self_attention_forward, \
        flash_self_attention_init_wrapper, parallel_transformer_forward_wrapper, flash_self_attention_init_add_config_wrapper
    from .model.transformer import parallel_attention_init, parallel_attention_forward
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.forward',
                        parallel_transformer_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.__init__', parallel_mlp_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.forward', flash_self_attention_forward)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.__init__',
                        flash_self_attention_init_add_config_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.__init__',
                        flash_self_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.__init__', parallel_attention_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.forward',
                        parallel_attention_forward)


def legacy_model_transformer(aspm, args):
    from .model.transformer import parallel_mlp_forward, parallel_transformer_init_wrapper, \
        parallel_transformer_init
    from .model.transformer import core_attention_init_wrapper, core_attention_forward
    from .core.transformer.transformer import parallel_transformer_checkpointed_forward_wrapper
    from .model.transformer import switch_mlp_init_wrapper, switch_mlp_forward_wrapper, \
        parallel_transformer_layer_init_wrapper
    if not args.automated_pipeline and args.noop_layers:
        aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', parallel_transformer_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__',
                        parallel_transformer_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.forward', parallel_mlp_forward)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.__init__', core_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.forward', core_attention_forward)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward',
                        parallel_transformer_checkpointed_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.__init__', switch_mlp_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.forward', switch_mlp_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.__init__',
                        parallel_transformer_layer_init_wrapper)


def megatron_training_adaptation_l0(aspm):
    from .initialize import _compile_dependencies, set_jit_fusion_options_wrapper
    from .utils import get_batch_on_this_cp_rank
    from .training import pretrain, get_device_wrapper
    from .arguments import parse_args_wrapper, validate_args_wrapper, core_transformer_config_from_args_wrapper
    from .yaml_arguments import core_transformer_config_from_yaml_wrapper, print_args_wrapper

    from .core.training import train_decorator, train_step_decorator
    from .core.transformer.transformer_config import transformer_config_post_init_wrapper
    aspm.register_patch('megatron.training.training.train', train_decorator)
    aspm.register_patch('megatron.training.training.train_step', train_step_decorator)
    aspm.register_patch('megatron.training.yaml_arguments.core_transformer_config_from_yaml',
                        core_transformer_config_from_yaml_wrapper)
    aspm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
    aspm.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
    aspm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
    aspm.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
    aspm.register_patch('megatron.training.arguments._print_args', print_args_wrapper)
    aspm.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args_wrapper)
    aspm.register_patch('megatron.training.yaml_arguments._print_args', print_args_wrapper)
    aspm.register_patch('megatron.training.arguments.core_transformer_config_from_args',
                        core_transformer_config_from_args_wrapper)
    aspm.register_patch('megatron.training.initialize.set_jit_fusion_options', set_jit_fusion_options_wrapper)
    aspm.register_patch('megatron.training.training.pretrain', pretrain)
    aspm.register_patch('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                        transformer_config_post_init_wrapper)
    aspm.register_patch('megatron.training.dist_signal_handler.get_device', get_device_wrapper)


def megatron_training_adaptation(aspm, mindspeed_args):
    from .core.performance.auto_pipeline_perf.global_vars import get_num_microbatches_wrapper
    from .core.training import training_log
    from .utils import get_batch_on_this_tp_rank
    from .tokenizer import build_tokenizer_wrapper
    from .core.training import pretrain_decorator, setup_model_and_optimizer_decorator
    aspm.register_patch('megatron.core.num_microbatches_calculator.get_num_microbatches', get_num_microbatches_wrapper)
    aspm.register_patch('megatron.training.training.pretrain', pretrain_decorator)
    aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_decorator)
    aspm.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)
    if mindspeed_args.op_cal_tflops:
        aspm.register_patch('megatron.training.training.training_log', training_log)
    aspm.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer_wrapper)


def megatron_training_ema_adaptation(aspm, mindspeed_args):
    if mindspeed_args.optimizer_selection == 'fused_ema_adamw':
        from .checkpointing import generate_state_dict_ema_wrapper, save_checkpoint_ema_wrapper
        from .optimizer.distrib_optimizer import ema_distrib_optimizer_init_wrapper
        aspm.register_patch('megatron.training.checkpointing.save_checkpoint', save_checkpoint_ema_wrapper)
        aspm.register_patch('megatron.training.checkpointing.generate_state_dict', generate_state_dict_ema_wrapper)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                            ema_distrib_optimizer_init_wrapper)
        if hasattr(mindspeed_args, "ema_decay"):
            from .optimizer.optimizer import get_megatron_optimizer_func_wrapper
            aspm.register_patch('megatron.core.optimizer.get_megatron_optimizer',
                                get_megatron_optimizer_func_wrapper)
    elif mindspeed_args.use_ema:
        from .training import pretrain, train_step
        from .checkpointing import save_checkpoint, _load_base_checkpoint
        aspm.register_patch('megatron.training.training.train_step', train_step)
        aspm.register_patch('megatron.training.checkpointing.save_checkpoint', save_checkpoint)
        aspm.register_patch('megatron.training.checkpointing._load_base_checkpoint', _load_base_checkpoint)


def memory_fragmentation_adaptation(aspm, args):
    from megatron.legacy.model.transformer import ParallelTransformerLayer
    if args.memory_fragmentation:
        from .core.memory.memory_fragmentation.pluggable_allocator_adpator import change_allocator
        time.sleep(10)
        change_allocator()

        from .core.memory.memory_fragmentation.memory_recorder import memory_recorder_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', memory_recorder_wrapper)

        from .core.memory.memory_fragmentation.malloc_recorder import malloc_recorder_wrapper
        aspm.register_patch('megatron.training.training.train_step', malloc_recorder_wrapper)

        from .core.memory.memory_fragmentation.optimizer_init_precise import optimizer_init_wrapper
        aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step', optimizer_init_wrapper)

        from .core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        allowed_recomputing_module_wrapper(ParallelTransformerLayer)
        from .core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)
    if (args.adaptive_recompute_enable and not args.memory_fragmentation) or args.swap_attention:
        from .core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        if hasattr(args, "use_legacy_models") and not args.use_legacy_models:
            from megatron.core.transformer.transformer_layer import TransformerLayer
            allowed_recomputing_module_wrapper(TransformerLayer)
        else:
            allowed_recomputing_module_wrapper(ParallelTransformerLayer)
        from .core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)
    if args.smart_swap and (not args.memory_fragmentation and not args.adaptive_recompute_enable):
        from .core.memory.smart_swap.swap_adaptor import change_allocator
        time.sleep(10)
        change_allocator()
        from .core.memory.smart_swap.swap_megatron_adaptor import train_step_wrapper
        aspm.register_patch('megatron.training.training.train_step', train_step_wrapper)
    if args.adaptive_memory_optimization and not (args.adaptive_recompute_enable or args.memory_fragmentation or args.swap_attention or args.smart_swap):
        from .core.memory.adaptive_memory.adaptive_memory_opt import addup_allowed_mem_adapt_module
        if hasattr(args, "use_legacy_models") and args.use_legacy_models:
            addup_allowed_mem_adapt_module(ParallelTransformerLayer)
        else:
            from megatron.core.transformer.transformer_layer import TransformerLayer
            addup_allowed_mem_adapt_module(TransformerLayer)
        from .core.memory.adaptive_memory.adaptive_memory_opt import setup_adapt_memory_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_adapt_memory_optimizer_wrapper)
        from .core.memory.adaptive_recomputing.pluggable_allocator_adpator import change_allocator
        time.sleep(10)
        change_allocator()

    if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE':
        print(f"OOTB_OPTIMIZER_PROFILING success open")
        from .core.memory.adaptive_recomputing.pluggable_allocator_adpator import change_allocator
        import megatron.training
        from mindspeed.auto_tuning.module.parse.recompute_parser import allowed_recompute_parser_module_wrapper
        allowed_recompute_parser_module_wrapper(megatron.legacy.model.transformer.ParallelTransformerLayer)
        from mindspeed.auto_tuning.module.parse.recompute_parser import setup_model_and_optimizer_decorator
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_decorator)
        print(f"setup_model_and_optimizer_decorator success")

    if args.adaptive_recompute_enable or args.memory_fragmentation:
        import megatron.training.initialize
        aspm.register_patch('megatron.training.initialize_megatron', megatron.training.initialize.initialize_megatron)


def mcore_moe_adaptation_l0(pm):
    from .core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, get_device_capability, \
        assert_grouped_gemm_is_available
    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
                      grouped_gemm_is_available)
    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.assert_grouped_gemm_is_available',
                      assert_grouped_gemm_is_available)
    pm.register_patch('torch.cuda.get_device_capability', get_device_capability)


def mcore_moe_adaptation(pm, args):
    from .core.pipeline_parallel.schedules import forward_step
    pm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                        forward_step)
    if args.moe_permutation_async_comm:
        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall':
            from .core.transformer.moe.experts import sequential_mlp_forward
            from .core.transformer.moe.moe_utils import permute, unpermute
            if args.moe_tp_extend_ep:
                from .core.transformer.moe.token_dispatcher import (
                    preprocess_tp_extend_ep, alltoall_token_unpermutation_tp_extend_ep,
                    alltoall_token_permutation_tp_extend_ep
                )
                from .core.transformer.moe.router import routing_tp_extend_ep
                from .core.transformer.moe.moe_layer import base_moe_init_wrapper
                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__',
                                  base_moe_init_wrapper)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
                    preprocess_tp_extend_ep)
                pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing', routing_tp_extend_ep)

                if args.moe_alltoall_overlap_comm:
                    from .core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \
                        alltoall_token_unpermutation_new
                    from .core.transformer.moe.experts import group_mlp_forward
                    from .core.transformer.mlp import mlp_init
                    from .core.transformer.moe.moe_layer import moe_layer_init
                    pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new)
                    pm.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', moe_layer_init)
                else:
                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                                      alltoall_token_permutation_tp_extend_ep)
                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                                      alltoall_token_unpermutation_tp_extend_ep)
            else:
                from .core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation, \
                    alltoall_token_unpermutation_with_bmm
                pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
                                  preprocess)
                if args.moe_alltoall_overlap_comm:
                    from .core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \
                        alltoall_token_unpermutation_new
                    from .core.transformer.moe.experts import group_mlp_forward
                    from .core.transformer.mlp import mlp_init
                    from .core.transformer.moe.moe_layer import moe_layer_init
                    pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new)
                    pm.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', moe_layer_init)
                else:
                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                                      alltoall_token_permutation)
                    if args.moe_bmm_mc2:
                        pm.register_patch(
                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                            alltoall_token_unpermutation_with_bmm)
            pm.register_patch('megatron.core.transformer.moe.experts.SequentialMLP.forward', sequential_mlp_forward)
            pm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute)
            pm.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute)
        else:
            from .core.transformer.moe.router import aux_loss_load_balancing
            pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing', aux_loss_load_balancing)

            if args.moe_tp_extend_ep:
                from .core.transformer.moe.moe_layer import base_moe_init_wrapper
                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__', base_moe_init_wrapper)

            if args.moe_allgather_overlap_comm:
                from .core.transformer.moe.token_dispatcher import (allgather_token_permutation_new,
                                                                    allgather_token_unpermutation_new)
                from .core.transformer.moe.experts import group_mlp_forward
                from .core.transformer.mlp import mlp_init
                pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
                pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
                    allgather_token_permutation_new)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
                    allgather_token_unpermutation_new)
            else:
                from .core.transformer.moe.token_dispatcher import (allgather_token_permutation,
                                                                    allgather_token_unpermutation)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
                    allgather_token_permutation)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
                    allgather_token_unpermutation)

        from .core.transformer.moe.moe_layer import moe_layer_init_wrapper
        pm.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', moe_layer_init_wrapper)
    else:
        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall':
            from .core.transformer.moe.token_dispatcher import alltoall_preprocess_npu, \
                alltoall_token_unpermutation_with_bmm, alltoall_token_permutation_with_bmm
            pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
                    alltoall_preprocess_npu)
            if args.moe_bmm_mc2:
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                    alltoall_token_permutation_with_bmm)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                    alltoall_token_unpermutation_with_bmm)
        else:
            from .core.transformer.moe.token_dispatcher import allgather_token_permutation_npu
            pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation', allgather_token_permutation_npu)
    
    from .core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward
    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.__init__', groupedmlp_init_wrapper)
    if not args.moe_alltoall_overlap_comm and not args.moe_allgather_overlap_comm:
        pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', groupedmlp_forward)

    if args.use_ascend_mc2 and not hasattr(args, 'moe_grouped_gemm'):
        # MoE MLP not use mc2 linear
        from .core.models.gpt.gpt_layer_specs import build_layers_wrapper
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.transformer_block import TransformerBlock
        TransformerBlock._build_layers = build_layers_wrapper(TransformerBlock._build_layers,
                                                              ColumnParallelLinear.forward,
                                                              RowParallelLinear.forward)


def deepspeed_moe_adaptation(pm, args):
    if args.use_pipe_experts or args.use_nanopipe or args.ampipe_degree > 1:
        from .core.tensor_parallel.layers import (row_parallel_moe, column_parallel_moe,
                                                  linear_with_grad_accumulation_and_async_allreduce_moe)
        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward', row_parallel_moe)
        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward', column_parallel_moe)
        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                          linear_with_grad_accumulation_and_async_allreduce_moe)
    if args.use_pipe_experts:
        from .core.distributed.param_and_grad_buffer import pipe_register_grad_ready
        pm.register_patch('megatron.core.distributed.ParamAndGradBuffer.register_grad_ready', pipe_register_grad_ready)
    if args.ampipe_degree > 1:
        from mindspeed.model.language_model import embedding_forward_ampipe
        from mindspeed.model.transformer import parallel_transformer_forward_ampipe
        from mindspeed.model.transformer import parallel_transformer_layer_forward_ampipe
        pm.register_patch('megatron.legacy.model.language_model.Embedding.forward', embedding_forward_ampipe)
        pm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.forward',
                          parallel_transformer_forward_ampipe)
        pm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.forward',
                          parallel_transformer_layer_forward_ampipe)


def coc_adaptation(aspm, args):
    from .initialize import coc_registration_wrapper, mc2_wrapper
    if args.use_ascend_mc2:
        from .core.memory.auto_pipeline.autopipeline import initialize_cfg_from_args_wrapper
        aspm.register_patch('megatron.training.initialize.initialize_megatron', mc2_wrapper)
        aspm.register_patch('mindspeed.core.tensor_parallel.ascend_turbo.initialize.initialize_cfg_from_args',
                            initialize_cfg_from_args_wrapper)
    if args.use_ascend_coc:
        aspm.register_patch('megatron.training.initialize.initialize_megatron', coc_registration_wrapper)


def zero3_adaptation(aspm, args):
    if args.enable_zero3:
        from .core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_zero3, \
            distributed_data_parallel_zero_grad_wrapper
        from .core.tensor_parallel.layers import (parallel_linear_init_zero3_wrapper,
                                                  column_parallel_linear_forward_zero3,
                                                  linear_forward_zero3_wrapper, linear_backward_zero3_wrapper,
                                                  row_parallel_linear_forward_zero3,
                                                  linear_with_grad_accumulation_and_async_allreduce_zero3)
        from .optimizer.distrib_optimizer import (build_optimizer_group_ranges_zero3_wrapper,
                                                  _copy_main_params_to_model_params_zero3,
                                                  _copy_model_grads_to_main_grads_zero3,
                                                  build_model_and_main_param_groups_zero3_wrapper,
                                                  distributed_optimizer_zero3_init)
        aspm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                            linear_with_grad_accumulation_and_async_allreduce_zero3)
        aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.__init__',
                            parallel_linear_init_zero3_wrapper)
        aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__',
                            parallel_linear_init_zero3_wrapper)
        aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward',
                            column_parallel_linear_forward_zero3)
        aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                            row_parallel_linear_forward_zero3)
        aspm.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._build_optimizer_group_ranges',
            build_optimizer_group_ranges_zero3_wrapper)
        aspm.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_main_params_to_model_params',
            _copy_main_params_to_model_params_zero3)
        aspm.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads',
            _copy_model_grads_to_main_grads_zero3)
        aspm.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._build_model_and_main_param_groups',
            build_model_and_main_param_groups_zero3_wrapper)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                            distributed_optimizer_zero3_init)
        aspm.register_patch(
            'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
            linear_forward_zero3_wrapper)
        aspm.register_patch(
            'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
            linear_backward_zero3_wrapper)
        aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                            distributed_data_parallel_init_zero3)
        aspm.register_patch(
            'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.zero_grad_buffer',
            distributed_data_parallel_zero_grad_wrapper)


def tensor_2d_adaptation(aspm, args):
    if args.tp_2d:
        from mindspeed.core.tensor_parallel.tp_2d.norm_factory import get_norm_tp_2d
        from mindspeed.core.tensor_parallel.tp_2d.norm_factory import _allreduce_layernorm_grads_wrapper
        from mindspeed.core.models.common.embeddings.rotary_pos_embedding import rotary_embedding_forward_wrapper
        from mindspeed.core.pipeline_parallel.flexible_schedules import forward_backward_pipelining_with_interleaving_patch
        aspm.register_patch('megatron.legacy.model.utils.get_norm', get_norm_tp_2d)
        aspm.register_patch('megatron.core.distributed.finalize_model_grads._allreduce_layernorm_grads',
                            _allreduce_layernorm_grads_wrapper)
        aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward',
                            rotary_embedding_forward_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                            forward_backward_pipelining_with_interleaving_patch)
        from .core.transformer.transformer_config import transformer_config_post_init
        aspm.register_patch('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                            transformer_config_post_init)
        from mindspeed.model.language_model import model_parallel_config_post_init_wrapper
        aspm.register_patch('megatron.core.model_parallel_config.ModelParallelConfig.__post_init__',
                            model_parallel_config_post_init_wrapper)
        from mindspeed.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_wrapper
        aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs._get_mlp_module_spec',
                            get_mlp_module_spec_wrapper)
        from mindspeed.core.transformer.attention import self_attention_init_tp2d_wrapper
        aspm.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init_tp2d_wrapper)


def megatron_training_adaptation_with_layerzero(aspm, mindspeed_args):
    '''This function is used to add layerzero feature within mindspeed
    layerzero manages the paramter in a different manner compared to Megatron Optimizer
    
    So if layerzero is on, setup_model_and_optimizer will return a module wrapped by layerzero and the Optimizer will be replaced. 
    '''
    if mindspeed_args.layerzero:
        from mindspeed.core.distributed.layerzero import (layerzero_setup_model_and_optimizer_wrapper, 
                                                        layerzero_initialize_model_parallel_wrapper, 
                                                        mga_finalize_model_grads_wrapper,
                                                        save_checkpoint,
                                                        )
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', layerzero_setup_model_and_optimizer_wrapper)
        aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel', layerzero_initialize_model_parallel_wrapper)
        aspm.register_patch('megatron.core.distributed.finalize_model_grads', mga_finalize_model_grads_wrapper)
        aspm.register_patch('megatron.training.checkpointing.save_checkpoint', save_checkpoint)


def auto_parallel_mm_adaptation(aspm, mindspeed_args):
    from mindspeed.core.auto_parallel.mm_search.schedules import backward_step_decorator
    if mindspeed_args.auto_parallel_mm or mindspeed_args.auto_parallel_profile:
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.backward_step',
                            backward_step_decorator)


def dist_train_adaptation(aspm, args):
    if args.dist_train:
        from mindspeed.multi_modal import dist_train
        # pipeline parallel adaption
        aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func', dist_train.pipeline_parallel.dist_schedules.get_forward_backward_func_wrapper)
        aspm.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops', dist_train.pipeline_parallel.dist_schedules.p2p_ops_wrapper)
        # parallel state adaption
        aspm.register_patch('megatron.training.initialize._initialize_distributed', dist_train.training.initialize_distributed_wrapper)
        aspm.register_patch('megatron.core.mpu.initialize_model_parallel', dist_train.parallel_state.initialize_model_parallel)
        aspm.register_patch('megatron.core.mpu.is_pipeline_last_stage', dist_train.parallel_state.get_is_pipeline_last_stage_wrapper)
        aspm.register_patch('megatron.core.mpu.is_pipeline_first_stage', dist_train.parallel_state.get_is_pipeline_first_stage_wrapper)
        aspm.register_patch('megatron.core.mpu.get_tensor_model_parallel_src_rank', dist_train.parallel_state.get_tensor_model_parallel_src_rank_wrapper)
        aspm.register_patch('megatron.core.mpu.is_initialized', dist_train.parallel_state.is_initialized)
        aspm.register_patch('megatron.core.mpu.model_parallel_is_initialized', dist_train.parallel_state.model_parallel_is_initialized)
        aspm.register_patch('megatron.core.mpu.get_model_parallel_group', dist_train.parallel_state.get_model_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_tensor_model_parallel_group', dist_train.parallel_state.get_tensor_model_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_group', dist_train.parallel_state.get_pipeline_model_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_data_parallel_group', dist_train.parallel_state.get_data_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_data_parallel_group_gloo', dist_train.parallel_state.get_data_parallel_group_gloo)
        aspm.register_patch('megatron.core.mpu.get_context_parallel_group', dist_train.parallel_state.get_context_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_context_parallel_global_ranks', dist_train.parallel_state.get_context_parallel_global_ranks)
        aspm.register_patch('megatron.core.mpu.get_embedding_group', dist_train.parallel_state.get_embedding_group)
        aspm.register_patch('megatron.core.mpu.get_position_embedding_group', dist_train.parallel_state.get_position_embedding_group)
        aspm.register_patch('megatron.core.mpu.get_data_modulo_expert_parallel_group_gloo', dist_train.parallel_state.get_data_modulo_expert_parallel_group_gloo)
        aspm.register_patch('megatron.core.mpu.get_amax_reduction_group', dist_train.parallel_state.get_amax_reduction_group)
        aspm.register_patch('megatron.core.mpu.get_tensor_and_data_parallel_group', dist_train.parallel_state.get_tensor_and_data_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_tensor_and_context_parallel_group', dist_train.parallel_state.get_tensor_and_context_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_expert_model_parallel_group', dist_train.parallel_state.get_expert_model_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_tensor_and_expert_parallel_group', dist_train.parallel_state.get_tensor_and_expert_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_data_modulo_expert_parallel_group', dist_train.parallel_state.get_data_modulo_expert_parallel_group)
        aspm.register_patch('megatron.core.mpu.get_tensor_model_parallel_world_size', dist_train.parallel_state.get_tensor_model_parallel_world_size)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_world_size', dist_train.parallel_state.get_pipeline_model_parallel_world_size)
        aspm.register_patch('megatron.core.mpu.get_tensor_model_parallel_rank', dist_train.parallel_state.get_tensor_model_parallel_rank)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_rank', dist_train.parallel_state.get_pipeline_model_parallel_rank)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_split_rank', dist_train.parallel_state.get_pipeline_model_parallel_split_rank)
        aspm.register_patch('megatron.core.mpu.is_rank_in_embedding_group', dist_train.parallel_state.is_rank_in_embedding_group)
        aspm.register_patch('megatron.core.mpu.is_rank_in_position_embedding_group', dist_train.parallel_state.is_rank_in_position_embedding_group)
        aspm.register_patch('megatron.core.mpu.get_virtual_pipeline_model_parallel_rank', dist_train.parallel_state.get_virtual_pipeline_model_parallel_rank)
        aspm.register_patch('megatron.core.mpu.get_virtual_pipeline_model_parallel_world_size', dist_train.parallel_state.get_virtual_pipeline_model_parallel_world_size)
        aspm.register_patch('megatron.core.mpu.get_data_parallel_src_rank', dist_train.parallel_state.get_data_parallel_src_rank)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_first_rank', dist_train.parallel_state.get_pipeline_model_parallel_first_rank)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_last_rank', dist_train.parallel_state.get_pipeline_model_parallel_last_rank)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_next_rank', dist_train.parallel_state.get_pipeline_model_parallel_next_rank)
        aspm.register_patch('megatron.core.mpu.get_pipeline_model_parallel_prev_rank', dist_train.parallel_state.get_pipeline_model_parallel_prev_rank)
        aspm.register_patch('megatron.core.mpu.get_expert_model_parallel_world_size', dist_train.parallel_state.get_expert_model_parallel_world_size)
        aspm.register_patch('megatron.core.mpu.get_expert_model_parallel_rank', dist_train.parallel_state.get_expert_model_parallel_rank)
        aspm.register_patch('megatron.core.mpu.get_global_memory_buffer', dist_train.parallel_state.get_global_memory_buffer)
        aspm.register_patch('megatron.core.mpu.get_moe_layer_wise_logging_tracker', dist_train.parallel_state.get_moe_layer_wise_logging_tracker)
        # checkpoint
        aspm.register_patch('megatron.training.checkpointing.get_checkpoint_name', dist_train.checkpointing.get_checkpoint_name_wrapper)


def optimizer_selection(aspm, mindspeed_args):
    if mindspeed_args.optimizer_selection == 'fused_torch_adamw':
        from .optimizer.adamw import FusedTorchAdamW as AdamW
    elif mindspeed_args.optimizer_selection == 'fused_adamw':
        from .optimizer.adamw import AdamW
    elif mindspeed_args.optimizer_selection == 'fused_ema_adamw':
        from .optimizer.ema_adamw import FusedEmaAdamW as AdamW
    aspm.register_patch('apex.optimizers.FusedAdam', AdamW, create_dummy=True)


def adaptation_l0(aspm, mindspeed_args):
    """
    The minimum patch set for megatron to adapt to NPU
    """
    # transformer_engine
    te_adaptation(aspm)
    apex_adaptation(aspm)
    torch_adaptation(aspm)
    # Need replace transformer_engine modules before import megatron
    aspm.apply_patches()

    mcore_models_adaptation_l0(aspm)
    mcore_tensor_parallel_adaptation_l0(aspm)
    mcore_transformer_adaptation_l0(aspm)
    mcore_moe_adaptation_l0(aspm)
    legacy_model_transformer_l0(aspm)
    megatron_training_adaptation_l0(aspm)
    # context parallel(ring attention) requires mcore parallel state patch
    mcore_parallel_state_adaptation(aspm)
    communication_adaptation(aspm, mindspeed_args)


def adaptation_l1(aspm, mindspeed_args):
    """
    Affinity optimization (fusion operators, etc.)
    """
    # fusion operators
    mcore_fusions_adaptation(aspm, mindspeed_args)
    legacy_model_fusions_adaptation(aspm)
    # affinity optimization
    mcore_tensor_parallel_adaptation_l1(aspm)


def adaptation_l2(aspm, mindspeed_args):
    """
    Advanced acceleration algorithm
    """
    mcore_models_adaptation(aspm, mindspeed_args)
    mcore_optimizer_adapation(aspm, mindspeed_args)
    mcore_pipeline_parallel_adaptation(aspm, mindspeed_args)
    mcore_multiparam_pipeline_parallel_adaptation(aspm, mindspeed_args)
    mcore_tensor_parallel_adaptation(aspm, mindspeed_args)
    mcore_transformer_adaptation(aspm, mindspeed_args)

    # megatron legacy
    megatron_legacy_adaptation(aspm)
    legacy_model_transformer(aspm, mindspeed_args)
    legacy_model_rms_norm_adaptation(aspm)

    megatron_training_adaptation(aspm, mindspeed_args)
    megatron_training_ema_adaptation(aspm, mindspeed_args)
    memory_fragmentation_adaptation(aspm, mindspeed_args)
    coc_adaptation(aspm, mindspeed_args)
    mcore_moe_adaptation(aspm, mindspeed_args)
    deepspeed_moe_adaptation(aspm, mindspeed_args)
    zero3_adaptation(aspm, mindspeed_args)
    tensor_2d_adaptation(aspm, mindspeed_args)
    auto_parallel_mm_adaptation(aspm, mindspeed_args)
    dist_train_adaptation(aspm, mindspeed_args)


def delete_lock_file(directory, lock):
    with lock:
        flag_lock = False
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for name in files:
                    if name.endswith('.lock') or name.endswith('lock'):
                        if os.path.exists(directory):
                            flag_lock = True
                            print(f"Process (PID: {os.getpid()}) is deleting Lock directory")
                            shutil.rmtree(directory)
                            print(f"Process (PID: {os.getpid()}) deleted Lock directory")
                            if flag_lock:
                                break
                        else:
                            print(f"Process (PID: {os.getpid()}) Directory {directory} does not exist.")
                if flag_lock:
                    break


def exe_adaptation():
    modified_argv_path = os.getenv("OOTB_OPTIMIZER_MODIFIED_ARGV_PATH", None)
    if modified_argv_path:
        from mindspeed.auto_tuning.mindspeed_adaptor import MindSpeedAdaptor
        MindSpeedAdaptor.set_argv(sys.argv, modified_argv_path)
        print("================OOTB_OPTIMIZER_MODIFIED_ARGV DONE!====================")
    mindspeed_args = get_mindspeed_args()

    from torch.utils.cpp_extension import _get_build_directory
    build_directory = _get_build_directory("", True)
    delete_lock = Lock()
    delete_lock_file(build_directory, delete_lock)
    mindspeed_args.adaptive_recompute_enable = mindspeed_args.adaptive_recompute_device_size > 0 or mindspeed_args.adaptive_recompute_device_swap
    if (mindspeed_args.adaptive_recompute_enable and not mindspeed_args.memory_fragmentation) or mindspeed_args.swap_attention:
        from .core.memory.adaptive_recomputing.pluggable_allocator_adpator import change_allocator
        if not mindspeed_args.swap_attention:
            time.sleep(10)
            change_allocator()
    from .patch_utils import MindSpeedPatchesManager as aspm

    if mindspeed_args.optimization_level >= 0:
        # The minimum patch set for megatron to adapt to NPU
        optimizer_selection(aspm, mindspeed_args)
        adaptation_l0(aspm, mindspeed_args)

    if mindspeed_args.optimization_level >= 1:
        # Affinity optimization (fusion operators, etc.)
        adaptation_l1(aspm, mindspeed_args)

    if mindspeed_args.optimization_level >= 2:
        # Advanced acceleration algorithm
        adaptation_l2(aspm, mindspeed_args)
        
    if mindspeed_args.layerzero:
        # layerzero features
        megatron_training_adaptation_with_layerzero(aspm, mindspeed_args)
        
    aspm.apply_patches()

    # New features structure
    for feature in FEATURES_LIST:
        if getattr(mindspeed_args, feature.feature_name, None) or feature.default_patches:
            feature.register_patches(aspm, mindspeed_args)

    aspm.apply_patches()

    # accelerate package will check TE on sys.modules，so we need remove this patch
    del sys.modules['transformer_engine']


exe_adaptation()
