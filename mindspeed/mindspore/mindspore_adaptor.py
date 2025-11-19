# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

def mindspore_adaptation(aspm, mindspeed_args):
    if not hasattr(mindspeed_args, "ai_framework") or mindspeed_args.ai_framework != "mindspore" or mindspeed_args.optimization_level < 0:
        return

    from .optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
    aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                        reuse_fp32_param_distrib_optimizer_init_wrapper, force_patch=True)

    from .core.models.common.embeddings.rotary_pos_embedding import local_rotate_half  # resolve warning
    aspm.register_patch('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

    from .core.pipeline_parallel.schedules import deallocate_output_tensor_
        
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.deallocate_output_tensor',
                        deallocate_output_tensor_)

    from .core.tensor_parallel.random import local_set_cuda_rng_state
    aspm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', local_set_cuda_rng_state,
                        force_patch=True)

    from .core.timers import _get_global_min_max_time
    aspm.register_patch('megatron.core.timers.Timers._get_global_min_max_time', _get_global_min_max_time)

    from mindspeed.mindspore.ops.npu_matmul_add import npu_matmul_add_fp32
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, force_patch=True)
    aspm.register_patch('mindspeed.ops.npu_matmul_add.npu_matmul_add_fp32', npu_matmul_add_fp32)

    from mindspeed.mindspore.core.optimizer.adamw import step_func
    aspm.register_patch('apex.optimizers.FusedAdam.step', step_func)

    from mindspeed.mindspore.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
    aspm.register_patch("mindspeed.ops.npu_rotary_position_embedding.npu_rotary_position_embedding", npu_rotary_position_embedding)

    from mindspeed.mindspore.ops.npu_ring_attention_update import _RING_PROXY
    aspm.register_patch("mindspeed.ops.npu_ring_attention_update.npu_ring_attention_update", _RING_PROXY.npu_ring_attention_update)

    # Resolve the issue of being unable to assign values to `tensor.data` when `tensor.requires_grad` is set to `True`.
    from mindspeed.mindspore.core.utils import _kernel_make_viewless_tensor
    aspm.register_patch('megatron.core.utils._kernel_make_viewless_tensor', _kernel_make_viewless_tensor)

    from mindspeed.mindspore.third_party.safetensors.torch import save_file, load_file
    aspm.register_patch('safetensors.torch.save_file', save_file)
    aspm.register_patch('safetensors.torch.load_file', load_file)

    # accelerate
    from mindspeed.mindspore.third_party.accelerate.extract import extract_model_from_parallel
    aspm.register_patch('accelerate.utils.extract_model_from_parallel', extract_model_from_parallel)

    # transformers
    from mindspeed.mindspore.third_party.transformers.configuration_utils import dict_torch_dtype_to_str
    aspm.register_patch('transformers.configuration_utils.PretrainedConfig.dict_torch_dtype_to_str',
                        dict_torch_dtype_to_str)

    from mindspeed.mindspore.third_party.transformers.modeling_utils import load_state_dict, \
        _load_state_dict_into_meta_model, safe_open, get_parameter_dtype
    aspm.register_patch('transformers.modeling_utils.load_state_dict', load_state_dict)
    aspm.register_patch('transformers.modeling_utils._load_state_dict_into_meta_model',
                        _load_state_dict_into_meta_model)
    aspm.register_patch('transformers.modeling_utils.safe_open', safe_open)
    aspm.register_patch('transformers.modeling_utils.get_parameter_dtype', get_parameter_dtype)

    # swap attention
    from mindspeed.mindspore.core.memory.swap_attention.prefetch import sawp_attention_init, no_swap_tensor, pack_hook, \
        prefetch_register_post_backward_hook
    aspm.register_patch('mindspeed.core.memory.swap_attention.prefetch.SwapTensor.__init__', sawp_attention_init)
    aspm.register_patch('mindspeed.core.memory.swap_attention.prefetch.SwapPrefetch.no_swap_tensor', no_swap_tensor)
    aspm.register_patch('mindspeed.core.memory.swap_attention.prefetch.SwapPrefetch.pack_hook', pack_hook)
    aspm.register_patch('mindspeed.core.memory.swap_attention.prefetch.prefetch_register_post_backward_hook',
                        prefetch_register_post_backward_hook)