# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

def mindspore_adaptation(aspm, mindspeed_args):
    if not hasattr(mindspeed_args, "ai_framework") or mindspeed_args.ai_framework != "mindspore" or mindspeed_args.optimization_level < 0:
        return

    from .core.fusions.npu_moe_token_permute import permute_wrapper
    aspm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper, force_patch=True)
    from .optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
    aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                        reuse_fp32_param_distrib_optimizer_init_wrapper, force_patch=True)
    from .core.distributed.distributed_data_parallel import distributed_data_parallel_init, \
        _make_backward_post_hook
    aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                        distributed_data_parallel_init, force_patch=True)
    aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
                        _make_backward_post_hook)

    from .core.distributed.param_and_grad_buffer import register_grad_ready
    aspm.register_patch('megatron.core.distributed.param_and_grad_buffer.register_grad_ready',
                        register_grad_ready)  # name overlap

    from .core.models.common.embeddings.rotary_pos_embedding import get_rotary_seq_len, local_rotate_half  # resolve warning
    aspm.register_patch(
        'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
        get_rotary_seq_len)  # name overlap
    aspm.register_patch('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

    from .core.optimizer.optimizer import megatron_optimizer_init
    aspm.register_patch('megatron.core.optimizer.optimizer.MegatronOptimizer.__init__', megatron_optimizer_init)

    from .core.pipeline_parallel.schedules import forward_step, backward_step, forward_backward_no_pipelining
    from .core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving, \
        forward_backward_pipelining_without_interleaving, deallocate_output_tensor_  # NOTE there's import patch
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step', forward_step, force_patch=True)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.backward_step', backward_step, force_patch=True)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining',
                        forward_backward_no_pipelining)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                        forward_backward_pipelining_with_interleaving)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                        forward_backward_pipelining_without_interleaving)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.deallocate_output_tensor',
                        deallocate_output_tensor_)

    from .core.tensor_parallel.data import local_build_key_size_numel_dictionaries  # resolve error
    aspm.register_patch('megatron.core.tensor_parallel.data._build_key_size_numel_dictionaries',
                        local_build_key_size_numel_dictionaries)

    from .core.tensor_parallel.random import local_set_cuda_rng_state, checkpoint_function_forward, \
        checkpoint_function_backward
    aspm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', local_set_cuda_rng_state,
                        force_patch=True)
    aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.forward', checkpoint_function_forward)
    aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                        checkpoint_function_backward, force_patch=True)

    from .core.timers import _get_global_min_max_time
    aspm.register_patch('megatron.core.timers.Timers._get_global_min_max_time', _get_global_min_max_time)

    from .core.tensor_parallel.random import CheckpointWithoutOutput, CheckpointFunctionWithoutOutput
    aspm.register_patch('mindspeed.core.tensor_parallel.random.CheckpointWithoutOutput', CheckpointWithoutOutput)
    aspm.register_patch('mindspeed.core.tensor_parallel.random.CheckpointFunctionWithoutOutput',
                        CheckpointFunctionWithoutOutput)

    from mindspeed.mindspore.ops.npu_matmul_add import npu_matmul_add_fp32
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, force_patch=True)
    aspm.register_patch('mindspeed.ops.npu_matmul_add.npu_matmul_add_fp32', npu_matmul_add_fp32)

    from mindspeed.mindspore.core.optimizer.adamw import step_func
    aspm.register_patch('apex.optimizers.FusedAdam.step', step_func)