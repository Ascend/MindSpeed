#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import contextlib
import functools
import types
import sys

import torch
import mindspore
from mindspore import context

context.set_context(deterministic="ON")

_GRAD_FN = None
mindspore.set_context(pynative_synchronize=True)


def type_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, str):
            out = out.replace('torch', 'torch.cuda')
        return out

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    """calc grad norm"""
    total_grad_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_grad_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_grad_norm.clone())
    # if not list
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_grad_norm = grad_norm ** norm_type
    # norm_type can not zero
    if norm_type != 0:
        return total_grad_norm ** (1 / norm_type), ret_per_tensor
    return total_grad_norm


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'.format(len(tensor_lists[0]),
                                                                                              len(tensor_lists[1])))
    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def _lazy_call(callable, **kwargs):
    callable()


def dummy_function(*args, **kwargs):
    pass


def dummy_return(res):
    @functools.wraps(res)
    def warpper(*args, **kwargs):
        return res

    return warpper


class DummyTracker:

    @contextlib.contextmanager
    def fork(self, *args, **kwargs):
        yield

    def reset(self):
        ...

    def get_states(self):
        return None

    def add(self, name, seed):
        ...


def dummy_decorate(fn):
    return fn


def _custom_fwd(fwd=None, *, cast_inputs=None):
    return fwd


def _custom_bwd(bwd):
    return bwd


def bprop_commn(self, grad_output):
    grad_output = torch.cast_to_adapter_tensor(grad_output)
    if isinstance(grad_output, (list, tuple)):
        res = self.backward(self.ctx, *grad_output)
    else:
        res = self.backward(self.ctx, grad_output)
    res = torch.cast_to_ms_tensor(res)
    if res is None:
        return 0
    elif isinstance(res, (list, tuple)):
        return tuple([0 if x is None else x for x in res])
    return res


def fused_layer_norm_affine(input_, weight, bias, normalized_shape, eps):
    return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)


def apex_adaptation(mspm):
    import math
    sys.modules['apex'] = types.ModuleType('apex')
    mspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
    mspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
    mspm.register_patch('fused_layer_norm_cuda', create_dummy=True)
    mspm.register_patch('apex.optimizers.FusedSGD', torch.optim.SGD, create_dummy=True)
    mspm.register_patch('apex.optimizers.FusedAdam', torch.optim._adamw.Float32AdamW, create_dummy=True)
    mspm.register_patch('apex.__spec__', math.__spec__, create_dummy=True)
    mspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)


def te_adaptation(mspm):
    # Need replace modules before import megatron
    mspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
    mspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
    mspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
    mspm.register_patch('transformer_engine.common.recipe.DelayedScaling', torch.nn.Module, create_dummy=True)
    mspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)


def megatron_torch_adaptation(mspm):
    torch.cuda.amp.custom_fwd = dummy_decorate
    torch.cuda.amp.custom_bwd = dummy_decorate
    torch.preserve_format = None
    torch.Tensor.type = type_wrapper(torch.Tensor.type)
    torch.nn.parameter.Parameter.type = type_wrapper(torch.nn.parameter.Parameter.type)


def megatron_training_adaptation(mspm):
    from mindspeed.arguments import parse_args_wrapper, validate_args_wrapper
    from mindspeed.mindspore.training.initialize import _initialize_distributed, _compile_dependencies
    from mindspeed.mindspore.model.transformer import parallel_transformer_forward_wrapper
    from mindspeed.yaml_arguments import print_args_wrapper
    mspm.register_patch('megatron.training.initialize.parse_args', parse_args_wrapper)
    mspm.register_patch('megatron.training.initialize.validate_args', validate_args_wrapper)
    mspm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
    mspm.register_patch('megatron.training.initialize.set_jit_fusion_options', dummy_function)
    mspm.register_patch('megatron.training.utils.report_memory', dummy_function)
    mspm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
    mspm.register_patch('megatron.training.arguments._print_args', print_args_wrapper)
    mspm.register_patch('megatron.training.initialize._initialize_distributed', _initialize_distributed)
    mspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.forward',
                        parallel_transformer_forward_wrapper)


def megatron_core_adaptation(mspm):
    from mindspeed.mindspore.core.pipeline_parallel.schedules import deallocate_output_tensor
    from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_no_pipelining
    from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving
    from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving
    from mindspeed.mindspore.core.distributed.distributed_data_parallel import distributed_data_parallel_init

    from mindspeed.mindspore.core.tensor_parallel.layers import backward
    from mindspeed.mindspore.core.tensor_parallel.random import checkpoint_function_backward
    from mindspeed.mindspore.core.models.gpt.gpt_model import model_forward
    from mindspeed.mindspore.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_bshd
    from mindspeed.mindspore.optimizer.adamw import adamw


    mspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                        distributed_data_parallel_init)
    mspm.register_patch('megatron.core.pipeline_parallel.schedules.deallocate_output_tensor',
                        deallocate_output_tensor)
    mspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining', forward_backward_no_pipelining, force_patch=True)
    mspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving', forward_backward_pipelining_with_interleaving, force_patch=True)
    mspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving', forward_backward_pipelining_without_interleaving, force_patch=True)

    mspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward', backward, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                        checkpoint_function_backward, force_patch=True)
    mspm.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', model_forward, force_patch=True)
    mspm.register_patch('megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd',
                        apply_rotary_pos_emb_bshd, force_patch=True)
    mspm.register_patch('mindspeed.optimizer.adamw.adamw', adamw, force_patch=True)


def auto_settings_adaptation(mspm):
    from mindspeed.mindspore.auto_settings.auto_settings import ms_init, _init_global_group
    from mindspeed.mindspore.auto_settings.module.parse.profiling_parse import ms_get_settings
    from mindspeed.mindspore.auto_settings.module.parse.profiling_parse.profiling_constant import SpecialOperatorName
    from mindspeed.mindspore.auto_settings.module.parse.profiling_parse.profiling_memory_parse import analyse_cann_and_driver, analyse_loss
    from mindspeed.mindspore.auto_settings.profile.profiler import profiler_run
    from mindspeed.mindspore.auto_settings.profile.runner import runner_run
    from mindspeed.mindspore.auto_settings.utils.utils import get_module_info
    from mindspeed.mindspore.auto_settings.config.system_config import __post_init__
    from mindspeed.mindspore.auto_settings.module.time_cost_black import get_module_time

    mspm.register_patch('mindspeed.auto_settings.auto_settings.AutoSettings.init',
                        ms_init)
    mspm.register_patch('mindspeed.auto_settings.auto_settings._init_global_group',
                        _init_global_group)
    mspm.register_patch('mindspeed.auto_settings.module.parse.profiling_parse.get_settings',
                        ms_get_settings)
    mspm.register_patch('mindspeed.auto_settings.module.parse.profiling_parse.profiling_constant.SpecialOperatorName',
                        SpecialOperatorName)
    mspm.register_patch('mindspeed.auto_settings.module.parse.profiling_parse.profiling_memory_parse.AnalyseMemoryMsg.analyse_cann_and_driver',
                        analyse_cann_and_driver)
    mspm.register_patch(
        'mindspeed.auto_settings.module.parse.profiling_parse.profiling_memory_parse.AnalyseMemoryMsg.analyse_loss',
        analyse_loss)
    mspm.register_patch('mindspeed.auto_settings.profile.profiler.Profiler.run',
                        profiler_run)
    mspm.register_patch('mindspeed.auto_settings.profile.runner.Runner.run',
                        runner_run)
    mspm.register_patch('mindspeed.auto_settings.utils.utils.get_module_info',
                        get_module_info)
    mspm.register_patch('mindspeed.auto_settings.config.system_config.SystemConfig.__post_init__',
                        __post_init__)
    mspm.register_patch('mindspeed.auto_settings.module.time_cost_black.TimeCostBlack.get_module_time',
                        get_module_time)


def exe_mindspore_adaptation(mindspeed_args):
    print('=====Enter Mindspore adaptor =====')
    if not hasattr(mindspeed_args, "ai_framework") or mindspeed_args.ai_framework != "mindspore" or mindspeed_args.optimization_level < 0:
        return
    from mindspeed.patch_utils import MindSpeedPatchesManager as mspm
    megatron_core_adaptation(mspm)
    auto_settings_adaptation(mspm)
    mspm.apply_patches()

