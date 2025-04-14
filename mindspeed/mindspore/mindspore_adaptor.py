#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import contextlib
import functools
import types
import sys

import mindtorch
import mindtorch.torch as torch
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
            grad_norm = mindtorch.torch.norm(grad, norm_type)
            total_grad_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_grad_norm.clone())
    # if not list
    if not tensor_lists:
        grad_norm = mindtorch.torch.cuda.FloatTensor([0])
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
    with mindtorch.torch.no_grad():
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
    grad_output = mindtorch.torch.cast_to_adapter_tensor(grad_output)
    if isinstance(grad_output, (list, tuple)):
        res = self.backward(self.ctx, *grad_output)
    else:
        res = self.backward(self.ctx, grad_output)
    res = mindtorch.torch.cast_to_ms_tensor(res)
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
    from mindspeed.mindspore.core.distributed.finalize_model_grads import allreduce_layernorm_grads
    from mindspeed.mindspore.core.optimizer.optimizer import float16_optimizer_init
    from mindspeed.mindspore.core.optimizer.optimizer import float16_optimizer_collect_for_unscaling
    from mindspeed.mindspore.core.optimizer.optimizer import mixed_precision_optimizer_unscale_and_check_for_nan
    from mindspeed.mindspore.core.optimizer.clip_grads import clip_grad_norm_fp32
    from mindspeed.mindspore.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_backward
    from mindspeed.mindspore.core.utils import make_viewless_tensor
    from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_no_pipelining
    from mindspeed.mindspore.core.distributed.distributed_data_parallel import distributed_data_parallel_init
    from mindspeed.mindspore.core.distributed.param_and_grad_buffer import get
    from mindspeed.mindspore.core.tensor_parallel.mappings import mapping_reduce
    from mindspeed.mindspore.core.tensor_parallel.mappings import reduce_from_model_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.mappings import gather_from_model_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.mappings import copy_to_model_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.mappings import scatter_to_model_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.mappings import gather_from_sequence_parallel_region_bprop
    from mindspeed.mindspore.core.tensor_parallel.cross_entropy import bocab_parallel_cross_entropy_bprop
    from mindspeed.mindspore.core.tensor_parallel.layers import linear_with_frozen_weight_bprop
    from mindspeed.mindspore.core.tensor_parallel.layers import \
        linear_with_grad_accumulation_and_async_communication_bprop

    mspm.register_patch('megatron.core.tensor_parallel.mappings._reduce', mapping_reduce)
    mspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                        distributed_data_parallel_init)
    mspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining',
                        forward_backward_no_pipelining)
    mspm.register_patch('megatron.core.utils.make_viewless_tensor', make_viewless_tensor)
    mspm.register_patch('megatron.core.distributed.finalize_model_grads._allreduce_layernorm_grads',
                        allreduce_layernorm_grads)
    mspm.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                        float16_optimizer_init)
    mspm.register_patch(
        'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._collect_main_grad_data_for_unscaling',
        float16_optimizer_collect_for_unscaling)
    mspm.register_patch(
        'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan',
        mixed_precision_optimizer_unscale_and_check_for_nan)
    mspm.register_patch('megatron.core.optimizer.clip_grads.clip_grad_norm_fp32', clip_grad_norm_fp32)
    mspm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer._get', get)
    mspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.backward',
                        vocab_parallel_cross_entropy_backward)

    mspm.register_patch('megatron.core.tensor_parallel.mappings._ReduceFromModelParallelRegion.bprop',
                        reduce_from_model_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.mappings._ScatterToSequenceParallelRegion.bprop',
                        scatter_to_sequence_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.mappings._ReduceScatterToSequenceParallelRegion.bprop',
                        reduce_scatter_to_sequence_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.bprop',
                        bocab_parallel_cross_entropy_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.mappings._GatherFromModelParallelRegion.bprop',
                        gather_from_model_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.mappings._CopyToModelParallelRegion.bprop',
                        copy_to_model_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.mappings._ScatterToModelParallelRegion.bprop',
                        scatter_to_model_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.mappings._GatherFromSequenceParallelRegion.bprop',
                        gather_from_sequence_parallel_region_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithFrozenWeight.bprop',
                        linear_with_frozen_weight_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.bprop',
                        linear_with_grad_accumulation_and_async_communication_bprop, create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.random._CUDA_RNG_STATE_TRACKER', DummyTracker(),
                        create_dummy=True)
    mspm.register_patch('megatron.core.tensor_parallel.random.CudaRNGStatesTracker', DummyTracker,
                        create_dummy=True)


def exe_adaptation():
    print('=====Enter Mindspore adaptor =====')
    from mindspeed.patch_utils import MindSpeedPatchesManager as mspm
    apex_adaptation(mspm)
    te_adaptation(mspm)
    mspm.apply_patches()
    megatron_torch_adaptation(mspm)
    megatron_training_adaptation(mspm)
    megatron_core_adaptation(mspm)
    mspm.apply_patches()


exe_adaptation()
