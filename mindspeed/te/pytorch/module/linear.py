# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Any, Callable, Optional

import torch
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size, get_expert_tensor_and_model_parallel_group,
)
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu, _initialize_affine_weight_gpu, \
    set_tensor_model_parallel_attributes
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import divide
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.te.pytorch.fp8 import MatmulKey, fp8_matmul, fp8_matmul_add, is_fp8_tensor_2d
from mindspeed.te.pytorch.fp8.metadata import FP8Metadata
from mindspeed.te.pytorch.module.ops import DummyHandle, get_ops
from mindspeed.te.pytorch.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG


class TEColumnParallelLinear(torch.nn.Module):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        super(TEColumnParallelLinear, self).__init__()
        self.fp8_meta = FP8Metadata()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.config = config
        self.skip_weight_param_allocation = skip_weight_param_allocation

        if is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
            tp_group = get_expert_tensor_and_model_parallel_group()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
            tp_group = get_tensor_model_parallel_group()

        self.fp8_meta.set_tp_config(world_size, rank, tp_group)

        self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

        self.output_size_per_partition = divide(output_size, world_size)

        # Initialize weight.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=world_size,
                    )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.weight = None

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter('bias', None)

        self.sequence_parallel = config.sequence_parallel and world_size > 1
        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward"
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)},"
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None
        if self.explicit_expert_comm and self.fp8_meta.fp8_enable:
            from mindspeed.te.pytorch.fp8.recipes import matmul_fp8
            output = matmul_fp8(input_, weight)
        elif self.explicit_expert_comm:
            output = input_.matmul(weight.t())
        elif self.sequence_parallel:
            output = ColumnParallelSeq.apply(input_, weight, bias, self.fp8_meta)
        else:
            output = ColumnParallelNoSeq.apply(input_, weight, bias, self.fp8_meta)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None


class ColumnParallelSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, fp8_meta: FP8Metadata):
        ctx.use_bias = bias is not None
        ctx.fp8_meta = fp8_meta
        ctx.fp8_enable = fp8_meta.is_fp8_enable()
        ctx.total_input = None
        ctx.gradient_accumulation_fusion = get_args().gradient_accumulation_fusion

        output_parallel, total_input, weight_fp8 = get_ops().allgather_matmul(input_, weight, None,
                                                                              fp8_meta, MatmulKey.forward,
                                                                              ctx.fp8_enable)
        if COMM_OVERLAP_CONFIG.save_allgather_input:
            ctx.total_input = total_input

        if ctx.fp8_enable:
            BackwardStateStorage.save(ctx, None, weight_fp8, weight)
        else:
            BackwardStateStorage.save(ctx, input_, weight, weight)
        ctx.input_size = input_.size()

        return output_parallel

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        fp8_meta: FP8Metadata = ctx.fp8_meta
        fp8_enable = ctx.fp8_enable
        input_size = ctx.input_size
        tp_group = get_tensor_model_parallel_group()
        ori_grad = grad_output

        input_, weight, weight_param = BackwardStateStorage.load(ctx)
        all_gather_handle, total_input = DummyHandle, ctx.total_input
        if ctx.needs_input_grad[1] and not COMM_OVERLAP_CONFIG.save_allgather_input:
            # It won't touch this branch temporarily, it's not suitable for now
            pass
        if not fp8_enable:
            grad_input = grad_output.matmul(weight)
            sub_grad_input = torch.empty(input_.size(), dtype=input_.dtype, device=input_.device, requires_grad=False)
        else:
            grad_input, grad_output, _ = fp8_matmul(grad_output, weight, fp8_meta, MatmulKey.dx)
            # After enabling FP8, total_input is saved here instead of input_ due to lack of FP8 communication
            sub_grad_input = torch.empty(input_size, dtype=total_input.dtype, device=total_input.device,
                                         requires_grad=False)

        reduce_scatter_handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input, group=tp_group,
                                                                       async_op=True)

        all_gather_handle.wait()
        grad_weight, grad_bias = calculate_grad(ctx, total_input, weight_param, grad_output,
                                                ori_grad)

        reduce_scatter_handle.wait()
        return sub_grad_input, grad_weight, grad_bias, None


class ColumnParallelNoSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, fp8_meta: FP8Metadata):
        ctx.use_bias = bias is not None
        ctx.fp8_meta = fp8_meta
        ctx.fp8_enable = fp8_meta.is_fp8_enable()
        ctx.gradient_accumulation_fusion = get_args().gradient_accumulation_fusion
        if fp8_meta is None or not fp8_meta.is_fp8_enable():
            output = torch.matmul(input_, weight.t())
            BackwardStateStorage.save(ctx, input_, weight, weight)
        else:
            output, fp8_input, fp8_weight = fp8_matmul(input_, weight, fp8_meta, MatmulKey.forward)
            BackwardStateStorage.save(ctx, fp8_input, fp8_weight, weight)

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        tp_group = get_tensor_model_parallel_group()
        tp_world_size = get_tensor_model_parallel_world_size()
        ori_grad = grad_output
        input_, weight, weight_param = BackwardStateStorage.load(ctx)

        if not ctx.fp8_enable:
            grad_input = grad_output.matmul(weight)
        else:
            grad_input, grad_output, _ = fp8_matmul(grad_output, weight, ctx.fp8_meta, MatmulKey.dx)

        # Handle allreduce for zero-shape input to avoid issues when TP=1, will be removed later
        handle = DummyHandle
        if tp_world_size > 1:
            handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)

        grad_weight, grad_bias = calculate_grad(ctx, input_, weight_param, grad_output, ori_grad)
        # Wait after computation
        handle.wait()

        return grad_input, grad_weight, grad_bias, None


class TERowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

        super(TERowParallelLinear, self).__init__()
        self.fp8_meta = FP8Metadata()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.skip_bias_add = skip_bias_add
        self.sequence_parallel = config.sequence_parallel and config.tensor_model_parallel_size > 1

        # Divide the weight matrix along the last dimension.
        if self.is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
            tp_group = get_expert_tensor_and_model_parallel_group()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
            tp_group = get_tensor_model_parallel_group()

        self.fp8_meta.set_tp_config(world_size, rank, tp_group)
        self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

        self.input_size_per_partition = divide(input_size, world_size)

        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    is_expert=self.is_expert,
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.register_parameter('bias', None)

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

    def forward(self, input_: torch.Tensor):
        if self.explicit_expert_comm and self.fp8_meta.fp8_enable:
            from mindspeed.te.pytorch.fp8.recipes import matmul_fp8
            output = matmul_fp8(input_, self.weight)
        elif self.explicit_expert_comm:
            output = input_.matmul(self.weight.t())
        elif self.sequence_parallel:
            output = RowParallelSeq.apply(input_, self.weight, None, self.fp8_meta)
        else:
            output = RowParallelNoSeq.apply(input_, self.weight, None, self.fp8_meta)

        if not self.skip_bias_add:
            output = (output + self.bias) if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias

        return output, output_bias

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None


class RowParallelSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, fp8_meta: FP8Metadata):
        ctx.use_bias = bias is not None
        ctx.fp8_meta = fp8_meta
        ctx.fp8_enable = fp8_meta.is_fp8_enable()
        ctx.gradient_accumulation_fusion = get_args().gradient_accumulation_fusion
        output_parallel, input_, _weight = get_ops().matmul_reduce_scatter(
            input_, weight, bias, fp8_meta, MatmulKey.forward, ctx.fp8_enable
        )
        BackwardStateStorage.save(ctx, input_, _weight, weight)

        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        ori_grad = grad_output
        input_, weight, weight_param = BackwardStateStorage.load(ctx)

        grad_input, grad_output, _ = get_ops().allgather_matmul(
            grad_output, weight, None, ctx.fp8_meta, MatmulKey.dx, ctx.fp8_enable
        )
        grad_weight, grad_bias = calculate_grad(ctx, input_, weight_param, grad_output, ori_grad)
        return grad_input, grad_weight, grad_bias, None


class RowParallelNoSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, fp8_meta: FP8Metadata):
        ctx.use_bias = bias is not None
        ctx.fp8_meta = fp8_meta
        ctx.fp8_enable = fp8_meta.is_fp8_enable()
        ctx.gradient_accumulation_fusion = get_args().gradient_accumulation_fusion

        output_, _input, _weight = get_ops().matmul_all_reduce(input_, weight, bias, fp8_meta,
                                                               MatmulKey.forward, ctx.fp8_enable)
        BackwardStateStorage.save(ctx, _input, _weight, weight)
        return output_

    @staticmethod
    def backward(ctx, grad_output):
        ori_grad = grad_output
        input_, weight, weight_param = BackwardStateStorage.load(ctx)
        if not ctx.fp8_enable:
            grad_input = grad_output.matmul(weight)
        else:
            grad_input, grad_output, _ = fp8_matmul(grad_output, weight, ctx.fp8_meta, MatmulKey.dx)

        grad_weight, grad_bias = calculate_grad(ctx, input_, weight_param, grad_output, ori_grad)
        return grad_input, grad_weight, grad_bias, None


def async_gather_along_first_dim(input_, group, world_size):
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output_ = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.npu.current_device(), requires_grad=False
    )
    work = torch.distributed._all_gather_base(
        output_, input_.contiguous(), group=group, async_op=True
    )
    return work, output_


class BackwardStateStorage:
    """
    Manages state storage from forward to backward pass in Autograd Functions.
    
    Handles three modes:
    1. FP8 mode: Store quantized input/weight and original weight parameter
    2. Gradient Accumulation Fusion mode: Save input only, store weight as attribute
    3. Normal mode: Save input and weight via save_for_backward
    """

    @staticmethod
    def save(ctx, input_: torch.Tensor,
             weight: torch.Tensor, weight_param: Optional[torch.Tensor] = None) -> None:
        """Save input and weight from forward pass for backward pass."""
        if ctx.fp8_enable:
            # FP8 mode: quantized tensors don't need grad tracking, store as attributes
            ctx.input_fp8 = input_
            ctx.weight_fp8 = weight
            ctx.weight_param = weight_param
        elif ctx.gradient_accumulation_fusion:
            # Fusion mode: save input only, store weight as attribute
            ctx.save_for_backward(input_)
            ctx.weight = weight
            ctx.weight_param = weight_param
        else:
            # Normal mode: save input and weight
            ctx.save_for_backward(input_, weight)
            ctx.weight_param = weight_param

    @staticmethod
    def load(ctx) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Load input and weight saved from forward pass in backward pass."""
        if ctx.fp8_enable:
            return ctx.input_fp8, ctx.weight_fp8, ctx.weight_param
        elif ctx.gradient_accumulation_fusion:
            return ctx.saved_tensors[0], ctx.weight, ctx.weight_param
        else:
            input_, weight = ctx.saved_tensors
            return input_, weight, ctx.weight_param


def _calculate_grad_weight(ctx, inp: torch.Tensor, weight_param: torch.Tensor,
                           grad: torch.Tensor) -> Optional[torch.Tensor]:
    """Calculate gradient of weight."""
    grad, total_input = reshape_to_2D(grad), reshape_to_2D(inp)

    if ctx.gradient_accumulation_fusion and weight_param.main_grad.dtype != torch.float32:
        raise RuntimeError(
            f"Unsupported gradient type ({weight_param.main_grad.dtype}) for gradient accumulation fusion, "
            f"expected type is float32"
        )

    if ctx.fp8_enable:
        from mindspeed.te.pytorch.fp8.recipes import MXFP8BlockScaling
        if ctx.gradient_accumulation_fusion and isinstance(ctx.fp8_meta.fp8_recipe, MXFP8BlockScaling):
            fp8_matmul_add(weight_param.main_grad, grad, total_input, ctx.fp8_meta)
            return _create_grad_weight_placeholder(weight_param)
        else:
            grad_weight, _, _ = fp8_matmul(grad, total_input, ctx.fp8_meta, MatmulKey.dw)
            return grad_weight
    elif ctx.gradient_accumulation_fusion:
        from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
        npu_matmul_add_fp32(total_input, grad, weight_param.main_grad)
        return _create_grad_weight_placeholder(weight_param)
    else:
        return grad.t().matmul(total_input)


def _calculate_grad_bias(ctx, grad: torch.Tensor, ori_grad: torch.Tensor) -> Optional[torch.Tensor]:
    """Calculate gradient of bias."""
    if not ctx.use_bias:
        return None
    if ctx.fp8_enable:
        grad = reshape_to_2D(ori_grad)
    return grad.sum(dim=0)


def _create_grad_weight_placeholder(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """
        # When overlap_grad_reduce is True, need to ensure that backward hooks
        # are all run on the main backprop thread to prevent deadlocks. Setup
        # dummy grad_weight tensor to prevent backward hooks from being run
        # in a background thread.
    """
    if not hasattr(weight, 'grad_added_to_main_grad'):
        return None

    if getattr(weight, 'zero_out_wgrad', False):
        grad_weight = torch.zeros(
            weight.main_grad.shape,
            dtype=weight.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
    else:
        grad_weight = torch.empty(
            weight.main_grad.shape,
            dtype=weight.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
    weight.grad_added_to_main_grad = True
    return grad_weight


def calculate_grad(ctx, inp: torch.Tensor, weight_param: torch.Tensor,
                   grad: torch.Tensor, ori_grad: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculate gradients of weight and bias.
    
    Args:
        ctx: Autograd context
        inp: Input tensor (may be FP8 quantized)
        weight_param: Original weight parameter (for gradient accumulation)
        grad: Output gradient (may be FP8 quantized)
        ori_grad: Original output gradient (for bias calculation)
    
    Returns:
        (grad_weight, grad_bias)
    """
    _, is_grad_weight_needed, is_grad_bias_needed, _ = ctx.needs_input_grad

    grad_weight = None
    if is_grad_weight_needed:
        grad_weight = _calculate_grad_weight(ctx, inp, weight_param, grad)

    grad_bias = None
    if is_grad_bias_needed:
        grad_bias = _calculate_grad_bias(ctx, grad, ori_grad)

    return grad_weight, grad_bias


def reshape_to_2D(input_tensor: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D for execution compatibility."""
    if is_fp8_tensor_2d(input_tensor):
        return input_tensor
    return input_tensor.reshape(-1, input_tensor.shape[-1])
