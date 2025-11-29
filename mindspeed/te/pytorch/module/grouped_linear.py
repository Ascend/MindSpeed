# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional, Callable, Tuple

import torch
import torch_npu
from torch.nn import Parameter

from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu, _initialize_affine_weight_gpu
from megatron.core.transformer.moe.experts import expert_dist_ckpt_decorator
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core import parallel_state
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import condition_init_method
from megatron.core.parallel_state import (
    get_expert_model_parallel_world_size,
    get_expert_model_parallel_rank,
    get_expert_data_parallel_rank,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.transformer.moe.grouped_gemm_util import Ops


class MindSpeedTEGroupedLinearGMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor,
                m_split=None,
                group_list_type=None,
                ori_weight=None,
                *weight_input_T) -> torch.Tensor:
        
        # Due to ascend gmm kernal k split limitations, we need a tensor m_split, not a tensor List.
        # Also can be solved in token_dispatcher.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split
        weight_T = weight_input_T
        ctx.group_list_type = group_list_type
        fwd_output = torch_npu.npu_grouped_matmul([input_tensor], weight_T, bias=None, group_list=ctx.group_list,
                                                  split_item=2, group_type=0, group_list_type=ctx.group_list_type)[0]
        ctx.save_for_backward(input_tensor, *ori_weight)
        return fwd_output

    @staticmethod  
    def backward(ctx, grad_output):

        group_list = ctx.group_list
        inp = ctx.saved_tensors[0]
        weight = ctx.saved_tensors[1:]
        group_list_type = ctx.group_list_type
        grad = torch_npu.npu_grouped_matmul([grad_output], weight, bias=None, group_list=group_list,
                                            split_item=2, group_type=0, group_list_type=group_list_type)[0]
        # K spilt gmm.
        grad_weight = torch_npu.npu_grouped_matmul([inp.T], [grad_output], bias=None, group_list=group_list,
                                    split_item=3, group_type=2, group_list_type=group_list_type)[0]
        
        return grad, None, None, None, *grad_weight


class MindSpeedTEGroupedLinear(torch.nn.Module):
    def __init__(self, num_gemms: int, input_size: int, output_size: int, *, parallel_mode: Optional[str], config,
                 init_method: Callable, bias: bool, skip_bias_add: bool, is_expert: bool = False,
                 tp_comm_buffer_name: Optional[str] = None, **kwargs):
        super().__init__()
        self.num_gemms = num_gemms
        self.config = config
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.use_bias = bias
        self.output_size = output_size
        self.input_size = input_size
        self.partition_dim = 1 if parallel_mode == "column" else 0
        self.parallel_mode = parallel_mode

        if is_expert:
            tp_group = get_expert_tensor_parallel_group(check_initialized=False)
            tp_size = get_expert_tensor_parallel_world_size()
        else:
            tp_group = get_tensor_model_parallel_group(check_initialized=False)
            tp_size = get_tensor_model_parallel_world_size()

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                if output_size % tp_size != 0:
                    raise AssertionError("{} is not divisible by {}".format(output_size, tp_size))
                self.output_size = output_size // tp_size
                self.input_size = input_size
            elif parallel_mode == "row":
                if input_size % tp_size != 0:
                    raise AssertionError("{} is not divisible by {}".format(input_size, tp_size))
                self.output_size = output_size
                self.input_size = input_size // tp_size
            self.tp_size = 1
            self.tp_group = None

        self.total_weight = []

        for i in range(self.num_gemms):
            expert_weight = Parameter(torch.empty(self.output_size, self.input_size,
                                                  device=torch.device('cpu') if self.config.use_cpu_initialization else torch.npu.current_device(),
                                                  dtype=config.params_dtype))
            self.register_parameter('weight{}'.format(i), expert_weight)
            if self.config.perform_initialization:
                if self.config.use_cpu_initialization:
                    _initialize_affine_weight_cpu(expert_weight, output_size, input_size,
                                                  self.output_size if parallel_mode == "column" else self.input_size,
                                                  partition_dim=self.partition_dim,
                                                  init_method=init_method,
                                                  stride=1,
                                                  rank=torch.distributed.get_rank(tp_group),
                                                  world_size=tp_size)
                else:
                    _initialize_affine_weight_gpu(expert_weight, init_method, partition_dim=self.partition_dim, stride=1, is_expert=is_expert)
            self.total_weight.append(expert_weight)

        for param in self.parameters():
            setattr(param, 'allreduce', not (is_expert and self.expert_parallel))

    def forward(self, x, m_splits):
        if not getattr(get_args(), "fp8", False):
            group_list_type = 1
            for w in self.total_weight:
                if self.parallel_mode == 'column':
                    w = w.view(self.config.hidden_size, -1)
                else:
                    w = w.view(-1, self.config.hidden_size)
            self.total_weight_T = [w.T for w in self.total_weight]
            output = MindSpeedTEGroupedLinearGMM.apply(x, m_splits, group_list_type, self.total_weight,
                                                       *self.total_weight_T)
        else:
            if self.parallel_mode == 'column':
                weight = [w.view(self.config.hidden_size, -1) for w in self.total_weight]
            else:
                weight = [w.view(-1, self.config.hidden_size) for w in self.total_weight]
            from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_op_cls
            output = get_gmm_op_cls().gmm_apply(x, torch.stack(weight, dim=0), None, m_splits)
        return output, None

    def _sharded_state_dict_grouped(
            self, tp_axis_map, prefix='', sharded_offsets=(), metadata=None
    ):
        """
        prefix should be module_name to make keys identical to sequetial ones.
        """
        sharded_state_dict = {}
        full_state_dict = self.state_dict(prefix='', keep_vars=True)
        num_global_experts = get_expert_model_parallel_world_size() * self.num_gemms
        local_expert_indices_offset = get_expert_model_parallel_rank() * self.num_gemms
        ep_axis = len(sharded_offsets)
        for gemm_idx in range(self.num_gemms):
            state_dict = {
                f'{gemm_idx}.weight': full_state_dict[f'weight{gemm_idx}'],
            }
            if self.use_bias:
                state_dict[f'{gemm_idx}.bias'] = full_state_dict[f'bias{gemm_idx}']
            sub_sd = make_sharded_tensors_for_checkpoint(
                state_dict,
                '',
                tp_axis_map,
                (
                    *sharded_offsets,
                    (ep_axis, local_expert_indices_offset + gemm_idx, num_global_experts),
                ),
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(sub_sd, f'{gemm_idx}.', prefix)
            sharded_state_dict.update({f'{prefix}weight{gemm_idx}': sub_sd[f'{gemm_idx}.weight']})
            if self.use_bias:
                sharded_state_dict[f'{prefix}bias{gemm_idx}'] = sub_sd[f'{gemm_idx}.bias']
        # Adjust replica ids - replication along DP modulo EP
        for k, sh_ten in sharded_state_dict.items():
            replica_id = sh_ten.replica_id
            assert (
                    len(replica_id) == 3
            ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
            if getattr(sh_ten, "is_data_parallel_fully_shard", False):
                edp_replica_id = 0
            else:
                edp_replica_id = get_expert_data_parallel_rank()
            sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
        return sharded_state_dict

    @expert_dist_ckpt_decorator
    def sharded_state_dict(
            self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ):
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(module, f'{name}.', sharded_offsets, metadata)
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = (
                        parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
                )
                local_expert_indices_offset = (
                        parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
                )
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + i, num_global_experts),
                    )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(sub_sd[k], new_sharded_offsets)
            # Add prefix here to match sequential's keys
            replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict


class MindSpeedTEColumnParallelGroupedLinear(MindSpeedTEGroupedLinear):
    """
    Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
    to column-parallel style.
    """

    def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: Optional[str] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        For each gemm, sharding along axis 0, bias sharded.
        Assume sharded_offsets[-1] is the expert parallel offset.
        """
        tp_axis_map = {}
        for gemm_idx in range(self.num_gemms):
            tp_axis_map.update({f'{gemm_idx}.weight': 0, f'{gemm_idx}.bias': 0})
        return super()._sharded_state_dict_grouped(
            tp_axis_map, prefix, sharded_offsets, metadata
        )


class MindSpeedTERowParallelGroupedLinear(MindSpeedTEGroupedLinear):
    """
    Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
    to row-parallel style.
    """

    def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: Optional[str] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        For each gemm, sharding along axis 1, bias not sharded.
        Assume sharded_offsets[-1] is the expert parallel offset.
        """
        tp_axis_map = {f'{gemm_idx}.weight': 1 for gemm_idx in range(self.num_gemms)}
        return super()._sharded_state_dict_grouped(
            tp_axis_map, prefix, sharded_offsets, metadata
        )
    

def mindspeed_groupedmlp_weighted_bias_swiglu_impl(x, bias, probs, fp8_input_store=False):
    """Patch of TEGroupedMLP with MindSpeed.
    Use ascend fused_swiglu instead weighted_bias_swiglu_impl for better performance.
    """
    from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
    if bias is not None:
        raise NotImplementedError("Bias is not support for weighted swiglu fusion.")
    dtype = x.dtype
    res = fused_swiglu(x) * probs
    return res.to(dtype)
