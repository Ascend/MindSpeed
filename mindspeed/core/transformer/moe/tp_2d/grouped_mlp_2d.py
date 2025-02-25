# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from copy import deepcopy
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint
from megatron.core.utils import divide
from megatron.training import get_args
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
from mindspeed.core.tensor_parallel.layers import _initialize_affine_weight_cpu_2d
from mindspeed.core.tensor_parallel.comm_utils import _split_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_reduce_scatter_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_last_dim
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.ops.gmm import GMMFunction


G_FORWARD_PADDING_SIZE = 0
G_BACKWARD_PADDING_SIZE = 0


class GroupedMLP2D(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
                config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            self.activation_func = fused_swiglu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        self.moe_extended_tp = config.moe_extended_tp

        self.config = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert config.add_bias_linear is False, (
            "bias in the expert layer is not supported in Grouped GEMM yet, "
            "please set '--disable-bias-linear' instead."
        )

        self.init_paras()

        def remove_extra_states_check(self, incompatible_keys):
            """
            Remove _extra_state from unexpected keys.
            These keys are for dist ckpt compatibility with SequentialMLP.
            """
            keys = deepcopy(incompatible_keys.unexpected_keys)
            for key in keys:
                if "_extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def init_paras(self):
        config = self.config
        # How many feature each rank holds for fc1.
        all_local_expert_fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        expert_fc1_output_size = self.config.ffn_hidden_size
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            all_local_expert_fc1_output_size *= 2
            expert_fc1_output_size *= 2

        tpx_comm_world_sz = TPXCollectiveComm.get_comm_group_world_size()
        tpy_comm_world_sz = TPYCollectiveComm.get_comm_group_world_size()
        assert self.config.hidden_size % tpy_comm_world_sz == 0, (
            "fc1 input size should be " "divisible by tp-y"
        )
        assert (
                all_local_expert_fc1_output_size % tpx_comm_world_sz == 0
        ), "fc1 output size should be divisible by tp-x"
        # h/y
        # 2e*dff_h/x
        all_local_experts_fc1_output_size_per_partition = divide(
            all_local_expert_fc1_output_size, tpx_comm_world_sz
        )
        # How many feature each rank holds for fc2.
        all_local_experts_fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        assert (
                all_local_experts_fc2_input_size % tpx_comm_world_sz == 0
        ), "all local expert fc2 output size should be divisible by tp-y"
        assert self.config.hidden_size % tpy_comm_world_sz == 0, (
            "fc2 input size should be " "divisible by tp-x"
        )
        # e*dff_h/x
        all_local_experts_fc2_input_size_per_partition = divide(
            all_local_experts_fc2_input_size, tpx_comm_world_sz
        )
        # h/y
        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            w1s = []  # e1: splited_w1, e2: splited_w1 ..
            w2s = []  # e1: splited_w2, e2: splited_w2 ..
            master_w1s = []
            master_w2s = []
            for idx in range(self.num_local_experts):
                # [h/y, 2*dff_h/x]
                w1 = Parameter(
                    torch.empty(
                        self.config.hidden_size // tpy_comm_world_sz,
                        expert_fc1_output_size // tpx_comm_world_sz,
                        dtype=config.params_dtype,
                    )
                )

                master_w1 = _initialize_affine_weight_cpu_2d(w1, 1, return_master_weight=True, config=self.config)
                w1s.append(w1)
                master_w1s.append(master_w1)
                # [dff_h/x, h/y]
                w2 = Parameter(
                    torch.empty(
                        self.config.ffn_hidden_size // tpx_comm_world_sz,
                        self.config.hidden_size // tpy_comm_world_sz,
                        dtype=config.params_dtype,
                    )
                )
                master_w2 = _initialize_affine_weight_cpu_2d(w2, 0, return_master_weight=True, config=self.config)
                w2s.append(w2)
                master_w2s.append(master_w2)

            self.master_weight1 = Parameter(torch.cat(master_w1s, dim=-1).contiguous().npu())
            self.master_weight2 = Parameter(torch.cat(master_w2s, dim=0).contiguous().npu())
            # [h/y, e*2*dff_h/x]
            self.weight1 = Parameter(torch.cat(w1s, dim=-1).contiguous().npu())
            # [e*dff_h/x, h/y]
            self.weight2 = Parameter(torch.cat(w2s, dim=0).contiguous().npu())
        else:
            # [h/y, 2e*dff_h/x]
            self.weight1 = Parameter(
                torch.empty(
                    divide(self.config.hidden_size, tpy_comm_world_sz),
                    all_local_experts_fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            # [e*dff_h/x, h/y]
            self.weight2 = Parameter(
                torch.empty(
                    all_local_experts_fc2_input_size_per_partition,
                    divide(self.config.hidden_size, tpy_comm_world_sz),
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )

        setattr(self.weight1, "allreduce", not self.expert_parallel)
        setattr(self.weight2, "allreduce", not self.expert_parallel)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        grouped_mlp_paras = dict()
        grouped_mlp_paras['tokens_per_expert'] = tokens_per_expert
        grouped_mlp_paras['hidden_size'] = self.config.hidden_size
        grouped_mlp_paras['num_local_experts'] = self.num_local_experts
        grouped_mlp_paras['gemm_fusion'] = get_args().gemm_gradient_accumulation_fusion
        grouped_mlp_paras['tp_y'] = get_args().tp_y

        # [n, h] -> [n1/y, 2e*dff_h/x]
        fc1_output = CustomGMM2DFC1.apply(permuted_local_hidden_states, self.weight1, grouped_mlp_paras)

        # [n1/y, 2e*dff_h/x] -> [n1/y, e*dff_h/x]
        intermediate_parallel = self.activation_func(fc1_output)

        # [n1/y, e*dff_h/x] -> [n, h]  partial-x
        fc2_output = CustomGMM2DFC2.apply(intermediate_parallel, self.weight2, grouped_mlp_paras)

        return fc2_output, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        if self.moe_extended_tp:
            raise NotImplementedError(
                "Currently distributed checkpointing is not supported for moe_extended_tp"
            )

        sharded_state_dict = {}
        num_global_experts = (
                parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        tp_size = TPXCollectiveComm.get_comm_group_world_size()
        tp_rank = TPXCollectiveComm.get_comm_rank()

        prepend_axis_num = len(sharded_offsets)
        replica_id = (
            0,
            0,
            parallel_state.get_data_modulo_expert_parallel_rank(with_context_parallel=True),
        )

        @torch.no_grad()
        def sh_ten_build_fn(
                key: str,
                t: torch.Tensor,
                replica_id: ReplicaId,
                flattened_range: Optional[slice],
                tp_axis: int,
                with_glu: bool,
        ):
            if tp_axis == 0:
                real_shape = (self.num_local_experts, self.config.hidden_size // get_args().tp_y, -1)
            elif tp_axis == 1:
                real_shape = (self.num_local_experts, -1, self.config.hidden_size // get_args().tp_y)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is None:
                t = t.view(real_shape).transpose(-1, -2)
                if with_glu:
                    local_tensors = torch.chunk(t, 2, -2)
                    sub_states = [
                        ShardedTensor.from_rank_offsets(
                            key,
                            local_tensors[0].contiguous(),
                            *sharded_offsets,
                            (
                                prepend_axis_num,
                                parallel_state.get_expert_model_parallel_rank(),
                                parallel_state.get_expert_model_parallel_world_size(),
                            ),
                            (prepend_axis_num + 1, tp_rank, tp_size * 2),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        ),
                        ShardedTensor.from_rank_offsets(
                            key,
                            local_tensors[1].contiguous(),
                            *sharded_offsets,
                            (
                                prepend_axis_num,
                                parallel_state.get_expert_model_parallel_rank(),
                                parallel_state.get_expert_model_parallel_world_size(),
                            ),
                            (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        ),
                    ]
                else:
                    sub_states = ShardedTensor.from_rank_offsets(
                        key,
                        t.contiguous(),
                        *sharded_offsets,
                        (
                            prepend_axis_num,
                            parallel_state.get_expert_model_parallel_rank(),
                            parallel_state.get_expert_model_parallel_world_size(),
                        ),
                        (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
            else:
                raise NotImplementedError(
                    "Currently GroupedMLP does not support distributed checkpointing "
                    "with the distributed optimizer."
                )
            return sub_states

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if tp_axis == 0:
                weight_shape = (self.config.hidden_size, -1)
            elif tp_axis == 1:
                weight_shape = (-1, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if with_glu:
                sub_state_dict = torch.cat(sub_state_dict, -2)
            return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix="", keep_vars=True)
        # To align with SequentialMLP, the weight tensors are transposed,
        # and the tp_axis is also for the transposed tensors
        for name, tensor in state_dict.items():
            if name == "weight1":
                tp_axis = 0
                with_glu = self.config.gated_linear_unit
                wkey = f"{prefix}experts.linear_fc1.weight"
            else:
                tp_axis = 1
                with_glu = False
                wkey = f"{prefix}experts.linear_fc2.weight"
            sharded_state_dict[f"{prefix}{name}"] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                replica_id,
            )

        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_modulo_expert_parallel_rank(with_context_parallel=True),
        )
        # Add fake _extra_state to be compatible with SequentialMLP
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )
            for mod in ["linear_fc1", "linear_fc2"]:
                sharded_state_dict[
                    f"{prefix}expert{expert_global_idx}.{mod}._extra_state"
                ] = make_sharded_object_for_checkpoint(
                    None, f"{prefix}experts.{mod}._extra_state", expert_sharded_offsets, replica_id,
                )

        return sharded_state_dict


class CustomGMM2DFC1(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, activation_input, weight, grouped_mlp_paras):
        # activation_input: [n, h], weight: [h/y, 2e*dff_h/x]

        ctx.grouped_mlp_paras = grouped_mlp_paras
        ctx.weight = weight

        num_local_experts = grouped_mlp_paras.get('num_local_experts')
        hidden_size = grouped_mlp_paras.get('hidden_size')
        tokens_per_expert = grouped_mlp_paras.get('tokens_per_expert')
        gemm_fusion = grouped_mlp_paras.get('gemm_fusion')
        tp_y = grouped_mlp_paras.get('tp_y')

        # [n, h] -> [n, h/y]
        activation_input = _split_along_last_dim(activation_input, TPYCollectiveComm)
        ctx.save_for_backward(activation_input)

        # [h/y, 2e*dff_h/x]-> [2e*dff_h/x, h/y]
        w1 = weight.transpose(0, -1).contiguous()
        # [2e*dff_h/x, h/y] -> [e, 2*dff_h/x, h/y]
        w1 = w1.view(num_local_experts, -1, hidden_size // tp_y)
        # [e, 2*dff_h/x, h/y] -> [e, h/y, 2*dff_h/x]
        w1 = w1.transpose(1, -1).contiguous()

        # [n, h/y] @ [e, h/y, 2*dff_h/x] -> [n, 2e*dff_h/x] partial-y
        fc1_output = gg.ops.gmm(
            activation_input,
            w1,
            tokens_per_expert,
            trans_b=False,
            gemm_fusion=gemm_fusion,
            original_weight=weight
        )

        # padding for reduce scatter, [n, 2e*dff_h/x] partial-y -> [n1, 2e*dff_h/x] partial-y
        global G_FORWARD_PADDING_SIZE
        n_tokens, h = fc1_output.shape
        rs_size = TPYCollectiveComm.get_comm_group_world_size()
        remaining = n_tokens - n_tokens // rs_size * rs_size
        G_FORWARD_PADDING_SIZE = rs_size - remaining if remaining else 0
        if G_FORWARD_PADDING_SIZE != 0:
            padding_tensor = torch.zeros(
                G_FORWARD_PADDING_SIZE, h, dtype=fc1_output.dtype, device=fc1_output.device
            )
            fc1_output = torch.cat((fc1_output, padding_tensor), dim=0)

        # [n1, 2e*dff_h/x] partial-y -> [n1/y, 2e*dff_h/x]
        fc1_output = sync_reduce_scatter_along_first_dim(fc1_output, TPYCollectiveComm)

        return fc1_output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output shape: [n1/y, 2e*dff_h/x]

        # activation_input shape: [n, h/y]
        activation_input, = ctx.saved_tensors
        grouped_mlp_paras = ctx.grouped_mlp_paras

        # weight shape: [h/y, 2e*dff_h/x]
        weight = ctx.weight

        num_local_experts = grouped_mlp_paras.get('num_local_experts')
        tokens_per_expert = grouped_mlp_paras.get('tokens_per_expert')
        hidden_size = grouped_mlp_paras.get('hidden_size')
        gemm_fusion = grouped_mlp_paras.get('gemm_fusion')
        tp_y = grouped_mlp_paras.get('tp_y')

        #  weight shape: [h/y, 2e*dff_h/x] -> [2e*dff_h/x, h/y]
        w1 = weight.t().contiguous()
        # [2e*dff_h/x, h/y] -> [e, 2*dff_h/x, h/y]
        w1 = w1.view(num_local_experts, -1, hidden_size // tp_y)

        # [n1/y, 2e*dff_h/x] -> [n1, 2e*dff_h/x]
        total_grad_output = sync_gather_along_first_dim(grad_output, TPYCollectiveComm)

        # unpadding, [n1, 2e*dff_h/x] -> [n, 2e*dff_h/x]
        global G_BACKWARD_PADDING_SIZE
        if G_BACKWARD_PADDING_SIZE != 0:
            real_input_num = total_grad_output.shape[0] - G_BACKWARD_PADDING_SIZE
            total_grad_output = total_grad_output[:real_input_num, :]

        # [n, 2e*dff_h/x] @ [e, 2*dff_h/x, h/y] = [n, h/y] partial-x
        grad_gmm_output = gg.ops.gmm(
            total_grad_output,
            w1,
            tokens_per_expert,
            trans_b=False,
            gemm_fusion=gemm_fusion,
        )

        group_list = torch.cumsum(tokens_per_expert, dim=0)
        # [h/y, n] @ [n, 2e*dff_h/x] = [e, h/y, 2*dff_h/x]
        grad_weight_output = GMMFunction.builder.load().npu_gmm(
            [activation_input.t()],
            [total_grad_output],
            [],
            group_list,
            2,
            0)[0]

        # [e, h/y, 2*dff_h/x] -> [e, 2*dff_h/x, h/y]
        grad_weight_output = grad_weight_output.transpose(1, -1).contiguous()

        # [e, 2*dff_h/x, h/y] -> [2e*dff_h/x, h/y]
        grad_weight_output = grad_weight_output.view(-1, grad_weight_output.shape[-1])
        # [2e*dff_h/x, h/y] -> [h/y, 2e*dff_h/x]
        grad_weight_output = grad_weight_output.transpose(0, 1).contiguous()

        # [n, h/y] partial-x -> [n, h] partial-x
        grad_gmm_output = sync_gather_along_last_dim(grad_gmm_output, TPYCollectiveComm)

        return grad_gmm_output, grad_weight_output, None


class CustomGMM2DFC2(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, activation_input, weight, grouped_mlp_paras):
        # activation_input shape: [n1/y, e*dff_h/x], weight shape: [e*dff_h/x, h/y]

        ctx.grouped_mlp_paras = grouped_mlp_paras
        ctx.weight = weight

        num_local_experts = grouped_mlp_paras.get('num_local_experts')
        hidden_size = grouped_mlp_paras.get('hidden_size')
        tokens_per_expert = grouped_mlp_paras.get('tokens_per_expert')
        gemm_fusion = grouped_mlp_paras.get('gemm_fusion')
        tp_y = grouped_mlp_paras.get('tp_y')

        # [e*dff_h/x, h/y] -> [e, dff_h/x, h/y]
        w2 = weight.view(num_local_experts, -1, hidden_size // tp_y)

        # [n1/y, e*dff_h/x] -> [n1, e*dff_h/x]
        total_input = sync_gather_along_first_dim(activation_input, TPYCollectiveComm)

        # unpadding, [n1, e*dff_h/x] -> [n, e*dff_h/x]
        global G_FORWARD_PADDING_SIZE
        if G_FORWARD_PADDING_SIZE != 0:
            real_input_num = total_input.shape[0] - G_FORWARD_PADDING_SIZE
            total_input = total_input[:real_input_num, :]

        ctx.save_for_backward(total_input)

        # [n, e*dff_h/x] @ [e, dff_h/x, h/y] -> [n, h/y] partial-x
        fc2_output = gg.ops.gmm(
            total_input,
            w2,
            tokens_per_expert,
            trans_b=False,
            gemm_fusion=gemm_fusion,
            original_weight=weight
        )

        # [n, h/y] partial-x -> [n, h] partial-x
        fc2_output = sync_gather_along_last_dim(fc2_output, TPYCollectiveComm)

        return fc2_output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output shape: [n, h]

        # activation_input shape: [n, e*dff_h/x]
        activation_input, = ctx.saved_tensors
        grouped_mlp_paras = ctx.grouped_mlp_paras

        # weight 2 shape: [e*dff_h/x, h/y]
        weight = ctx.weight

        num_local_experts = grouped_mlp_paras.get('num_local_experts')
        tokens_per_expert = grouped_mlp_paras.get('tokens_per_expert')
        hidden_size = grouped_mlp_paras.get('hidden_size')
        gemm_fusion = grouped_mlp_paras.get('gemm_fusion')
        tp_y = grouped_mlp_paras.get('tp_y')

        # weight shape: [e*dff_h/x, h/y] -> [e, dff_h/x, h/y]
        w2 = weight.view(num_local_experts, -1, hidden_size // tp_y)
        # [e, dff_h/x, h/y] -> [e, h/y, dff_h/x]
        w2 = w2.transpose(1, -1).contiguous()

        # [n, h] -> [n, h/y]
        grad_output = _split_along_last_dim(grad_output, TPYCollectiveComm)

        # [n, h/y] @ [e, h/y, dff_h/x] = [n, e*dff_h/x] partial-y
        partial_grad_gmm_output = gg.ops.gmm(
            grad_output,
            w2,
            tokens_per_expert,
            trans_b=False,
            gemm_fusion=gemm_fusion,
        )

        # padding for reduce scatter, [n, e*dff_h/x] -> [n1, e*dff_h/x]
        global G_BACKWARD_PADDING_SIZE
        n_tokens, h = partial_grad_gmm_output.shape
        rs_size = TPYCollectiveComm.get_comm_group_world_size()
        remaining = n_tokens - n_tokens // rs_size * rs_size
        G_BACKWARD_PADDING_SIZE = rs_size - remaining if remaining else 0
        if G_BACKWARD_PADDING_SIZE != 0:
            padding_tensor = torch.zeros(G_BACKWARD_PADDING_SIZE, h, dtype=partial_grad_gmm_output.dtype,
                                         device=partial_grad_gmm_output.device)
            partial_grad_gmm_output = torch.cat((partial_grad_gmm_output, padding_tensor), dim=0)

        # [n1, e*dff_h/x] -> [n1/y, e*dff_h/x]
        grad_gmm_output = sync_reduce_scatter_along_first_dim(partial_grad_gmm_output, TPYCollectiveComm)

        group_list = torch.cumsum(tokens_per_expert, dim=0)
        # [e*dff_h/x, n] @ [n, h/y] = [e, dff_h/x, h/y]
        grad_weight_output = GMMFunction.builder.load().npu_gmm(
            [activation_input.t()],
            [grad_output],
            [],
            group_list,
            2,
            0)[0]
        # [e, dff_h/x, h/y] -> [e*dff_h/x, h/y]
        grad_weight_output = grad_weight_output.view(-1, grad_weight_output.shape[-1])

        return grad_gmm_output, grad_weight_output, None
