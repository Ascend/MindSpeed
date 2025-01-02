# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import math
from typing import Literal

from ..utils import logger
from ..utils.parallel_config import ParallelConfig
from ..utils.model_config import get_model_config
from ..utils.utils import get_module_info


class Activation:
    """
    Modeling the activation memory generated during the forward of a single transformer block.
    """

    def __init__(self, config: ParallelConfig, method: Literal['white_box', 'black_box']):
        args = get_model_config().args
        self.unit_gb = 1024 ** 3
        self.method = method
        self.config = config
        self.tp = config.tensor_model_parallel_size
        self.cp = config.ring_attention_size
        self.up = config.ulysses_size
        self.ep = config.expert_model_parallel_size
        self.mbs = config.micro_batch_size
        self.seq_len = args.seq_length
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.num_query_groups = args.num_query_groups
        self.num_attention_heads = args.num_attention_heads
        self.num_experts = args.num_experts
        self.swiglu = args.swiglu
        self.top_k = args.moe_router_topk
        self.recompute_activation_function = args.recompute_activation_function
        self.swap_attention = False
        if hasattr(args, 'swap_attention'):
            self.swap_attention = args.swap_attention

    @property
    def activation_mem(self):
        if self.method == 'black_box':
            cropped_config: ParallelConfig = self.config.crop_config()
            act_mem = get_module_info(cropped_config.module_profile_path(node_rank=0), '0', 'memory')
            if math.isinf(act_mem):
                cropped_config.micro_batch_size = 1
                act_mem = get_module_info(cropped_config.module_profile_path(node_rank=0), "0", "memory")
                act_mem *= self.config.micro_batch_size
            return act_mem * self.unit_gb

        func_table = []
        # input_layernorm
        func_table.append(self.layer_norm)
        # attention
        if not self.swap_attention:
            func_table.append(self.linear_qkv)
            func_table.append(self.core_attention)
            func_table.append(self.linear_proj)
        # pre_mlp_layernorm
        func_table.append(self.layer_norm)
        # mlp
        func_table.append(self.moe_layer if get_model_config().is_moe_model() else self.mlp)
        return sum([func() for func in func_table])

    def layer_norm(self):
        shape = [self.seq_len // self.cp // self.up // self.tp, self.mbs, self.hidden_size]
        return 2 * math.prod(shape)

    def linear_qkv(self):
        ng = self.num_query_groups // self.tp
        np = self.num_attention_heads // self.tp
        head_dim = self.hidden_size // self.num_attention_heads
        shape = [self.seq_len // self.cp // self.up, self.mbs, ng * (np // ng + 2) * head_dim]
        return 2 * math.prod(shape)

    def linear_proj(self):
        shape = [self.seq_len // self.cp // self.up // self.tp, self.mbs, self.hidden_size]
        return 2 * math.prod(shape)

    def core_attention(self):
        ng = self.num_query_groups // self.tp
        np = self.num_attention_heads // self.tp
        head_dim = self.hidden_size // self.num_attention_heads
        q_shape = [self.seq_len // self.cp // self.up, self.mbs, np, head_dim]
        q_mem = 2 * math.prod(q_shape)
        ret = q_mem
        if self.up > 1:
            ret += 4 * q_mem
        if self.cp > 1:
            ret += (2048 * 2048)
        return ret

    def mlp(self):
        ffn_hidden_size = self.ffn_hidden_size
        if self.swiglu:
            ffn_hidden_size *= 2

        if self.ep == 0:
            linear1_shape = [self.seq_len // self.cp // self.up, self.mbs, ffn_hidden_size // self.tp]
            linear1_mem = 2 * math.prod(linear1_shape)

            activation_func_mem = linear1_mem
            if self.swiglu:
                activation_func_mem /= 2

            linear2_shape = [self.seq_len // self.cp // self.up // self.tp, self.mbs, self.hidden_size]
            linear2_mem = 2 * math.prod(linear2_shape)
        else:
            num_total_tokens = self.seq_len // self.cp // self.up * self.ep * self.top_k
            linear1_shape = [num_total_tokens // self.num_experts, self.mbs, ffn_hidden_size // self.tp]
            linear1_mem = 2 * math.prod(linear1_shape)

            activation_func_mem = linear1_mem
            if self.swiglu:
                activation_func_mem = activation_func_mem // 2

            linear2_shape = [num_total_tokens // self.num_experts, self.mbs, self.hidden_size]
            linear2_mem = 2 * math.prod(linear2_shape)

        if self.recompute_activation_function:
            activation_func_mem = 0

        return linear1_mem + activation_func_mem + linear2_mem

    def moe_layer(self):
        num_local_experts = self.num_experts // self.ep
        num_total_tokens = self.seq_len // self.cp // self.up * self.ep * self.top_k

        shape = [num_total_tokens // self.num_experts * num_local_experts, self.mbs, self.hidden_size]
        dispatcher = 2 * math.prod(shape)

        sequential_mlp = self.mlp() * num_local_experts

        shape = [num_total_tokens // self.num_experts * num_local_experts, self.mbs, self.hidden_size]
        undispatcher = 2 * math.prod(shape)

        return dispatcher + sequential_mlp + undispatcher


class MemoryCostModel:

    unit_gb = 1024 ** 3
    cann_memory = 4.5 * 1024 ** 3

    @classmethod
    def compute_params(cls, config: ParallelConfig):
        """Calculate model parameters on stage0."""
        args = get_model_config().args
        pp = config.pipeline_model_parallel_size
        tp = config.tensor_model_parallel_size
        ep = config.expert_model_parallel_size
        num_experts = args.num_experts if args.num_experts else 1

        gated_linear_multiplier = 3 / 2 if args.swiglu else 1
        embedding_size = args.hidden_size * args.padded_vocab_size
        num_parameters_in_transformer_layers = (
                2
                * args.num_layers
                * args.hidden_size
                * args.hidden_size
                * (
                        1
                        + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
                        + (args.num_query_groups / args.num_attention_heads)
                        + (2 / args.hidden_size)
                        + (1 / (args.num_layers * args.hidden_size))
                )
        )
        mlp_params_shard = (
                2
                * args.hidden_size * args.ffn_hidden_size
                * num_experts * gated_linear_multiplier
                * args.num_layers / pp
        )
        total_params_count = (
                (
                        num_parameters_in_transformer_layers / pp
                        + embedding_size
                        - mlp_params_shard
                ) / tp
                + (mlp_params_shard / tp if ep == 0 else mlp_params_shard / tp / ep)
        )
        if args.untie_embeddings_and_output_weights and pp == 1:
            total_params_count += embedding_size / tp
        logger.debug(f'num_parameters_in_transformer_layers: {num_parameters_in_transformer_layers}')
        logger.debug(f'mlp_params_shard: {mlp_params_shard}')
        logger.debug(f'total_params_count: {total_params_count}')

        return int(total_params_count)

    @classmethod
    def compute_static_memory(cls, params: int, config: ParallelConfig):
        args = get_model_config().args
        dp = config.data_parallel_size
        if args.fp16:
            mem_para = 2 * params
            mem_grad = 2 * params
            if args.reuse_fp32_param and args.use_distributed_optimizer:
                mem_optimizer = 4 * params + 8 * params / dp
            elif args.use_distributed_optimizer:
                mem_optimizer = 4 * params + 4 * params + 8 * params / dp
            elif args.reuse_fp32_param:
                mem_optimizer = 12 * params
            else:
                mem_optimizer = 16 * params
        elif args.bf16:
            if args.reuse_fp32_param and args.use_distributed_optimizer:
                mem_para = 0
                mem_grad = 4 * params
                mem_optimizer = 4 * params + 8 * params / dp
            elif args.use_distributed_optimizer:
                mem_para = 2 * params
                mem_grad = 4 * params
                mem_optimizer = 4 * params + 8 * params / dp
            elif args.reuse_fp32_param:
                mem_para = 0
                mem_grad = 4 * params
                mem_optimizer = 4 * params + 8 * params
            else:
                mem_para = 2 * params
                mem_grad = 4 * params
                mem_optimizer = 4 * params + 4 * params + 4 * params
        else:
            raise AssertionError('not support fp32 training')
        return mem_para, mem_grad, mem_optimizer

    @classmethod
    def get_peak_memory(cls, config: ParallelConfig, method: Literal['white_box', 'black_box']):
        args = get_model_config().args
        pp = config.pipeline_model_parallel_size
        vpp = config.virtual_pipeline_model_parallel_size
        activation = Activation(config, method)

        params = cls.compute_params(config)
        mem_para, mem_grad, mem_optimizer = cls.compute_static_memory(params, config)
        mem_activation_per_layer = activation.activation_mem
        if vpp == 1:
            # non-interleaved pipeline
            mem_activation_per_batch = mem_activation_per_layer * (args.num_layers // pp)
            mem_activation = mem_activation_per_batch * pp
        else:
            num_layers_per_vpp_stage = args.num_layers // pp // vpp
            mem_activation_per_batch = mem_activation_per_layer * num_layers_per_vpp_stage
            mem_activation = mem_activation_per_batch * (pp * vpp + (pp - 1))

        if args.recompute_granularity == 'full':
            mem_activation = 0
            mem_activation_per_layer = 0
            mem_activation_per_batch = 0

        m1 = mem_para + mem_optimizer + mem_activation
        m2 = mem_para + mem_optimizer + mem_activation + mem_grad - mem_activation_per_batch
        m3 = mem_para + mem_optimizer + mem_activation + mem_grad
        peak_memory = (max(m1, m2, m3) + cls.cann_memory) / cls.unit_gb

        logger.debug(
            f"### config: {config} method: {method} \n"
            f"mem_para: {mem_para / cls.unit_gb}\n"
            f"mem_grad: {mem_grad / cls.unit_gb}\n"
            f"mem_optimizer: {mem_optimizer / cls.unit_gb}\n"
            f"mem_activate_per_layer: {mem_activation_per_layer / cls.unit_gb}\n"
            f"mem_activation: {mem_activation / cls.unit_gb}\n"
            f"peak_memory: {peak_memory}"
        )
        return peak_memory
