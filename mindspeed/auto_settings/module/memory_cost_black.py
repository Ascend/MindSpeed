"""
算子预估
"""
import math
import os

from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.utils import get_module_info, get_black_prof_file
from mindspeed.auto_settings.config.search_config import SearchConfig


class Activation:
    """
    Modeling the activation memory generated during the forward of a single transformer block.
    """

    def __init__(self, config: SearchConfig):
        self.unit_gb = 1024 ** 3
        self.config = config
        self.tp = config.tensor_model_parallel_size
        self.cp = config.ring_attention_size
        self.up = config.ulysses_size
        self.ep = config.expert_model_parallel_size
        self.mbs = config.micro_batch_size
        self.seq_len = config.seq_length
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.num_query_groups = config.num_query_groups
        self.num_attention_heads = config.num_attention_heads
        self.num_experts = config.num_experts
        self.swiglu = config.swiglu
        self.top_k = config.moe_router_topk
        self.recompute_activation_function = config.recompute_activation_function
        self.swap_attention = config.swap_attention
        # if hasattr(config, 'swap_attention'):
        #     self.swap_attention = config.swap_attention

    @property
    def activation_mem(self):
        _config = self.config.crop()
        file_path = get_black_prof_file(_config)
        act_mem = get_module_info(file_path, '0', 'memory')
        if math.isinf(act_mem):
            _config.micro_batch_size = 1
            act_mem = get_module_info(file_path, "0", "memory")
            act_mem *= self.config.micro_batch_size
        return act_mem * self.unit_gb

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


class MemoryCostBlack(object):
    unit_gb = 1024 ** 3
    cann_memory = 4.5 * 1024 ** 3

    def __init__(self):
        self.logger = get_logger("MemoryCostBlack")

    def compute_params(self, config: SearchConfig):
        """Calculate model parameters on stage0."""
        pp = config.pipeline_model_parallel_size
        tp = config.tensor_model_parallel_size
        ep = config.expert_model_parallel_size
        num_experts = config.num_experts if config.num_experts else 1

        gated_linear_multiplier = 3 / 2 if config.swiglu else 1
        embedding_size = config.hidden_size * config.padded_vocab_size
        num_parameters_in_transformer_layers = (
                2
                * config.num_layers
                * config.hidden_size
                * config.hidden_size
                * (
                        1
                        + ((config.ffn_hidden_size / config.hidden_size) * num_experts * gated_linear_multiplier)
                        + (config.num_query_groups / config.num_attention_heads)
                        + (2 / config.hidden_size)
                        + (1 / (config.num_layers * config.hidden_size))
                )
        )
        mlp_params_shard = (
                2
                * config.hidden_size * config.ffn_hidden_size
                * num_experts * gated_linear_multiplier
                * config.num_layers / pp
        )
        total_params_count = (
                (
                        num_parameters_in_transformer_layers / pp
                        + embedding_size
                        - mlp_params_shard
                ) / tp
                + (mlp_params_shard / tp if ep is None else mlp_params_shard / tp / ep)
        )
        if config.untie_embeddings_and_output_weights and pp == 1:
            total_params_count += embedding_size / tp
        self.logger.debug(f'num_parameters_in_transformer_layers: {num_parameters_in_transformer_layers}')
        self.logger.debug(f'mlp_params_shard: {mlp_params_shard}')
        self.logger.debug(f'total_params_count: {total_params_count}')

        return int(total_params_count)

    def compute_static_memory(self, params: int, config: SearchConfig):
        dp = config.data_parallel_size
        if config.fp16:
            mem_para = 2 * params
            mem_grad = 2 * params
            if config.reuse_fp32_param and config.use_distributed_optimizer:
                mem_optimizer = 4 * params + 8 * params / dp
            elif config.use_distributed_optimizer:
                mem_optimizer = 4 * params + 4 * params + 8 * params / dp
            elif config.reuse_fp32_param:
                mem_optimizer = 12 * params
            else:
                mem_optimizer = 16 * params
        elif config.bf16:
            if config.reuse_fp32_param and config.use_distributed_optimizer:
                mem_para = 0
                mem_grad = 4 * params
                mem_optimizer = 4 * params + 8 * params / dp
            elif config.use_distributed_optimizer:
                mem_para = 2 * params
                mem_grad = 4 * params
                mem_optimizer = 4 * params + 8 * params / dp
            elif config.reuse_fp32_param:
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

    def get_peak_memory(self, config: SearchConfig):
        model_config = get_model_config()
        pp = config.pp
        vpp = config.vpp
        activation = Activation(config)

        params = self.compute_params(config)
        mem_para, mem_grad, mem_optimizer = self.compute_static_memory(params, config)
        mem_activation_per_layer = activation.activation_mem
        if vpp == 1:
            # non-interleaved pipeline
            mem_activation_per_batch = mem_activation_per_layer * (model_config.num_layers // pp)
            mem_activation = mem_activation_per_batch * pp
        else:
            num_layers_per_vpp_stage = model_config.num_layers // pp // vpp
            mem_activation_per_batch = mem_activation_per_layer * num_layers_per_vpp_stage
            mem_activation = mem_activation_per_batch * (pp * vpp + (pp - 1))

        if model_config.recompute_granularity == 'full':
            mem_activation = 0
            mem_activation_per_layer = 0
            mem_activation_per_batch = 0

        m1 = mem_para + mem_optimizer + mem_activation
        m2 = mem_para + mem_optimizer + mem_activation + mem_grad - mem_activation_per_batch
        m3 = mem_para + mem_optimizer + mem_activation + mem_grad
        peak_memory = (max(m1, m2, m3) + self.cann_memory) / self.unit_gb

        self.logger.debug(
            f"### config: {config} \n"
            f"mem_para: {mem_para / self.unit_gb}\n"
            f"mem_grad: {mem_grad / self.unit_gb}\n"
            f"mem_optimizer: {mem_optimizer / self.unit_gb}\n"
            f"mem_activate_per_layer: {mem_activation_per_layer / self.unit_gb}\n"
            f"mem_activation: {mem_activation / self.unit_gb}\n"
            f"peak_memory: {peak_memory}"
        )
        return peak_memory
