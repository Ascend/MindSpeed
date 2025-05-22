from functools import wraps
from typing import Optional
import types
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.training import get_args
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.extensions.transformer_engine import TENorm
from mindspeed.core.transformer.transformer import norm_recompute_forward
from mindspeed.core.transformer.transformer_block import NoopTransformerLayer
from mindspeed.model.transformer import should_recompute_norm


def get_gpt_decoder_block_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
):
        res = fn(config, use_transformer_engine)
        res.layer_norm = TENorm
        return res

    return wrapper


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: Optional[int] = None,
                moe_grouped_gemm: Optional[bool] = False,
                qk_layernorm: Optional[bool] = False,
                multi_latent_attention: Optional[bool] = False,
                fp8: Optional[str] = None,  # pylint: disable=unused-arguments
                moe_use_legacy_grouped_gemm: Optional[bool] = False,
                normalization: Optional[str] = None,
                qk_l2_norm: Optional[bool] = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm, multi_latent_attention)
        args = get_args()
        if args.multi_head_latent_attention:
            res.submodules.self_attention.submodules = SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=TENorm if args.qk_layernorm else IdentityOp,
                k_layernorm=TENorm if args.qk_layernorm else IdentityOp,
                linear_qb=ColumnParallelLinear,
                linear_kvb=ColumnParallelLinear
            )
        else:
            if qk_layernorm:
                res.submodules.self_attention.submodules.q_layernorm = TENorm
                res.submodules.self_attention.submodules.k_layernorm = TENorm
        res.submodules.input_layernorm = TENorm
        res.submodules.pre_mlp_layernorm = TENorm
        return res

    return wrapper


def build_layers_wrapper(fn, column_forward, row_forward):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        for layer in self.layers:
            if isinstance(getattr(layer, 'mlp', None), MoELayer):
                for local_expert in layer.mlp.experts.local_experts:
                    local_expert.linear_fc1.forward = types.MethodType(column_forward, local_expert.linear_fc1)
                    local_expert.linear_fc2.forward = types.MethodType(row_forward, local_expert.linear_fc2)
    return wrapper


def build_norm_recompute_layer_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        for layer in self.layers:
            if isinstance(layer, NoopTransformerLayer):
                continue
            if should_recompute_norm(layer):
                layer.forward = types.MethodType(norm_recompute_forward, layer)
    return wrapper