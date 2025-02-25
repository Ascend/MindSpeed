from functools import wraps
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.training import get_args
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.spec_utils import ModuleSpec
from mindspeed.core.transformer.transformer import norm_recompute_forward
from mindspeed.core.transformer.transformer_block import NoopTransformerLayer
from mindspeed.model.transformer import should_recompute_norm
from mindspeed.core.transformer.moe.tp_2d.moe_layer_2d import MoELayer2D
import types


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm)
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


def get_mlp_module_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        use_te, num_experts, moe_grouped_gemm = kwargs['use_te'], kwargs['num_experts'], kwargs['moe_grouped_gemm']
        if num_experts is None:
            # Dense MLP w/ or w/o TE modules.
            return ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                ),
            )
        else:
            # Mixture of experts with modules in megatron core.
            if use_te and moe_grouped_gemm:
                linear_fc1 = TEColumnParallelGroupedLinear
                linear_fc2 = TERowParallelGroupedLinear
            else:
                linear_fc1 = ColumnParallelLinear
                linear_fc2 = RowParallelLinear

            use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

            return ModuleSpec(
                module=MoELayer2D,
                submodules=(
                    MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                    if not moe_grouped_gemm or use_te_grouped_gemm
                    else None
                ),
            )

    return wrapper
