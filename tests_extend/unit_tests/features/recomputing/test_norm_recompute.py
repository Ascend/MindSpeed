from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args

from mindspeed import megatron_adaptor  # noqa: F401
from mindspeed.core.memory.recompute.norm.adaptor import (
    _split_fused_norm_linear,
    norm_recompute_layer_init_wrapper,
)
from mindspeed.features_manager.recompute.norm_function import RecomputeNormFeature
from mindspeed.model.transformer import set_attention_mask
from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

pytestmark = pytest.mark.slow


class RecordingPatchManager:
    def __init__(self):
        self.targets = []

    def register_patch(self, target, patch):
        self.targets.append((target, patch))


def test_norm_recompute_registers_megatron_018_split_forward_patches():
    args = SimpleNamespace(recompute_norm=True)
    patch_manager = RecordingPatchManager()

    RecomputeNormFeature().register_patches(patch_manager, args)

    assert [target for target, _ in patch_manager.targets] == [
        "megatron.core.transformer.transformer_layer.TransformerLayer.__init__",
        "megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention",
        "megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp",
    ]


def test_norm_recompute_layer_init_uses_native_megatron_flags(monkeypatch):
    class LocalNorm:
        pass

    layer = SimpleNamespace(
        config=SimpleNamespace(transformer_impl="local"),
        layer_number=1,
        input_layernorm=LocalNorm(),
        pre_mlp_layernorm=LocalNorm(),
        recompute_input_layernorm=False,
        recompute_pre_mlp_layernorm=False,
    )
    monkeypatch.setattr(
        "mindspeed.core.memory.recompute.norm.adaptor.should_recompute_norm",
        lambda layer_number, config: True,
    )

    norm_recompute_layer_init_wrapper(lambda self: None)(layer)

    assert layer.recompute_input_layernorm
    assert layer.recompute_pre_mlp_layernorm
    assert not layer.mindspeed_recompute_fused_input_layernorm
    assert not layer.mindspeed_recompute_fused_pre_mlp_layernorm


def test_norm_recompute_layer_init_marks_transformer_engine_fused_norms(monkeypatch):
    class FusedLinearWithNormRecompute:
        @staticmethod
        def enable_recompute_norm(checkpoint_manager):
            del checkpoint_manager

    layer = SimpleNamespace(
        config=SimpleNamespace(transformer_impl="transformer_engine"),
        layer_number=1,
        input_layernorm=IdentityOp(),
        pre_mlp_layernorm=IdentityOp(),
        self_attention=SimpleNamespace(linear_qkv=FusedLinearWithNormRecompute()),
        mlp=SimpleNamespace(linear_fc1=FusedLinearWithNormRecompute()),
        recompute_input_layernorm=False,
        recompute_pre_mlp_layernorm=False,
    )
    monkeypatch.setattr(
        "mindspeed.core.memory.recompute.norm.adaptor.should_recompute_norm",
        lambda layer_number, config: True,
    )

    norm_recompute_layer_init_wrapper(lambda self: None)(layer)

    assert not layer.recompute_input_layernorm
    assert not layer.recompute_pre_mlp_layernorm
    assert layer.mindspeed_recompute_fused_input_layernorm
    assert layer.mindspeed_recompute_fused_pre_mlp_layernorm


def test_norm_recompute_layer_init_splits_te_modules_without_legacy_recompute_api(
    monkeypatch,
):
    input_norm = object()
    pre_mlp_norm = object()
    linear_qkv = object()
    linear_fc1 = object()
    layer = SimpleNamespace(
        config=SimpleNamespace(transformer_impl="transformer_engine"),
        layer_number=1,
        input_layernorm=IdentityOp(),
        pre_mlp_layernorm=IdentityOp(),
        self_attention=SimpleNamespace(linear_qkv=object()),
        mlp=SimpleNamespace(linear_fc1=object()),
        recompute_input_layernorm=False,
        recompute_pre_mlp_layernorm=False,
    )

    def split_fused_norm_linear(
        layer, parent, submodule_name, norm_name, fused_linear_name
    ):
        del fused_linear_name
        if submodule_name == "linear_qkv":
            setattr(layer, norm_name, input_norm)
            setattr(parent, submodule_name, linear_qkv)
        else:
            setattr(layer, norm_name, pre_mlp_norm)
            setattr(parent, submodule_name, linear_fc1)

    monkeypatch.setattr(
        "mindspeed.core.memory.recompute.norm.adaptor.should_recompute_norm",
        lambda layer_number, config: True,
    )
    monkeypatch.setattr(
        "mindspeed.core.memory.recompute.norm.adaptor._split_fused_norm_linear",
        split_fused_norm_linear,
    )

    norm_recompute_layer_init_wrapper(lambda self: None)(layer)

    assert layer.input_layernorm is input_norm
    assert layer.pre_mlp_layernorm is pre_mlp_norm
    assert layer.self_attention.linear_qkv is linear_qkv
    assert layer.mlp.linear_fc1 is linear_fc1
    assert layer.recompute_input_layernorm
    assert layer.recompute_pre_mlp_layernorm
    assert not layer.mindspeed_recompute_fused_input_layernorm
    assert not layer.mindspeed_recompute_fused_pre_mlp_layernorm


def test_split_fused_norm_linear_preserves_distributed_checkpoint_keys(monkeypatch):
    tp_group = object()
    fused_linear = SimpleNamespace(_tp_group=tp_group)
    standalone_norm = object()
    standalone_linear = object()
    parent = SimpleNamespace(linear_qkv=fused_linear)
    layer = SimpleNamespace(
        config=object(),
        submodules_config=SimpleNamespace(sharded_state_dict_keys_map={}),
        input_layernorm=IdentityOp(),
    )

    def split_fused_norm_linear(fused, config, tp_group):
        del config
        assert fused.tp_group is tp_group
        assert fused.ub_name == "qkv"
        return standalone_norm, standalone_linear

    monkeypatch.setattr(
        "mindspeed.core.memory.recompute.norm.adaptor.split_te_layernorm_column_parallel_linear",
        split_fused_norm_linear,
    )

    _split_fused_norm_linear(
        layer,
        parent,
        "linear_qkv",
        "input_layernorm",
        "self_attention.linear_qkv",
    )

    assert layer.input_layernorm is standalone_norm
    assert parent.linear_qkv is standalone_linear
    assert layer.submodules_config.sharded_state_dict_keys_map == {
        "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
        "input_layernorm.bias": "self_attention.linear_qkv.layer_norm_bias",
    }


class TestNormRecompute(DistributedTest):
    world_size = 8

    def test_norm_recompute(self):
        args = parse_args(None, True)
        args.recompute_norm = True
        args.num_layers = 4
        args.recompute_norm_num_layers = 2
        args.pipeline_model_parallel_size = 2
        args.pipeline_dtype = torch.float32
        args.num_query_groups = None
        set_args(args)
        self.norm_recompute()

    def norm_recompute(self):
        initialize_model_parallel(2, 2)
        model_parallel_cuda_manual_seed(312)

        ref_config = TransformerConfig(
            num_layers=4,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        ref_config.hidden_dropout = 0
        ref_config.attention_dropout = 0
        ref_config.gradient_accumulation_fusion = False
        test_config = TransformerConfig(
            num_layers=4,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        test_config.hidden_dropout = 0
        test_config.attention_dropout = 0
        test_config.gradient_accumulation_fusion = False

        transformer_block_ref = TransformerBlock(
            ref_config, get_gpt_layer_local_spec(), post_layer_norm=True
        )
        transformer_block_test = TransformerBlock(
            test_config, get_gpt_layer_local_spec(), post_layer_norm=True
        )
        transformer_block_test.load_state_dict(
            transformer_block_ref.state_dict().copy()
        )

        # Megatron 0.18 owns the local LayerNorm checkpoint path selected by the MindSpeed init wrapper.
        for layer in transformer_block_test.layers:
            layer.recompute_input_layernorm = True
            layer.recompute_pre_mlp_layernorm = True

        sequence_length = 32
        micro_batch_size = 2
        transformer_block_ref.cuda()
        transformer_block_test.cuda()

        hidden_states_ref = torch.rand(
            (sequence_length, micro_batch_size, ref_config.hidden_size)
        ).cuda()
        hidden_states_ref.requires_grad = True
        hidden_states_test = hidden_states_ref.clone().detach()
        hidden_states_test.requires_grad = True

        attention_mask = torch.zeros(
            (1, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()
        set_attention_mask(attention_mask)

        out_ref = transformer_block_ref(
            hidden_states=hidden_states_ref, attention_mask=attention_mask
        )
        out_test = transformer_block_test(
            hidden_states=hidden_states_test, attention_mask=attention_mask
        )
        assert torch.allclose(out_ref, out_test)

        out_ref.backward(torch.ones_like(out_ref))
        out_test.backward(torch.ones_like(out_ref))
        assert torch.allclose(hidden_states_ref.grad, hidden_states_test.grad)
