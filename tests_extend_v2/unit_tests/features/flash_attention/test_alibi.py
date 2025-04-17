import pytest
import torch
import torch_npu

from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.transformer.transformer_config import TransformerConfig

from mindspeed import megatron_adaptor_v2
from mindspeed.core.transformer.flash_attention.alibi.adaptor import MindSpeedDotProductAttention

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def run_fusion_attn_with_pse_alibi(bs, seq_len, dtype):
    from megatron.core.transformer.enums import AttnMaskType

    args = parse_args(None, True)
    set_args(args)

    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        attention_dropout=0.0,
        params_dtype=dtype
    )

    # extra arguments mindspeed needed
    config.use_flash_attn = True
    config.use_fusion_attn_v2 = True
    config.alibi_fusion_attn_type = 2
    config.sparse_mode = 2
    config.seq_length = seq_len
    config.alibi_diagonal_opposite = False

    attn = MindSpeedDotProductAttention(
        config=config, 
        layer_number=1, 
        attn_mask_type=AttnMaskType.causal, 
        attention_type='self'
    )

    # attn.pse should exist and not be None
    assert attn.pse is not None

    b, n, s, d = bs, 4, seq_len, 8

    q = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)

    # global attn mask will be generated at DotProductAttention forward wrapper
    out = attn(q, k, v, None, None, None, None)
    assert isinstance(out, torch.Tensor)


class TestAlibi():

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_alibi(self, mocker):
        mock_world_size = mocker.patch(
            "megatron.core.parallel_state.get_tensor_model_parallel_world_size",
            return_value=1
        )
        mock_rank = mocker.patch(
            "megatron.core.parallel_state.get_tensor_model_parallel_rank",
            return_value=0
        )
        run_fusion_attn_with_pse_alibi(2, 256, torch.bfloat16)
        mock_world_size.assert_called()
        mock_rank.assert_called_once()
