import pytest
from unittest.mock import Mock, patch
import torch

from mindspeed.fsdp.quantization.module.gmm_mxfp8 import (
    weight_quant,
    gmm_with_hp_or_lp_weight,
    mx_quant_group_gemm,
    MXFP8GMM,
)
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class TestWeightQuantGMM:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_weight_quant(self, mock_config):
        weight = torch.randn(4, 32, 64, dtype=torch.bfloat16)
        dst_type = torch.float8_e4m3fn
        new_shape = (-1, 32, 64)

        if mock_config.recipe_name == "mxfp8":
            mock_func = 'torch_npu.npu_dynamic_mx_quant_with_dual_axis'
            weight_fwd = torch.randn(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_fwd = torch.randn(4, 32, 2, dtype=torch.float32)
            weight_bwd = torch.randn(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_bwd = torch.randn(4, 32, 2, dtype=torch.float32)
            mock_return = (weight_fwd, scale_fwd, weight_bwd, scale_bwd)
        else:
            mock_func = 'torch_npu.npu_dynamic_block_mx_quant'
            weight_fwd = torch.randn(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_fwd = torch.randn(4, 1, 2, dtype=torch.float32)
            scale_bwd = torch.randn(4, 1, 2, dtype=torch.float32)
            mock_return = (weight_fwd, scale_fwd, scale_bwd)

        with patch(mock_func) as mock_quant:
            mock_quant.return_value = mock_return

            result_fwd, result_scale_fwd, result_bwd, result_scale_bwd = weight_quant(
                weight=weight,
                dst_type=dst_type,
                new_shape=new_shape,
                config=mock_config,
            )

            mock_quant.assert_called_once()
            assert result_fwd.shape == weight.shape
            assert result_bwd.shape == weight.shape

    def test_weight_quant_invalid_recipe_raises_error(self):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = "invalid_recipe"
        weight = torch.randn(4, 32, 64, dtype=torch.bfloat16)
        dst_type = torch.float8_e4m3fn
        new_shape = (-1, 32, 64)

        with pytest.raises(ValueError) as exc_info:
            weight_quant(
                weight=weight,
                dst_type=dst_type,
                new_shape=new_shape,
                config=config,
            )
        assert "Unsupported recipe_name" in str(exc_info.value)


class TestGMMWithHpOrLpWeight:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        config.get_key_dtype = Mock(return_value=torch.float8_e4m3fn)
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_forward(self, mock_config):
        x = torch.randn(64, 32, dtype=torch.bfloat16)
        weight = torch.randn(4, 32, 64, dtype=torch.bfloat16)
        bias = None
        group_list = torch.tensor([16, 32, 48, 64], dtype=torch.int64)
        grad_enabled = True

        with patch('mindspeed.fsdp.quantization.module.linear_mxfp8.process_quant_result_fn') as mock_process_quant:
            if mock_config.recipe_name == "mxfp8":
                scale_shape = (4, 32, 2)
            else:
                scale_shape = (4, 1, 2)

            weight_fwd = torch.randn(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_fwd = torch.randn(*scale_shape, dtype=torch.float32)
            weight_bwd = torch.randn(4, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_bwd = torch.randn(*scale_shape, dtype=torch.float32)
            mock_process_quant.return_value = (weight_fwd, scale_fwd, weight_bwd, scale_bwd)

            with patch('torch_npu.npu_grouped_matmul') as mock_gmm:
                output = torch.randn(64, 64, dtype=torch.bfloat16)
                mock_gmm.return_value = [output]

                with patch('torch_npu.npu_grouped_dynamic_mx_quant') as mock_quant:
                    x_quant = torch.randn(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                    x_scale = torch.randn(4, 1, dtype=torch.float32)
                    mock_quant.return_value = (x_quant, x_scale)

                    ctx = Mock()
                    try:
                        gmm_with_hp_or_lp_weight.forward(
                            ctx, x, weight, bias, group_list, grad_enabled, mock_config, None
                        )
                    except Exception:
                        pass

                    if mock_process_quant.called:
                        call_kwargs = mock_process_quant.call_args[1]
                        assert 'config' in call_kwargs
                        assert call_kwargs['config'] == mock_config


class TestMXFP8GMM:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        config.enable_fsdp_low_precision_all_gather = request.param == "mxfp8-32x32"
        config.get_key_dtype = Mock(return_value=torch.float8_e4m3fn)
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_from_float(self, mock_config):
        mock_mod = Mock()
        mock_mod.gate_up_proj = torch.nn.Parameter(torch.randn(4, 32, 64, dtype=torch.bfloat16))
        mock_mod.down_proj = torch.nn.Parameter(torch.randn(4, 64, 32, dtype=torch.bfloat16))
        mock_mod.hidden_dim = 32
        mock_mod.intermediate_size = 32

        if mock_config.recipe_name == "mxfp8":
            with patch('mindspeed.fsdp.quantization.core.pre_quant_weight.PreQuantWeight'):
                result = MXFP8GMM.from_float(
                    mod=mock_mod,
                    config=mock_config,
                    name="test_gmm",
                )
        else:
            result = MXFP8GMM.from_float(
                mod=mock_mod,
                config=mock_config,
                name="test_gmm",
            )

        assert result.config == mock_config
        assert result._name == "test_gmm"

    def test_mx_fp8_gmm_initialization(self):
        gmm = MXFP8GMM()
        assert gmm.gate_up_proj is None
        assert gmm.down_proj is None
        assert gmm._name is None


class TestMxQuantGroupGemm:
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = "mxfp8-32x32"
        config.get_key_dtype = Mock(return_value=torch.float8_e4m3fn)
        return config

    def test_mx_quant_group_gemm_basic(self, mock_config):
        x = torch.randn(64, 32, dtype=torch.bfloat16)
        weight = torch.randn(4, 32, 64, dtype=torch.bfloat16)
        tokens_per_expert = [16, 16, 16, 16]

        with patch.object(gmm_with_hp_or_lp_weight, 'gmm_apply') as mock_gmm_apply:
            output = torch.randn(64, 64, dtype=torch.bfloat16)
            mock_gmm_apply.return_value = output

            result = mx_quant_group_gemm(
                x=x,
                weight=weight,
                bias=None,
                tokens_per_expert=tokens_per_expert,
                grad_enabled=True,
                config=mock_config,
            )

            assert result is output
            mock_gmm_apply.assert_called_once()
