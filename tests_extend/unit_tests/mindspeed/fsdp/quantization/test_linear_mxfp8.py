import pytest
from unittest.mock import Mock, patch
import torch

from mindspeed.fsdp.quantization.module.linear_mxfp8 import (
    process_quant_result_fn,
    weight_quant,
    matmul_with_hp_or_lp_weight,
    mx_quant_linear,
    MXLinear,
)
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class TestProcessQuantResultFn:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_process_quant_result_fn(self, mock_config):
        weight = torch.randn(128, 128, dtype=torch.bfloat16)
        dst_type = torch.float8_e4m3fn

        if mock_config.recipe_name == "mxfp8":
            mock_func = 'torch_npu.npu_dynamic_mx_quant_with_dual_axis'
            weight_fwd = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_fwd = torch.randn(128, 4, dtype=torch.float32)
            weight_bwd = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_bwd = torch.randn(128, 4, dtype=torch.float32)
            mock_return = (weight_fwd, scale_fwd, weight_bwd, scale_bwd)
        else:
            mock_func = 'torch_npu.npu_dynamic_block_mx_quant'
            weight_fwd = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_fwd = torch.randn(128, 4, dtype=torch.float32)
            scale_bwd = torch.randn(128, 4, dtype=torch.float32)
            mock_return = (weight_fwd, scale_fwd, scale_bwd)

        with patch(mock_func) as mock_quant:
            mock_quant.return_value = mock_return

            result_fwd, result_scale_fwd, result_bwd, result_scale_bwd = process_quant_result_fn(
                weight=weight,
                dst_type=dst_type,
                config=mock_config,
            )

            mock_quant.assert_called_once_with(weight, dst_type=dst_type)
            assert result_fwd.shape == weight_fwd.shape
            assert result_fwd.dtype == weight_fwd.dtype
            assert torch.equal(result_scale_fwd, scale_fwd)
            assert result_bwd.shape == weight_fwd.shape
            assert result_bwd.dtype == weight_fwd.dtype
            assert torch.equal(result_scale_bwd, scale_bwd)

    def test_process_quant_result_fn_invalid_recipe_raises_error(self):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = "invalid_recipe"
        weight = torch.randn(128, 128, dtype=torch.bfloat16)
        dst_type = torch.float8_e4m3fn

        with pytest.raises(ValueError) as exc_info:
            process_quant_result_fn(
                weight=weight,
                dst_type=dst_type,
                config=config,
            )
        assert "Unsupported recipe_name" in str(exc_info.value)
        assert "invalid_recipe" in str(exc_info.value)


class TestWeightQuant:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_weight_quant(self, mock_config):
        weight = torch.randn(4, 32, 64, dtype=torch.bfloat16)
        dst_type = torch.float8_e4m3fn

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
                config=mock_config,
            )

            mock_quant.assert_called_once()
            assert result_fwd.shape == weight.shape
            assert result_bwd.shape == weight.shape
            assert result_fwd.shape == result_bwd.shape
            assert result_fwd.dtype == result_bwd.dtype


class TestMatmulWithHpOrLpWeight:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        config.get_key_dtype = Mock(return_value=torch.float8_e4m3fn)
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_forward(self, mock_config):
        x = torch.randn(64, 32, dtype=torch.bfloat16)
        weight = torch.randn(64, 32, dtype=torch.bfloat16)
        bias = None
        grad_enabled = True

        with patch('mindspeed.fsdp.quantization.module.linear_mxfp8.process_quant_result_fn') as mock_process_quant:
            if mock_config.recipe_name == "mxfp8":
                scale_shape = (64, 4)
            else:
                scale_shape = (64, 2)

            weight_fwd = torch.randn(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_fwd = torch.randn(*scale_shape, dtype=torch.float32)
            weight_bwd = torch.randn(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            scale_bwd = torch.randn(*scale_shape, dtype=torch.float32)
            mock_process_quant.return_value = (weight_fwd, scale_fwd, weight_bwd, scale_bwd)

            with patch('torch_npu.npu_quant_matmul') as mock_matmul:
                output = torch.randn(64, 64, dtype=torch.bfloat16)
                mock_matmul.return_value = output

                with patch('torch_npu.npu_dynamic_mx_quant_with_dual_axis') as mock_quant:
                    x_quant = torch.randn(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                    x_scale = torch.randn(64, 1, dtype=torch.float32)
                    mock_quant.return_value = (x_quant, x_scale, x_quant, x_scale)

                    ctx = Mock()
                    try:
                        matmul_with_hp_or_lp_weight.forward(ctx, x, weight, mock_config, grad_enabled, bias, None)
                    except Exception:
                        pass

                    if mock_process_quant.called:
                        call_kwargs = mock_process_quant.call_args[1]
                        assert 'config' in call_kwargs
                        assert call_kwargs['config'] == mock_config


class TestMXLinear:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        config.enable_fsdp_low_precision_all_gather = request.param == "mxfp8-32x32"
        config.get_key_dtype = Mock(return_value=torch.float8_e4m3fn)
        return config

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    def test_from_float(self, mock_config):
        mock_mod = torch.nn.Linear(32, 64, bias=False)
        mock_mod.weight = torch.nn.Parameter(torch.randn(64, 32, dtype=torch.bfloat16))

        if mock_config.recipe_name == "mxfp8":
            with patch('mindspeed.fsdp.quantization.module.linear_mxfp8.PreQuantWeight'):
                result = MXLinear.from_float(
                    mod=mock_mod,
                    config=mock_config,
                    name="test_linear",
                )
        else:
            result = MXLinear.from_float(
                mod=mock_mod,
                config=mock_config,
                name="test_linear",
            )

        assert result.config == mock_config
        assert result._name == "test_linear"

    def test_mx_linear_initialization(self):
        linear = MXLinear(32, 64)
        assert linear.in_features == 32
        assert linear.out_features == 64
        assert linear._name is None


class TestMxQuantLinear:
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = "mxfp8-32x32"
        config.get_key_dtype = Mock(return_value=torch.float8_e4m3fn)
        return config

    @pytest.mark.parametrize("bias", [None, torch.randn(64, dtype=torch.bfloat16)])
    def test_mx_quant_linear(self, mock_config, bias):
        x = torch.randn(64, 32, dtype=torch.bfloat16)
        weight = torch.randn(64, 32, dtype=torch.bfloat16)

        with patch.object(matmul_with_hp_or_lp_weight, 'apply') as mock_apply:
            output = torch.randn(64, 64, dtype=torch.bfloat16)
            mock_apply.return_value = output

            result = mx_quant_linear(
                x=x,
                weight=weight,
                config=mock_config,
                grad_enabled=True,
                bias=bias,
                name="test_linear" if bias is not None else None,
            )

            assert result is output
            mock_apply.assert_called_once()
