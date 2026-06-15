import pytest
from unittest.mock import Mock, patch
import torch

from mindspeed.fsdp.quantization.core.pre_quant_weight import PreQuantWeight
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class TestPreQuantWeight:
    @pytest.fixture
    def mock_config(self, request):
        config = Mock(spec=QuantizeConfig)
        config.recipe_name = request.param
        config.enable_fsdp_low_precision_all_gather = True
        config.fsdp_low_precision_all_gather_mode = "all"
        return config

    @pytest.fixture
    def mock_quantizer(self):
        def quantizer(weight):
            weight_fwd = torch.randn_like(weight, dtype=torch.float8_e4m3fn)
            scale_fwd = torch.randn(weight.shape[0], weight.shape[1] // 32, dtype=torch.float32)
            weight_bwd = torch.randn_like(weight, dtype=torch.float8_e4m3fn)
            scale_bwd = torch.randn(weight.shape[0], weight.shape[1] // 32, dtype=torch.float32)
            return weight_fwd, scale_fwd, weight_bwd, scale_bwd

        return quantizer

    @pytest.mark.parametrize("mock_config", ["mxfp8", "mxfp8-32x32"], indirect=True)
    @pytest.mark.parametrize("world_size", [1, 2])
    def test_fsdp_pre_all_gather(self, mock_config, mock_quantizer, world_size):
        weight = torch.randn(128, 128, dtype=torch.bfloat16)
        pre_quant_weight = PreQuantWeight(
            tensor=weight,
            quantizer=mock_quantizer,
            config=mock_config,
            dtype=torch.bfloat16,
            name="test_weight",
        )

        with patch('mindspeed.fsdp.quantization.core.pre_quant_weight._get_module_fsdp_state') as mock_get_state:
            mock_fsdp_state = Mock()
            mock_fsdp_state._fsdp_param_group._training_state = Mock()
            mock_get_state.return_value = mock_fsdp_state

            with patch('mindspeed.fsdp.quantization.core.pre_quant_weight.cached_quant') as mock_cached_quant:
                weight_fwd = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                scale_fwd = torch.randn(128, 4, dtype=torch.float32)
                weight_bwd = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                scale_bwd = torch.randn(128, 4, dtype=torch.float32)
                mock_cached_quant.return_value = (weight_fwd, scale_fwd, weight_bwd, scale_bwd)

                mock_mesh = Mock()
                mock_mesh.size.return_value = world_size

                sharded_tensors, metadata = pre_quant_weight.fsdp_pre_all_gather(
                    mesh=mock_mesh if world_size > 1 else None,
                    orig_size=weight.shape,
                    contiguous_orig_stride=weight.stride(),
                    module=None,
                    mp_policy=None,
                )

                if world_size == 1:
                    expected_len = 2 if mock_config.recipe_name == "mxfp8" else 1
                    assert len(sharded_tensors) == expected_len
                else:
                    expected_len = 4 if mock_config.recipe_name == "mxfp8" else 3
                    assert len(sharded_tensors) == expected_len

                assert sharded_tensors[0].shape == (128, 128)
                fwd_usage, bwd_usage, _, _ = metadata
                assert fwd_usage is True
                assert bwd_usage is True

    @pytest.mark.parametrize("mock_config", ["mxfp8-32x32"], indirect=True)
    @pytest.mark.parametrize("gather_len", [1, 3])
    def test_fsdp_post_all_gather(self, mock_config, mock_quantizer, gather_len):
        weight = torch.randn(128, 128, dtype=torch.bfloat16)
        pre_quant_weight = PreQuantWeight(
            tensor=weight,
            quantizer=mock_quantizer,
            config=mock_config,
            dtype=torch.bfloat16,
            name="test_weight",
        )

        weight_gathered = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_fwd = torch.randn(128, 4, dtype=torch.float32)
        scale_bwd = torch.randn(128, 4, dtype=torch.float32)

        if gather_len == 1:
            all_gather_outputs = (weight_gathered,)
        else:
            all_gather_outputs = (weight_gathered, scale_fwd, scale_bwd)

        metadata = (True, True, scale_fwd, scale_bwd)

        result, _ = pre_quant_weight.fsdp_post_all_gather(
            all_gather_outputs=all_gather_outputs,
            metadata=metadata,
            param_dtype=torch.bfloat16,
        )

        assert result._weight_fwd is not None
        assert result._weight_bwd is not None
        assert result._scale_fwd is not None
        assert result._scale_bwd is not None
        assert result._weight_fwd.shape == weight_gathered.shape
        assert result._weight_fwd.dtype == weight_gathered.dtype
        assert result._weight_bwd.shape == weight_gathered.shape
        assert result._weight_bwd.dtype == weight_gathered.dtype
        assert torch.equal(result._scale_fwd, scale_fwd)
        assert torch.equal(result._scale_bwd, scale_bwd)

    @pytest.mark.parametrize("mock_config", ["mxfp8-32x32"], indirect=True)
    def test_fsdp_post_all_gather_invalid_len_raises_error(self, mock_config, mock_quantizer):
        weight = torch.randn(128, 128, dtype=torch.bfloat16)
        pre_quant_weight = PreQuantWeight(
            tensor=weight,
            quantizer=mock_quantizer,
            config=mock_config,
            dtype=torch.bfloat16,
            name="test_weight",
        )

        all_gather_outputs = (torch.randn(128, 128), torch.randn(128, 4))
        metadata = (True, True, torch.randn(128, 4), torch.randn(128, 4))

        with pytest.raises(ValueError) as exc_info:
            pre_quant_weight.fsdp_post_all_gather(
                all_gather_outputs=all_gather_outputs,
                metadata=metadata,
                param_dtype=torch.bfloat16,
            )
        assert "Unexpected gather outputs length for mxfp8-32x32 quant" in str(exc_info.value)
