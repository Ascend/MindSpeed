import pytest
import torch
import torch_npu
from mindspeed.ops.gmm_w4a8 import npu_gmm

if torch_npu.npu.is_available():
    DEVICE_NAME = torch_npu.npu.get_device_name()
else:
    DEVICE_NAME = None


@pytest.mark.skipif(
    not (DEVICE_NAME and DEVICE_NAME.startswith("Ascend950DT")), reason="Test only runs on Ascend950DT series"
)
class TestW4A8GMMFunction:
    @pytest.fixture
    def forward_inputs(self):
        num_experts = 32
        tokens_per_expert = 192
        hidden = 2048
        ffn = 4096
        batch = num_experts * tokens_per_expert

        x = torch.randn(batch, hidden, dtype=torch.bfloat16).npu()
        weight = torch.randn(num_experts, hidden, ffn, dtype=torch.bfloat16).npu()
        group_list = (
            torch.Tensor(
                [
                    192,
                    384,
                    576,
                    768,
                    960,
                    1152,
                    1344,
                    1536,
                    1728,
                    1920,
                    2112,
                    2304,
                    2496,
                    2688,
                    2880,
                    3072,
                    3264,
                    3456,
                    3648,
                    3840,
                    4032,
                    4224,
                    4416,
                    4608,
                    4800,
                    4992,
                    5184,
                    5376,
                    5568,
                    5760,
                    5952,
                    6144,
                ]
            )
            .to(torch.int64)
            .npu()
        )
        return x, weight, group_list

    def test_forward_block32(self, forward_inputs):
        x, weight, group_list = forward_inputs
        out = npu_gmm(x, weight, group_list=group_list, block_size=32)
        assert out.shape == (x.shape[0], weight.shape[-1])
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_block128(self, forward_inputs):
        x, weight, group_list = forward_inputs
        out = npu_gmm(x, weight, group_list=group_list, block_size=128)
        assert out.shape == (x.shape[0], weight.shape[-1])
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_backward_block32(self, forward_inputs):
        x, weight, group_list = forward_inputs
        x.requires_grad_(True)
        weight.requires_grad_(True)

        out = npu_gmm(x, weight, group_list=group_list, block_size=32)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert x.grad is not None
        assert weight.grad is not None
        assert x.grad.shape == x.shape
        assert weight.grad.shape == weight.shape
        assert not torch.isnan(x.grad).any(), "x.grad contains NaN"
        assert not torch.isnan(weight.grad).any(), "weight.grad contains NaN"
        assert not torch.isinf(x.grad).any(), "x.grad contains Inf"
        assert not torch.isinf(weight.grad).any(), "weight.grad contains Inf"

    def test_backward_block128(self, forward_inputs):
        x, weight, group_list = forward_inputs
        x.requires_grad_(True)
        weight.requires_grad_(True)

        out = npu_gmm(x, weight, group_list=group_list, block_size=128)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert x.grad is not None
        assert weight.grad is not None
        assert x.grad.shape == x.shape
        assert weight.grad.shape == weight.shape
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(weight.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isinf(weight.grad).any()

    def test_error_bias_requires_grad(self, forward_inputs):
        x, weight, group_list = forward_inputs
        bias = torch.randn(weight.shape[-1], device="npu", requires_grad=True)
        with pytest.raises(ValueError, match="Bias is not supported to compute gradient!"):
            npu_gmm(x, weight, bias=bias, group_list=group_list, block_size=32)
