"""
W4A4 QAT core function unit tests.

Tests the W4A4 activation quantization operator and linear forward/backward
core logic. Helpers are self-contained to avoid torch_npu dependency.

Usage: pytest tests_extend/unit_tests/features/qat/test_w4a4_core_functions.py -v
"""

import pytest
import torch


# ============================================================
# Helper implementations (self-contained, mirror production logic)
# ============================================================

FP32_EXPONENT_BIAS = 127.0
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def w4a16_fake_quant(tensor, ebits, mbits, qdim=-1):
    """MXFP4 fake quantization core function."""
    emax = 2 ** (ebits - 1)
    max_norm = 2**emax * (2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
    tensor = tensor.unflatten(qdim, (-1, 32))
    shared_exp = torch.amax(tensor.abs(), dim=qdim, keepdim=True)
    mask = (shared_exp == 0).float()
    shared_exp = torch.floor(torch.log2(shared_exp + FP32_MIN_NORMAL * mask))
    mask = (tensor > -FP32_EXPONENT_BIAS).float()
    tensor = tensor * mask
    shared_exp = shared_exp - emax
    scale_emax = 2 ** (8.0 - 1.0) - 1

    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float('nan')), shared_exp)
    shared_exp = torch.where(shared_exp < -scale_emax, torch.full_like(shared_exp, -scale_emax), shared_exp)
    tensor = tensor / (2**shared_exp)
    mask = (tensor == 0).float()
    private_exp = torch.floor(torch.log2(tensor.abs() + mask))

    min_exp = -(2 ** (ebits - 1)) + 2
    private_exp = torch.maximum(private_exp, torch.tensor(min_exp, device=tensor.device))
    tensor = tensor / (2**private_exp) * (2 ** (mbits - 2))
    tensor_sign = torch.sign(tensor)
    tensor = tensor_sign * torch.floor(tensor.abs() + 0.5)
    tensor = tensor / (2 ** (mbits - 2)) * (2**private_exp)

    tensor = torch.clamp(tensor, -max_norm, max_norm)

    tensor = torch.where(torch.isinf(tensor), tensor, tensor)
    tensor = torch.where(torch.isnan(tensor), tensor, tensor)
    recovered_tensor = tensor * (2**shared_exp)

    recovered_tensor = recovered_tensor.flatten(qdim - 1, qdim)

    return recovered_tensor


class FakeQuantization(torch.autograd.Function):
    """MXFP4 fake quantization with STE backward (used for both weight and activation)."""

    @staticmethod
    def forward(ctx, fp32_tensor, block_size, transpose):
        dequant_tensor = w4a16_fake_quant(fp32_tensor, 2.0, 3.0, qdim=-1)
        return dequant_tensor.to(fp32_tensor.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None, None


fakequant_func = FakeQuantization.apply


def linear_w4a4_forward(input_, weight, bias=None):
    """W4A4 Linear forward: quantize both weight and input, then matmul."""
    weight_q = fakequant_func(weight, [1, 32], False)
    input_q = fakequant_func(input_, [1, 32], False)
    output = torch.matmul(input_q, weight_q.t())
    if bias is not None:
        output = output + bias
    return output


def linear_w4a4_backward(grad_output, input_, weight):
    """W4A4 Linear backward: dX = dY @ quant_weight, dW = dY^T @ quant_input."""
    weight_q = fakequant_func(weight, [1, 32], False)
    input_q = fakequant_func(input_, [1, 32], False)
    grad_input = grad_output.matmul(weight_q)
    grad_weight = grad_output.t().matmul(input_q)
    return grad_input, grad_weight


# ============================================================
# Helpers
# ============================================================


def make_linear_inputs():
    """Standard linear layer inputs: (8, 128) input, (128, 128) weight."""
    s, h = 8, 128
    return torch.randn(s, h), torch.randn(h, h)


# ============================================================
# Unit tests
# ============================================================


class TestW4A4FakeQuantization:
    """Tests for the activation fake-quantization operator."""

    @pytest.mark.parametrize("scale", [10.0, 0.01])
    def test_quant_validity(self, scale):
        """Output preserves shape/dtype and contains no NaN/Inf."""
        act = torch.randn(64, 128) * scale
        result = fakequant_func(act, [1, 32], False)
        assert result.shape == act.shape
        assert result.dtype == act.dtype
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_quant_deterministic(self):
        """Same input produces identical output across calls."""
        act = torch.randn(64, 128)
        r1 = fakequant_func(act, [1, 32], False)
        r2 = fakequant_func(act, [1, 32], False)
        assert torch.allclose(r1, r2)

    def test_quant_zero_tensor(self):
        """All-zero input remains zero."""
        act = torch.zeros(64, 128)
        result = fakequant_func(act, [1, 32], False)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_ste_backward(self):
        """STE: gradient passes through unchanged."""
        act = torch.randn(64, 128, requires_grad=True)
        result = fakequant_func(act, [1, 32], False)
        grad_output = torch.ones_like(result)
        result.backward(grad_output)
        assert act.grad is not None
        assert act.grad.shape == act.shape
        assert torch.allclose(act.grad, grad_output)


class TestW4A4Linear:
    """Tests for W4A4 linear forward/backward core logic."""

    def test_forward(self):
        """Forward produces correct shape with and without bias, no NaN."""
        input_t, weight = make_linear_inputs()
        s, h = input_t.shape
        output = linear_w4a4_forward(input_t, weight)
        assert output.shape == (s, h)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        output_b = linear_w4a4_forward(input_t, weight, torch.randn(h))
        assert output_b.shape == (s, h)

    def test_forward_differentiable(self):
        """Forward is differentiable w.r.t. both input and weight."""
        input_t, weight = make_linear_inputs()
        input_t.requires_grad_(True)
        weight.requires_grad_(True)
        output = linear_w4a4_forward(input_t, weight)
        output.sum().backward()
        assert input_t.grad is not None
        assert weight.grad is not None
        assert input_t.grad.shape == input_t.shape
        assert weight.grad.shape == weight.shape

    def test_backward_shapes(self):
        """Backward gradients have correct shapes and no NaN/Inf."""
        input_t, weight = make_linear_inputs()
        s, h = input_t.shape
        grad_output = torch.randn(s, h)
        grad_input, grad_weight = linear_w4a4_backward(grad_output, input_t, weight)
        assert grad_input.shape == (s, h)
        assert grad_weight.shape == (h, h)
        assert not torch.isnan(grad_input).any()
        assert not torch.isnan(grad_weight).any()

    def test_dw_uses_quantized_input(self):
        """dW uses quantized input — the key difference from W4A16.

        W4A16: dW = dY^T @ input          (raw input)
        W4A4:  dW = dY^T @ quant_input    (quantized input)
        """
        input_t, weight = make_linear_inputs()
        s, h = input_t.shape
        grad_output = torch.randn(s, h)

        _, grad_weight = linear_w4a4_backward(grad_output, input_t, weight)

        input_q = fakequant_func(input_t, [1, 32], False)
        expected = grad_output.t().matmul(input_q)
        assert torch.allclose(grad_weight, expected, atol=1e-6)

    def test_end_to_end(self):
        """Full forward + backward pipeline with non-zero gradient flow."""
        input_t, weight = make_linear_inputs()
        input_t.requires_grad_(True)
        weight.requires_grad_(True)
        target = torch.randn(*input_t.shape)

        output = linear_w4a4_forward(input_t, weight)
        loss = (output - target).pow(2).sum()
        loss.backward()

        assert weight.grad.abs().sum().item() > 0
        assert input_t.grad.abs().sum().item() > 0
