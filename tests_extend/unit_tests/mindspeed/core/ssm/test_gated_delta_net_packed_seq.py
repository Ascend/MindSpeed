# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import pytest
import torch

try:
    import torch_npu
    HAVE_TORCH_NPU = True
except ImportError:
    HAVE_TORCH_NPU = False

import torch.distributed as dist
import torch.nn.functional as F
try:
    from mindspeed import megatron_adaptor
except ImportError:
    megatron_adaptor = None
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
try:
    from megatron.core.process_groups_config import ProcessGroupCollection
except ImportError:
    ProcessGroupCollection = None
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

try:
    from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet, _unpack_sequence
    HAVE_GATED_DELTA_NET = True
except ImportError:
    GatedDeltaNet = None
    _unpack_sequence = None
    HAVE_GATED_DELTA_NET = False

try:
    from mindspeed.core.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule, torch_chunk_gated_delta_rule
    HAVE_CHUNK_GATED_DELTA_RULE = True
except ImportError:
    chunk_gated_delta_rule = None
    torch_chunk_gated_delta_rule = None
    HAVE_CHUNK_GATED_DELTA_RULE = False

try:
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

try:
    from causal_conv1d import causal_conv1d

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False

try:
    import fla_npu  # noqa: F401
    from mindspeed.core.ssm.ops.npu_causal_conv1d import causal_conv1d as npu_causal_conv1d

    HAVE_NPU_CAUSAL_CONV1D = True
except ImportError:
    npu_causal_conv1d = None
    HAVE_NPU_CAUSAL_CONV1D = False

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    from mindspeed.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
except ImportError:
    GatedDeltaNetSubmodules = None


def make_test_packed_seq_params(sequence_length=None, cu_seqlens=None):
    if cu_seqlens is None:
        assert sequence_length is not None
        cu_seqlens = [0, 6, 19, 22, sequence_length]
    cu_seqlens = torch.IntTensor(cu_seqlens).npu()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


def initialize_test_environment():
    from tests_extend.commons import initialize_model_parallel

    initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)


def small_causal_conv1d(qkv, conv1d_weight, conv1d_bias, activation):
    seq_len = qkv.shape[1]
    qkv = qkv.transpose(1, 2).contiguous()
    conv_out = F.conv1d(
        input=qkv,
        weight=conv1d_weight,
        bias=conv1d_bias,
        stride=1,
        padding=conv1d_weight.shape[-1] - 1,
        dilation=1,
        groups=conv1d_weight.shape[0],
    )
    conv_out = conv_out[..., :seq_len]
    if activation in ("silu", "swish"):
        conv_out = F.silu(conv_out)
    elif activation is not None:
        raise NotImplementedError("activation must be None, silu, or swish")
    return conv_out.transpose(1, 2).contiguous()


def small_causal_conv1d_packed(qkv, conv1d_weight, conv1d_bias, activation, cu_seqlens):
    outputs = []
    cu_seqlens = cu_seqlens.detach().cpu().tolist()
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        outputs.append(small_causal_conv1d(qkv[:, start:end, :], conv1d_weight, conv1d_bias, activation))
    return torch.cat(outputs, dim=1)


@pytest.mark.skipif(not HAVE_GATED_DELTA_NET, reason="GatedDeltaNet not available")
class TestUnpackSequence:
    def test_unpack_basic(self):
        x = torch.arange(12).reshape(6, 2)
        cu_seqlens = torch.IntTensor([0, 2, 5, 6])
        result = _unpack_sequence(x, cu_seqlens, dim=0)
        assert len(result) == 3
        assert result[0].shape == (2, 2)
        assert result[1].shape == (3, 2)
        assert result[2].shape == (1, 2)
        assert torch.equal(result[0], x[0:2])
        assert torch.equal(result[1], x[2:5])
        assert torch.equal(result[2], x[5:6])

    def test_unpack_dim1(self):
        x = torch.arange(12).reshape(2, 6)
        cu_seqlens = torch.IntTensor([0, 2, 5, 6])
        result = _unpack_sequence(x, cu_seqlens, dim=1)
        assert len(result) == 3
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 3)
        assert result[2].shape == (2, 1)

    def test_unpack_single_sequence(self):
        x = torch.arange(6).reshape(3, 2)
        cu_seqlens = torch.IntTensor([0, 3])
        result = _unpack_sequence(x, cu_seqlens, dim=0)
        assert len(result) == 1
        assert torch.equal(result[0], x)

    def test_unpack_3d_tensor(self):
        x = torch.arange(24).reshape(6, 1, 4)
        cu_seqlens = torch.IntTensor([0, 3, 6])
        result = _unpack_sequence(x, cu_seqlens, dim=0)
        assert len(result) == 2
        assert result[0].shape == (3, 1, 4)
        assert result[1].shape == (3, 1, 4)

    def test_unpack_empty_raises(self):
        x = torch.arange(6).reshape(3, 2)
        cu_seqlens = torch.IntTensor([0, 0, 0])
        result = _unpack_sequence(x, cu_seqlens, dim=0)
        assert len(result) == 2
        assert result[0].shape == (0, 2)
        assert result[1].shape == (0, 2)


class TestNpuCausalConv1dNonPackedEquivalence:
    @pytest.mark.skipif(not HAVE_NPU_CAUSAL_CONV1D, reason="npu_causal_conv1d not available")
    @pytest.mark.skipif(not HAVE_TORCH_NPU or not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("activation", [None, "silu", "swish"])
    @pytest.mark.parametrize(
        "batch, seq_len, dim, width",
        [
            (1, 8, 16, 2),
            (2, 16, 16, 3),
            (4, 8, 32, 4),
        ],
    )
    def test_non_packed_small_op_matches_fused_op(self, activation, batch, seq_len, dim, width):
        torch.manual_seed(42)
        torch.npu.manual_seed(42)
        qkv = (torch.randn(batch, seq_len, dim, dtype=torch.float16, device="npu") * 0.1).contiguous()
        conv1d_weight = (
            torch.randn(dim, 1, width, dtype=torch.float16, device="npu") * 0.1
        ).contiguous()
        conv1d_bias = (torch.randn(dim, dtype=torch.float16, device="npu") * 0.1).contiguous()

        small_out = small_causal_conv1d(qkv, conv1d_weight, conv1d_bias, activation)
        fused_out, final_state = npu_causal_conv1d(
            x=qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
        )

        assert final_state is None
        torch.testing.assert_close(small_out.float(), fused_out.float(), atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not HAVE_NPU_CAUSAL_CONV1D, reason="npu_causal_conv1d not available")
    @pytest.mark.skipif(not HAVE_TORCH_NPU or not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("activation", [None, "silu", "swish"])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_non_packed_backward_matches_small_op(self, activation, with_bias):
        torch.manual_seed(42)
        torch.npu.manual_seed(42)
        batch, seq_len, dim, width = 2, 16, 16, 4
        qkv = (
            torch.randn(batch, seq_len, dim, dtype=torch.float16, device="npu") * 0.1
        ).contiguous().requires_grad_()
        conv1d_weight = (
            torch.randn(dim, 1, width, dtype=torch.float16, device="npu") * 0.1
        ).contiguous().requires_grad_()
        conv1d_bias = None
        if with_bias:
            conv1d_bias = (
                torch.randn(dim, dtype=torch.float16, device="npu") * 0.1
            ).contiguous().requires_grad_()

        qkv_ref = qkv.detach().clone().requires_grad_()
        weight_ref = conv1d_weight.detach().clone().requires_grad_()
        bias_ref = None
        if with_bias:
            bias_ref = conv1d_bias.detach().clone().requires_grad_()

        fused_out, final_state = npu_causal_conv1d(
            x=qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        small_out = small_causal_conv1d(qkv_ref, weight_ref, bias_ref, activation)

        assert final_state is None
        torch.testing.assert_close(small_out.float(), fused_out.float(), atol=1e-4, rtol=1e-4)

        grad_out = torch.randn_like(fused_out)
        fused_out.backward(grad_out)
        small_out.backward(grad_out)

        torch.testing.assert_close(qkv.grad.float(), qkv_ref.grad.float(), atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(
            conv1d_weight.grad.float(), weight_ref.grad.float(), atol=5e-3, rtol=5e-3
        )
        if with_bias:
            torch.testing.assert_close(
                conv1d_bias.grad.float(), bias_ref.grad.float(), atol=5e-3, rtol=5e-3
            )

    @pytest.mark.skipif(not HAVE_NPU_CAUSAL_CONV1D, reason="npu_causal_conv1d not available")
    @pytest.mark.skipif(not HAVE_TORCH_NPU or not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("activation", [None, "silu", "swish"])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_packed_backward_matches_segmented_small_op(self, activation, with_bias):
        torch.manual_seed(42)
        torch.npu.manual_seed(42)
        batch, total_seq_len, dim, width = 1, 20, 16, 4
        cu_seqlens = torch.IntTensor([0, 5, 13, total_seq_len]).npu()
        qkv = (
            torch.randn(batch, total_seq_len, dim, dtype=torch.float16, device="npu") * 0.1
        ).contiguous().requires_grad_()
        conv1d_weight = (
            torch.randn(dim, 1, width, dtype=torch.float16, device="npu") * 0.1
        ).contiguous().requires_grad_()
        conv1d_bias = None
        if with_bias:
            conv1d_bias = (
                torch.randn(dim, dtype=torch.float16, device="npu") * 0.1
            ).contiguous().requires_grad_()

        qkv_ref = qkv.detach().clone().requires_grad_()
        weight_ref = conv1d_weight.detach().clone().requires_grad_()
        bias_ref = None
        if with_bias:
            bias_ref = conv1d_bias.detach().clone().requires_grad_()

        fused_out, final_state = npu_causal_conv1d(
            x=qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        small_out = small_causal_conv1d_packed(
            qkv_ref, weight_ref, bias_ref, activation, cu_seqlens
        )

        assert final_state is None
        torch.testing.assert_close(small_out.float(), fused_out.float(), atol=1e-4, rtol=1e-4)

        grad_out = torch.randn_like(fused_out)
        fused_out.backward(grad_out)
        small_out.backward(grad_out)

        torch.testing.assert_close(qkv.grad.float(), qkv_ref.grad.float(), atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(
            conv1d_weight.grad.float(), weight_ref.grad.float(), atol=5e-3, rtol=5e-3
        )
        if with_bias:
            torch.testing.assert_close(
                conv1d_bias.grad.float(), bias_ref.grad.float(), atol=5e-3, rtol=5e-3
            )

    @pytest.mark.skipif(not HAVE_NPU_CAUSAL_CONV1D, reason="npu_causal_conv1d not available")
    @pytest.mark.skipif(not HAVE_TORCH_NPU or not torch.npu.is_available(), reason="NPU not available")
    @pytest.mark.parametrize("activation", [None, "silu", "swish"])
    def test_packed_backward_merged_matches_segmented(self, activation):
        # Verify the merged zero-padded conv backward path matches the per-segment reference.
        torch.manual_seed(42)
        torch.npu.manual_seed(42)
        batch, total_seq_len, dim, width = 1, 64, 32, 4
        cu_seqlens = torch.IntTensor([0, 8, 20, 35, 50, total_seq_len]).npu()
        qkv = (
            torch.randn(batch, total_seq_len, dim, dtype=torch.float16, device="npu") * 0.1
        ).contiguous().requires_grad_()
        conv1d_weight = (
            torch.randn(dim, 1, width, dtype=torch.float16, device="npu") * 0.1
        ).contiguous().requires_grad_()

        qkv_ref = qkv.detach().clone().requires_grad_()
        weight_ref = conv1d_weight.detach().clone().requires_grad_()

        fused_out, _ = npu_causal_conv1d(
            x=qkv,
            weight=conv1d_weight.squeeze(1),
            bias=None,
            activation=activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        small_out = small_causal_conv1d_packed(qkv_ref, weight_ref, None, activation, cu_seqlens)

        torch.testing.assert_close(small_out.float(), fused_out.float(), atol=1e-4, rtol=1e-4)

        grad_out = torch.randn_like(fused_out)
        fused_out.backward(grad_out)
        small_out.backward(grad_out)

        torch.testing.assert_close(qkv.grad.float(), qkv_ref.grad.float(), atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(
            conv1d_weight.grad.float(), weight_ref.grad.float(), atol=5e-3, rtol=5e-3
        )


class TestTorchChunkGatedDeltaRuleCuSeqlens:
    @pytest.mark.skipif(not HAVE_CHUNK_GATED_DELTA_RULE, reason="chunk_gated_delta_rule not available")
    def test_cu_seqlens_batch_size_assert(self):
        B, T, H, K, V = 2, 64, 4, 32, 32
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()
        cu_seqlens = torch.IntTensor([0, 32, 64])
        with pytest.raises(AssertionError, match="batch size is expected to be 1"):
            torch_chunk_gated_delta_rule(
                q, k, v, g, beta,
                cu_seqlens=cu_seqlens,
            )

    @pytest.mark.skipif(not HAVE_CHUNK_GATED_DELTA_RULE, reason="chunk_gated_delta_rule not available")
    def test_cu_seqlens_correctness(self):
        B, T, H, K, V = 1, 128, 4, 16, 16
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()

        cu_seqlens = torch.IntTensor([0, 32, 72, 128])

        o_packed, _ = torch_chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
        )

        o_seg0, _ = torch_chunk_gated_delta_rule(
            q[:, :32], k[:, :32], v[:, :32], g[:, :32], beta[:, :32],
        )
        o_seg1, _ = torch_chunk_gated_delta_rule(
            q[:, 32:72], k[:, 32:72], v[:, 32:72], g[:, 32:72], beta[:, 32:72],
        )
        o_seg2, _ = torch_chunk_gated_delta_rule(
            q[:, 72:128], k[:, 72:128], v[:, 72:128], g[:, 72:128], beta[:, 72:128],
        )
        o_ref = torch.cat([o_seg0, o_seg1, o_seg2], dim=1)

        torch.testing.assert_close(o_packed.float(), o_ref.float(), atol=5e-2, rtol=5e-2)

    @pytest.mark.skipif(not HAVE_CHUNK_GATED_DELTA_RULE, reason="chunk_gated_delta_rule not available")
    def test_cu_seqlens_with_initial_state(self):
        B, T, H, K, V = 1, 64, 2, 8, 8
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()

        cu_seqlens = torch.IntTensor([0, 32, 64])
        num_seqs = len(cu_seqlens) - 1
        h0 = torch.randn(num_seqs, H, K, V, dtype=torch.bfloat16)

        o_packed, _ = torch_chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens,
        )

        o_seg0, _ = torch_chunk_gated_delta_rule(
            q[:, :32], k[:, :32], v[:, :32], g[:, :32], beta[:, :32],
            initial_state=h0[0:1],
        )
        o_seg1, _ = torch_chunk_gated_delta_rule(
            q[:, 32:], k[:, 32:], v[:, 32:], g[:, 32:], beta[:, 32:],
            initial_state=h0[1:2],
        )
        o_ref = torch.cat([o_seg0, o_seg1], dim=1)

        torch.testing.assert_close(o_packed.float(), o_ref.float(), atol=5e-2, rtol=5e-2)

    @pytest.mark.skipif(not HAVE_CHUNK_GATED_DELTA_RULE, reason="chunk_gated_delta_rule not available")
    def test_cu_seqlens_output_final_state(self):
        B, T, H, K, V = 1, 64, 2, 8, 8
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()

        cu_seqlens = torch.IntTensor([0, 32, 64])
        num_seqs = len(cu_seqlens) - 1

        o_packed, ht_packed = torch_chunk_gated_delta_rule(
            q, k, v, g, beta,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

        assert ht_packed is not None
        assert ht_packed.shape == (num_seqs, H, K, V)

        o_seg0, ht0 = torch_chunk_gated_delta_rule(
            q[:, :32], k[:, :32], v[:, :32], g[:, :32], beta[:, :32],
            output_final_state=True,
        )
        o_seg1, ht1 = torch_chunk_gated_delta_rule(
            q[:, 32:], k[:, 32:], v[:, 32:], g[:, 32:], beta[:, 32:],
            output_final_state=True,
        )
        ht_ref = torch.stack([ht0, ht1], dim=0).squeeze(1)

        torch.testing.assert_close(ht_packed.float(), ht_ref.float(), atol=5e-2, rtol=5e-2)

    @pytest.mark.skipif(not HAVE_CHUNK_GATED_DELTA_RULE, reason="chunk_gated_delta_rule not available")
    def test_cu_seqlens_none_works(self):
        B, T, H, K, V = 1, 64, 4, 32, 32
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()
        out, _ = torch_chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=None,
        )
        assert out.shape == (B, T, H, V)


class TestChunkGatedDeltaRuleCuSeqlens:
    @pytest.mark.skipif(not HAVE_CHUNK_GATED_DELTA_RULE, reason="chunk_gated_delta_rule not available")
    @pytest.mark.skipif(not HAVE_TORCH_NPU or not torch.npu.is_available(), reason="NPU not available")
    def test_packed_sequence_output(self):
        B, T, H, K, V = 1, 128, 4, 32, 32
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='npu')
        k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='npu'), p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='npu')
        beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='npu').sigmoid()
        g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='npu'))

        cu_seqlens = torch.IntTensor([0, 32, 64, 96, 128]).npu()

        o_packed, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
        )

        o_unpacked, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=None,
        )

        assert o_packed.shape == o_unpacked.shape
        assert o_packed.shape == (B, T, H, V)

    @pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
    def test_packed_sequence_correctness(self):
        B, T, H, K, V = 1, 64, 2, 16, 16
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='npu')
        k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='npu'), p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='npu')
        beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='npu').sigmoid()
        g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='npu'))

        cu_seqlens = torch.IntTensor([0, 32, 64]).npu()

        o_packed, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
        )

        q_0 = q[:, :32]
        k_0 = k[:, :32]
        v_0 = v[:, :32]
        beta_0 = beta[:, :32]
        g_0 = g[:, :32]
        o_0, _ = chunk_gated_delta_rule(q_0, k_0, v_0, g_0, beta_0)

        q_1 = q[:, 32:]
        k_1 = k[:, 32:]
        v_1 = v[:, 32:]
        beta_1 = beta[:, 32:]
        g_1 = g[:, 32:]
        o_1, _ = chunk_gated_delta_rule(q_1, k_1, v_1, g_1, beta_1)

        o_ref = torch.cat([o_0, o_1], dim=1)

        torch.testing.assert_close(o_packed, o_ref, atol=1e-2, rtol=1e-2)
