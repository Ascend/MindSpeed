import sys
import pytest
import torch
import torch_npu

sys.argv.append('--use-flash-attn')
from mindspeed import megatron_adaptor # noqa

from megatron.training.global_vars import set_args # noqa
from megatron.training.arguments import parse_args # noqa
from megatron.core import mpu # noqa
from megatron.core.transformer.transformer_config import TransformerConfig # noqa
from megatron.core.transformer.dot_product_attention import DotProductAttention # noqa

from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ulysses,
                                             get_context_parallel_group_for_hybrid_ring,
                                             get_context_parallel_for_hybrid_ring_world_size,
                                             get_context_parallel_for_hybrid_ring_rank,
                                             get_context_parallel_for_hybrid_ring_global_ranks)
from mindspeed.core.transformer.flash_attention.generate_mask.generate_mask import get_attention_mask, set_attention_mask

from tests_extend.commons import set_random_seed, initialize_model_parallel # noqa
from tests_extend.unit_tests.common import DistributedTest # noqa

sys.argv.remove('--use-flash-attn')


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def run_dot_product_att(cp_size, bs, seq_len, dtype):
    from megatron.core.transformer.enums import AttnMaskType
    args = parse_args(None, True)
    args.context_parallel_size = cp_size
    args.context_parallel_algo = 'megatron_cp_algo' if cp_size > 1 else None
    args.use_flash_attn = True
    args.micro_batch_size = bs
    args.seq_length = seq_len

    set_args(args)
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)
    # clear global attn mask set by last test case
    set_attention_mask(None)

    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True)
    attn = DotProductAttention(
        config=config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type='self')

    b, n, s, d = bs, 4, seq_len, 8

    q = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)

    # global attn mask will be generated at DotProductAttention forward wrapper
    out = attn(q, k, v, None, None, None, None)
    assert isinstance(out, torch.Tensor)


class TestDotProductAttn(DistributedTest):

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_dot_product_att_bf16(self):
        run_dot_product_att(1, 2, 256, torch.bfloat16)

    def test_dot_product_att_bf16_cp(self):
        run_dot_product_att(2, 2, 256, torch.bfloat16)
