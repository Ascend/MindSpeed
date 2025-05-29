# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import sys

# Setting sys.argv is mainly to ensure that --context-parallel-size is not None, so that the code block (which will be executed
# after determining that context-parallel-size is not None) will be executed in megatron_adaptor.
original_argv = sys.argv.copy()
sys.argv = [
    sys.argv[0],
    '--use-flash-attn',
    '--context-parallel-algo', 'hybrid_cp_algo',
    '--context-parallel-size', '2',
    '--num-layers', '24',
    '--hidden-size', '8',
    '--ffn-hidden-size', '8',
    '--num-attention-heads', '8',
    '--tokenizer-type', 'Llama2Tokenizer',
    '--tokenizer-model', '/home/dataset/model/llama-2-7b-hf/tokenizer.model',
    '--seq-length', '128',
    '--max-position-embeddings', '128',
    '--micro-batch-size', '1',
    '--global-batch-size', '8',
    '--lr-warmup-fraction', '0.01',
    '--bf16',
    '--data-path',
    '/home/dataset/llama2/alpaca_text_document',
    '--seed', '1234',
]

import math

import pytest
import torch
import torch_npu
import torch.distributed as dist
from mindspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.legacy.model.transformer import FlashSelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa
from megatron.core.transformer.dot_product_attention import DotProductAttention  # noqa
from megatron.core.transformer.enums import AttnMaskType
import megatron.core.parallel_state as mpu
from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.context_parallel.model_parallel_utils import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.model.transformer import get_attention_mask, set_attention_mask

sys.argv = original_argv
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_data_on_this_cp_rank(data, r_size, u_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    cp_size = r_size * u_size
    if r_size == 1:
        data = data.chunk(cp_size, dim=dim)[cp_rank]
    elif u_size == 1:
        data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim + 1:])
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=data.device)
        data = data.index_select(dim, index)
        data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    else:
        r_rank = cp_rank // u_size
        u_rank = cp_rank % u_size
        data = data.view(*data.shape[0:dim], 2 * r_size, data.shape[dim] // (2 * r_size), *data.shape[dim + 1:])
        index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=data.device)
        data = data.index_select(dim, index)
        data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
        data = data.chunk(u_size, dim=dim)[u_rank]
    return data


def run_hybridattn_cp(cp_size, u_size, bs, seq_len, dtype, cp_args):
    causal, send_recv_overlap = cp_args
    r_size = cp_size // u_size
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    args.attention_mask_type = 'causal' if causal else 'full'
    args.use_flash_attn = True
    if u_size == 1:
        args.context_parallel_algo = 'megatron_cp_algo'
    elif u_size == 8:
        args.context_parallel_algo = 'ulysses_cp_algo'
    else:
        args.context_parallel_algo = 'hybrid_cp_algo'

    args.context_parallel_size = cp_size
    args.ulysses_degree_in_cp = u_size
    args.seq_length = seq_len
    set_args(args)
    # clear global attention mask set by last test case
    set_attention_mask(None)
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    if causal:
        attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=q.device))
    else:
        attn_mask = None
    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=3 if attn_mask is not None else 0
    )[0]
    out.backward(dout)

    out_ref = get_data_on_this_cp_rank(out.clone().detach(), r_size, u_size, rank)
    k_grad_ref = get_data_on_this_cp_rank(k.grad.clone().detach(), r_size, u_size, rank)
    v_grad_ref = get_data_on_this_cp_rank(v.grad.clone().detach(), r_size, u_size, rank)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), r_size, u_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), r_size, u_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), r_size, u_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), r_size, u_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    # core branch use core.transformer.DotProductAtttention as core attention
    config = TransformerConfig(num_layers=2, hidden_size=n * d, num_attention_heads=n, use_cpu_initialization=True,
                               context_parallel_size=cp_size)
    local_attn = DotProductAttention(config=config, layer_number=1,
                                     attn_mask_type=AttnMaskType.causal, attention_type='self', attention_dropout=0.)

    if args.context_parallel_algo == "megatron_cp_algo":
        attn = local_attn
    else:
        ulysses_group = get_context_parallel_group_for_hybrid_ulysses() if args.context_parallel_algo == 'hybrid_cp_algo' else mpu.get_context_parallel_group()
        attn = UlyssesContextAttention(local_attn, ulysses_group)

    # global mask for core branch is generated at DotProductAttention forward
    # no need to generate and mask here
    out_ = attn(q_.reshape(-1, b, n, d), k_.reshape(-1, b, n, d), v_.reshape(-1, b, n, d), None, None, None, None)

    out_.backward(dout_)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out_ref, out_, **tols)
    assert torch.allclose(k_grad_ref, k_.grad, **tols)
    assert torch.allclose(v_grad_ref, v_.grad, **tols)


class TestHybridAttnCP(DistributedTest):
    world_size = 8

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True), (False, True)])
    def test_hybrid_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_hybridattn_cp(self.world_size, 2, 2, 8192, torch.bfloat16, cp_args)  # hybrid

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True), (False, True)])
    def test_ring_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_hybridattn_cp(self.world_size, 1, 2, 8192, torch.bfloat16, cp_args)  # ring only

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True), (False, True)])
    def test_ulysses_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_hybridattn_cp(self.world_size, 8, 2, 8192, torch.bfloat16, cp_args)  # ulysses only


