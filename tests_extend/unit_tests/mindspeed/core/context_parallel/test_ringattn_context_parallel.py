# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import sys
# Setting sys.argv is mainly to ensure that --context-parallel-size is not None, so that the code block (which will be executed
# after determining that context-parallel-size is not None) will be executed in megatron_adaptor.
original_argv = sys.argv.copy()
sys.argv = [
    sys.argv[0],
    '--use-flash-attn',
    '--context-parallel-algo', 'megatron_cp_algo',
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
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_group_for_hybrid_ulysses,
                                           get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

sys.argv = original_argv
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim + 1:])
    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=data.device)
    data = data.index_select(dim, index)
    data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    return data


def get_data_on_all_cp_ranks(data, cp_size, dim=0):
    """ Combine data along sequence dimension from multiple chunks.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, -1, *data.shape[dim + 1:])
    index = [[i, 2 * cp_size - i - 1] for i in range(cp_size)]
    index = torch.tensor(index).flatten().to(data.device)
    index = index[:, None, None, None].repeat(1, *data.shape[1:])
    out = torch.empty_like(data)
    out = out.scatter(dim=0, index=index, src=data)
    out = out.view(-1, *out.shape[2:])
    return out


def run_ringattn_cp(cp_size, bs, seq_len, dtype, cp_args):
    from megatron.core import mpu
    causal, send_recv_overlap, cp_window_size, pse_type = cp_args
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    args.cp_window_size = cp_window_size
    args.context_parallel_algo = 'megatron_cp_algo'
    set_args(args)
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    pse = None
    if pse_type == 2 or pse_type == 3:
        pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(n)

    if causal:
        attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=q.device))
    else:
        attn_mask = None

    if pse is None:
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
    else:
        out = npu_fusion_attention( \
            q, k, v, n, 'SBH', \
            pse=pse, \
            pse_type=pse_type, \
            padding_mask=None, \
            atten_mask=attn_mask, \
            scale=scale, \
            pre_tokens=seq_len, \
            next_tokens=0, \
            keep_prob=1., \
            inner_precise=0, \
            sparse_mode=3 if attn_mask is not None else 0
        )[0]
        out.backward(dout)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    in_hybrid_mode = False
    if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
        in_hybrid_mode = True

    if not in_hybrid_mode:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()
    else:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

    cp_para = dict()
    cp_para['causal'] = causal
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None
    cp_para['pse'] = pse
    cp_para['pse_type'] = pse_type


    cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
    cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
    cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
    cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
    cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()


    out_ = ringattn_context_parallel(q_, k_, v_, n, cp_para, softmax_scale=scale, attn_mask=None)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, cp_size)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


def run_ringattn_cp_with_kv_cache(cp_size, bs, seq_len, dtype, test_args):
    (causal, send_recv_overlap), cache_args, cp_window_size = test_args
    from megatron.core import mpu
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    args.cp_window_size = cp_window_size
    args.context_parallel_algo = 'megatron_cp_algo'
    set_args(args)
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

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    in_hybrid_mode = False
    if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
        in_hybrid_mode = True

    if not in_hybrid_mode:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()
    else:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

    cp_para = dict()
    cp_para['causal'] = causal
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None


    cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
    cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
    cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
    cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
    cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
    cp_para['cache_policy'] = cache_args


    out_ = ringattn_context_parallel(q_, k_, v_, n, cp_para, softmax_scale=scale, attn_mask=None)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, cp_size)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


def run_kv_cache_by_layers(num_layers, correct_cache_list, cache_args, layer_args):
    generate_cache_list = list()
    for i in range(1, num_layers + 1):
        cache_policy = get_cache_policy(i, cache_args, layer_args)
        generate_cache_list.append(cache_policy)

    assert generate_cache_list == correct_cache_list


@pytest.mark.skip(reason='This test case may be blocked in some environments; temporarily skipping.')
class TestRingAttnCP(DistributedTest):
    world_size = 8

    num_layers = 12
    correct_cache_dict = dict()
    correct_cache_dict[("full", 1)] = ["full", None, "full", None, "full", None,
                                       "full", None, "full", None, "full", None]
    correct_cache_dict[("full", 4)] = ["full", None, None, None, None,
                                       "full", None, None, None, None,
                                       "full", None]
    correct_cache_dict[("full", 5)] = ["full", None, None, None, None, None,
                                       "full", None, None, None, None, None]
    correct_cache_dict[("half", 1)] = ["half", None, "half", None, "half", None,
                                       "half", None, "half", None, "half", None]
    correct_cache_dict[("half", 4)] = ["half", None, None, None, None,
                                       "half", None, None, None, None,
                                       "half", None]
    correct_cache_dict[("half", 5)] = ["half", None, None, None, None, None,
                                       "half", None, None, None, None, None]

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True, 1, 1), (True, True, 2, 1), (True, True, 4, 1), (False, False, 1, 1)])
    def test_ringattn_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_ringattn_cp(self.world_size, 2, 8192, torch.bfloat16, cp_args)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True, 1, 2), (True, True, 2, 2), ((True, True, 4, 2))])
    def test_ringattn_context_parallel_seq8192_bs2_bf16_pse2(self, cp_args):
        run_ringattn_cp(self.world_size, 2, 8192, torch.bfloat16, cp_args)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True, 1, 3), (True, True, 2, 3), ((True, True, 4, 3))])
    def test_ringattn_context_parallel_seq8192_bs2_bf16_pse3(self, cp_args):
        run_ringattn_cp(self.world_size, 2, 8192, torch.bfloat16, cp_args)

    # kv cache test
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True), (False, False)])
    @pytest.mark.parametrize("cp_window_size", [1, 2, 4])
    @pytest.mark.parametrize("cache_args", ["full", "half"])
    def test_ringattn_cp_with_kv_cache(self, cp_args, cp_window_size, cache_args):
        args = (cp_args, cache_args, cp_window_size)
        run_ringattn_cp_with_kv_cache(self.world_size, 2, 8192, torch.bfloat16, args)

    # cache depends on layer test
    @pytest.mark.parametrize("cache_args", ["full", "half"])
    @pytest.mark.parametrize("layer_args", [1, 4, 5])
    def test_kv_cache_by_layers(self, cache_args, layer_args):
        run_kv_cache_by_layers(self.num_layers, self.correct_cache_dict.get((cache_args, layer_args)), cache_args,
                               layer_args)
