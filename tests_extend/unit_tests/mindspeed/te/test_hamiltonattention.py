# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
import sys
from typing import Union, List, Tuple
import math
import pytest
import torch
import torch_npu
import torch.distributed as dist
from mindspeed import megatron_adaptor
from mindspeed.te.pytorch.attention.dot_product_attention.context_parallel import HamiltonCPStrategy
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

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


def get_data_on_all_cp_ranks_sbhd(data, cp_size, dim=0):
    """ Combine data along sequence dimension from multiple chunks.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, -1, *data.shape[dim + 1:])
    index = [[i, 2 * cp_size - i - 1] for i in range(cp_size)]
    index = torch.tensor(index).flatten().to(data.device)
    index_shape = [len(index)] + [1] * (len(data.shape) - 1)
    index = index.view(*index_shape).expand_as(data)
    out = torch.empty_like(data)
    out = out.scatter(dim=0, index=index, src=data)
    out = out.view(-1, *out.shape[2:])
    return out


def run_hamattn_context_parallel_bsh_te(cp_size, bs, seq_len, dtype, cp_args):
    from megatron.core import mpu
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 40, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    q_ref = q.view(s, b, n * d)
    k_ref = k.view(s, b, n * d)
    v_ref = v.view(s, b, n * d)
    pse = None
    attn_mask = None

    out_ref = torch_npu.npu_fusion_attention( \
        q_ref, k_ref, v_ref, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=3 if attn_mask is not None else 0)[0]
    out_ref.backward(dout)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    path_num = 4
    in_mapping_list = [[-1, 1, -1, -1, -1, 2, 3, 0],
        [0, -1, -1, -1, 3, 1, -1, 2],
        [-1, 3, -1, 2, 1, 0, -1, -1],
        [-1, 2, -1, -1, 0, 3, 1, -1],
        [2, -1, 0, 3, -1, -1, -1, 1],
        [3, 0, 1, -1, -1, -1, 2, -1],
        [1, -1, 2, 0, -1, -1, -1, 3],
        [-1, -1, 3, 1, 2, -1, 0, -1]]

    import numpy as np
    out_mapping_list = np.array(in_mapping_list).T.tolist()
    hamilton = HamiltonCPStrategy(
        softmax_scale=scale,
        attention_dropout=0.0,
        attention_type="self",
        deterministic=False,
        path_num=path_num,
        in_mapping=in_mapping_list,
        out_mapping=out_mapping_list,
        permute_index=None,
        restore_index=None)

    cp_group = mpu.get_context_parallel_group()
    out_ = hamilton(
            query_layer=q_,
            key_layer=k_,
            value_layer=v_,
            attention_mask=attn_mask,
            qkv_format="sbhd",
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            attn_mask_type="full",
            max_seqlen_q=None,
            max_seqlen_kv=None,
            cp_group=cp_group,
            cp_global_ranks=None
            )
    out_.backward(dout_)

    # Gather outputs from all CP ranks
    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_cp = torch.cat(output_list, dim=0)
    out_cp = get_data_on_all_cp_ranks(out_cp, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks_sbhd(k_grad, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks_sbhd(v_grad, cp_size)

    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    assert torch.allclose(out_ref, out_cp, **tols)
    assert torch.allclose(k.grad.view(s, b, n * d), k_grad.view(s, b, n * d), **tols)
    assert torch.allclose(v.grad.view(s, b, n * d), v_grad.view(s, b, n * d), **tols)


PERMUTE_INDEX_TENSOR = None
RESTORE_INDEX_TENSOR = None


def get_permute_and_restore_index(seq_lens, seq_dim, path_num, target_device):
    global PERMUTE_INDEX_TENSOR
    global RESTORE_INDEX_TENSOR
    if PERMUTE_INDEX_TENSOR is None:
        seq_num = len(seq_lens)
        total_seq_len = sum(seq_lens)
        target_indices = []
        for p in range(path_num):
            for s in range(seq_num):
                seq_start = sum(seq_lens[:s])
                chunk_start = seq_start + p * (seq_lens[s] // path_num)
                for pos in range(seq_lens[s] // path_num):
                    target_indices.append(chunk_start + pos)

        permute_indices = torch.tensor(target_indices, dtype=torch.long, device=target_device)
        restore_indices = torch.zeros_like(permute_indices)
        for i, idx in enumerate(permute_indices):
            restore_indices[idx] = i


        PERMUTE_INDEX_TENSOR = permute_indices
        RESTORE_INDEX_TENSOR = restore_indices

    return (PERMUTE_INDEX_TENSOR, RESTORE_INDEX_TENSOR)


def get_index_tnd(actual_seq_len, cp_size, cp_rank, dim=0):
    nseq = len(actual_seq_len)
    points = [0] + actual_seq_len

    def chunk_tensor(i):
        start = points[i]
        end = points[i + 1]

        size = (end - start) // (2 * cp_size)
        part1 = torch.arange(start + cp_rank * size, start + (cp_rank + 1) * size)
        part2 = torch.arange(end - (cp_rank + 1) * size, end - cp_rank * size)

        part = torch.cat((part1, part2))
        return part

    chunks = [chunk_tensor(i) for i in range(nseq)]
    return torch.cat(chunks)


def get_data_on_this_cp_rank_tnd_te(data, actual_seq_len, cp_size, cp_rank, dim=0):
    index_lst = get_index_tnd(actual_seq_len, cp_size, cp_rank).to(data.device)
    data_lst = data.index_select(0, index_lst)
    return data_lst


def run_hamattn_cp_tnd_te(cp_size, cu_seq_len, dtype):
    from megatron.core import mpu
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = 1, 2, cu_seq_len[-1], 64
    h = n * d
    t = b * s
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(t, n, d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(t, n, d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(t, n, d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(t, n, d, dtype=dtype, device='npu', requires_grad=True)

    pse = None
    attn_mask = None
    out_ref = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'TND', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=s, \
        next_tockens=0, \
        keep_prob=1., \
        actual_seq_qlen=cu_seq_len, \
        actual_seq_kvlen=cu_seq_len, \
        sparse_mode=3 if attn_mask is not None else 0
    )[0]
    out_ref.backward(dout)

    q_ = get_data_on_this_cp_rank_tnd_te(q.clone().detach(), cu_seq_len, cp_size, rank)
    k_ = get_data_on_this_cp_rank_tnd_te(k.clone().detach(), cu_seq_len, cp_size, rank)
    v_ = get_data_on_this_cp_rank_tnd_te(v.clone().detach(), cu_seq_len, cp_size, rank)
    dout_ = get_data_on_this_cp_rank_tnd_te(dout.clone().detach(), cu_seq_len, cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    if cp_size != 8:
        raise AssertionError(f"Current testcase only suits for cp_size=8, but got cp_size={cp_size}")
    path_num = 4
    in_mapping_list = [[-1, 1, -1, -1, -1, 2, 3, 0],
                       [0, -1, -1, -1, 3, 1, -1, 2],
                       [-1, 3, -1, 2, 1, 0, -1, -1],
                       [-1, 2, -1, -1, 0, 3, 1, -1],
                       [2, -1, 0, 3, -1, -1, -1, 1],
                       [3, 0, 1, -1, -1, -1, 2, -1],
                       [1, -1, 2, 0, -1, -1, -1, 3],
                       [-1, -1, 3, 1, 2, -1, 0, -1]]
    import numpy as np
    out_mapping_list = np.array(in_mapping_list).T.tolist()

    actual_seq_len = [0] * len(cu_seq_len)
    for i in range(len(cu_seq_len)):
        if i == 0:
            actual_seq_len[i] = (cu_seq_len[i] - 0) // cp_size
        else:
            actual_seq_len[i] = (cu_seq_len[i] - cu_seq_len[i - 1]) // cp_size

    permute_index, restore_index = get_permute_and_restore_index(actual_seq_len, 0, 4, q_.device)
    hamilton = HamiltonCPStrategy(
        softmax_scale=scale,
        attention_dropout=0.0,
        attention_type="self",
        deterministic=False,
        path_num=path_num,
        in_mapping=in_mapping_list,
        out_mapping=out_mapping_list,
        permute_index=permute_index,
        restore_index=restore_index
    )
    cp_group = mpu.get_context_parallel_group()
    out_ = hamilton(
        query_layer=q_,
        key_layer=k_,
        value_layer=v_,
        attention_mask=attn_mask,
        qkv_format="thd",
        cu_seqlens_q=cu_seq_len,
        cu_seqlens_kv=cu_seq_len,
        attn_mask_type="full",
        max_seqlen_q=None,
        max_seqlen_kv=None,
        cp_group=cp_group,
        cp_global_ranks=None
    )
    out_.backward(dout_)

    out_ref = get_data_on_this_cp_rank_tnd_te(out_ref.clone().detach(), cu_seq_len, cp_size, rank)
    k_grad_ref = get_data_on_this_cp_rank_tnd_te(k.grad.clone().detach(), cu_seq_len, cp_size, rank)
    v_grad_ref = get_data_on_this_cp_rank_tnd_te(v.grad.clone().detach(), cu_seq_len, cp_size, rank)

    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    assert torch.allclose(out_, out_ref, **tols)
    assert torch.allclose(k_.grad, k_grad_ref, **tols)
    assert torch.allclose(v_.grad, v_grad_ref, **tols)


class TestHamiltonAttnCP(DistributedTest):
    world_size = 8

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_hamattn_context_parallel_bsh_te(self, cp_args=(False, False, 1, 1)):
        run_hamattn_context_parallel_bsh_te(self.world_size, 2, 8192, torch.bfloat16, (False, False, 1, 1))

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_hamattn_context_parallel_tnd_te(self, cp_args=(False, False, 1, 1)):
        run_hamattn_cp_tnd_te(self.world_size, [1024, 4096, 8192], torch.bfloat16)
