# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import math

import pytest
import torch
import torch.distributed as dist
import torch_npu

from megatron.core import mpu

from mindspeed.core.context_parallel import get_batch_utils
from mindspeed.te.pytorch.attention.dot_product_attention import kvallgather_context_parallel as kvallgather_cp
from mindspeed.te.pytorch.attention.dot_product_attention.kvallgather_context_parallel import (
    AttnFuncWithCPAndKVAllGatherForTHD,
    clear_thd_load_balanced_cp_metadata_cache,
    get_thd_load_balanced_cp_metadata,
)
from tests_extend.commons import initialize_model_parallel, set_random_seed
from tests_extend.unit_tests.common import DistributedTest


def _get_local_token_index(cu_seqlens, cp_size, rank, device):
    ranges = []
    seq_start = 0
    for seq_end in cu_seqlens:
        chunk_len = (seq_end - seq_start) // (2 * cp_size)
        ranges.append(
            torch.arange(
                seq_start + rank * chunk_len,
                seq_start + (rank + 1) * chunk_len,
                device=device,
            )
        )
        ranges.append(
            torch.arange(
                seq_end - (rank + 1) * chunk_len,
                seq_end - rank * chunk_len,
                device=device,
            )
        )
        seq_start = seq_end
    return torch.cat(ranges).long()


def _diff_cu_seqlens(cu_seqlens):
    return [cu_seqlens[0]] + [cu_seqlens[idx] - cu_seqlens[idx - 1] for idx in range(1, len(cu_seqlens))]


def _get_combined_kv_prefix_index(cu_seqlens, cp_size, rank, device):
    chunk_ids = (rank, 2 * cp_size - rank - 1)
    ranges = []
    seq_start = 0
    for seq_end in cu_seqlens:
        chunk_len = (seq_end - seq_start) // (2 * cp_size)
        for chunk_id in chunk_ids:
            ranges.append(
                torch.arange(
                    seq_start,
                    seq_start + (chunk_id + 1) * chunk_len,
                    device=device,
                )
            )
        seq_start = seq_end
    return torch.cat(ranges).long()


@pytest.mark.parametrize(
    "cp_size,cu_seqlens",
    [
        (2, [16, 40, 72]),
        (4, [32, 80, 144]),
    ],
)
def test_thd_load_balanced_cp_metadata(cp_size, cu_seqlens):
    full_token_ids = torch.arange(cu_seqlens[-1])
    rank_major_token_ids = torch.cat(
        [
            full_token_ids.index_select(
                0,
                _get_local_token_index(cu_seqlens, cp_size, rank, torch.device("cpu")),
            )
            for rank in range(cp_size)
        ]
    )

    rank_workloads = []
    for rank in range(cp_size):
        metadata = get_thd_load_balanced_cp_metadata(cu_seqlens, cp_size, rank, torch.device("cpu"))

        q_lens = _diff_cu_seqlens(metadata["actual_seq_qlen"])
        kv_lens = _diff_cu_seqlens(metadata["actual_seq_kvlen"])
        expected_q_lens = []
        expected_kv_lens = []
        seq_start = 0
        for seq_end in cu_seqlens:
            chunk_len = (seq_end - seq_start) // (2 * cp_size)
            expected_q_lens.extend((chunk_len, chunk_len))
            expected_kv_lens.extend(
                (
                    (rank + 1) * chunk_len,
                    (2 * cp_size - rank) * chunk_len,
                )
            )
            seq_start = seq_end
        assert q_lens == expected_q_lens
        assert kv_lens == expected_kv_lens
        assert sum(q_lens) == metadata["local_total_len"]

        kv_index = metadata["kv_index_in_rank_major"]
        expected_kv_index = _get_combined_kv_prefix_index(cu_seqlens, cp_size, rank, torch.device("cpu"))
        assert torch.equal(
            rank_major_token_ids.index_select(0, kv_index),
            full_token_ids.index_select(0, expected_kv_index),
        )

        combined_grad = torch.arange(1, kv_index.numel() + 1, dtype=torch.float32)
        rank_major_grad = torch.zeros(metadata["full_total_len"])
        reference_grad = torch.zeros(metadata["full_total_len"])
        rank_major_grad.index_add_(0, kv_index, combined_grad)
        reference_grad.index_add_(0, expected_kv_index, combined_grad)
        assert torch.equal(
            rank_major_grad,
            reference_grad.index_select(0, rank_major_token_ids),
        )

        rank_workloads.append(sum(q_len * kv_len for q_len, kv_len in zip(q_lens, kv_lens)))

    assert len(set(rank_workloads)) == 1


def test_thd_metadata_tensor_identity_cache_avoids_reconversion():
    clear_thd_load_balanced_cp_metadata_cache()
    cu_seqlens = torch.tensor([16, 40, 72])
    metadata = kvallgather_cp._get_thd_load_balanced_cp_metadata_cached(cu_seqlens, 2, 0, torch.device("cpu"))

    cache_info = kvallgather_cp._get_thd_load_balanced_cp_metadata_for_tensor.cache_info()  # pylint: disable=no-value-for-parameter
    cached_metadata = kvallgather_cp._get_thd_load_balanced_cp_metadata_cached(cu_seqlens, 2, 0, torch.device("cpu"))
    updated_cache_info = kvallgather_cp._get_thd_load_balanced_cp_metadata_for_tensor.cache_info()  # pylint: disable=no-value-for-parameter
    assert cached_metadata is metadata
    assert updated_cache_info.hits == cache_info.hits + 1


def test_thd_metadata_tensor_mutation_invalidates_identity_cache():
    clear_thd_load_balanced_cp_metadata_cache()
    cu_seqlens = torch.tensor([16, 40, 72])
    metadata = kvallgather_cp._get_thd_load_balanced_cp_metadata_cached(cu_seqlens, 2, 0, torch.device("cpu"))

    cu_seqlens[-1] = 80
    updated_metadata = kvallgather_cp._get_thd_load_balanced_cp_metadata_cached(cu_seqlens, 2, 0, torch.device("cpu"))
    assert updated_metadata is not metadata
    assert updated_metadata["full_total_len"] == 80


class TestKVAllGatherContextParallelTHD(DistributedTest):
    world_size = 2

    def test_eod_load_balanced_batch_partition(self, monkeypatch):
        initialize_model_parallel(context_parallel_size=self.world_size)
        rank = dist.get_rank()
        cu_seqlens = [16, 40, 72]
        full_tokens = torch.arange(cu_seqlens[-1], device="npu").view(1, -1)
        batch = {
            "tokens": full_tokens.clone(),
            "labels": full_tokens.clone(),
            "loss_mask": torch.ones_like(full_tokens, dtype=torch.float32),
            "attention_mask": None,
            "position_ids": full_tokens.clone(),
        }

        monkeypatch.setattr(get_batch_utils, "get_ring_degree", lambda: 1)
        get_batch_utils._get_batch_on_this_cp_rank_in_megatron_cp_eod_padding(
            batch, torch.tensor(cu_seqlens, device="npu")
        )

        local_index = _get_local_token_index(cu_seqlens, self.world_size, rank, torch.device("npu"))
        expected_tokens = full_tokens.index_select(1, local_index)
        assert torch.equal(batch["tokens"], expected_tokens)
        assert torch.equal(batch["labels"], expected_tokens)
        assert torch.equal(batch["position_ids"], expected_tokens)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "query_heads,kv_heads,qk_head_dim,v_head_dim",
        [
            pytest.param(4, 4, 128, 128, id="mha"),
            pytest.param(8, 2, 128, 128, id="gqa"),
            pytest.param(4, 4, 192, 128, id="mla"),
        ],
    )
    def test_forward_and_backward_match_non_cp_reference(
        self,
        dtype,
        query_heads,
        kv_heads,
        qk_head_dim,
        v_head_dim,
    ):
        initialize_model_parallel(context_parallel_size=self.world_size)
        set_random_seed(1234)

        rank = dist.get_rank()
        cu_seqlens = [16, 40]
        total_tokens = cu_seqlens[-1]
        softmax_scale = 1.0 / math.sqrt(qk_head_dim)

        q_ref = torch.randn(
            total_tokens,
            query_heads,
            qk_head_dim,
            dtype=dtype,
            device="npu",
            requires_grad=True,
        )
        k_ref = torch.randn(
            total_tokens,
            kv_heads,
            qk_head_dim,
            dtype=dtype,
            device="npu",
            requires_grad=True,
        )
        v_ref = torch.randn(
            total_tokens,
            kv_heads,
            v_head_dim,
            dtype=dtype,
            device="npu",
            requires_grad=True,
        )
        dout = torch.randn(
            total_tokens,
            query_heads,
            v_head_dim,
            dtype=dtype,
            device="npu",
        )
        attention_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device="npu"))

        out_ref = torch_npu.npu_fusion_attention(
            q_ref,
            k_ref,
            v_ref,
            query_heads,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=softmax_scale,
            pre_tockens=65536,
            next_tockens=0,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=3,
            actual_seq_qlen=cu_seqlens,
            actual_seq_kvlen=cu_seqlens,
        )[0]
        out_ref.backward(dout)

        local_index = _get_local_token_index(cu_seqlens, self.world_size, rank, torch.device("npu"))
        q = q_ref.detach().index_select(0, local_index).requires_grad_(True)
        k = k_ref.detach().index_select(0, local_index).requires_grad_(True)
        v = v_ref.detach().index_select(0, local_index).requires_grad_(True)

        out = AttnFuncWithCPAndKVAllGatherForTHD.apply(
            q,
            k,
            v,
            query_heads,
            attention_mask,
            "thd",
            "causal",
            0.0,
            softmax_scale,
            False,
            mpu.get_context_parallel_group(),
            cu_seqlens,
            cu_seqlens,
        )
        assert out.shape == (q.shape[0], query_heads, v_head_dim)
        out.backward(dout.index_select(0, local_index))

        tolerances = {"atol": 5e-3, "rtol": 5e-3}
        if dtype == torch.bfloat16:
            tolerances = {"atol": 2.5e-2, "rtol": 2.5e-2}

        assert torch.allclose(out, out_ref.detach().index_select(0, local_index), **tolerances)
        assert torch.allclose(q.grad, q_ref.grad.index_select(0, local_index), **tolerances)
        assert torch.allclose(k.grad, k_ref.grad.index_select(0, local_index), **tolerances)
        assert torch.allclose(v.grad, v_ref.grad.index_select(0, local_index), **tolerances)
