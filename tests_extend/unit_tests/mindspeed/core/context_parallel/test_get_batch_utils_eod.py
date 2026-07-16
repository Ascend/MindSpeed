from argparse import Namespace

import pytest
import torch

from mindspeed.core.context_parallel import get_batch_utils
from mindspeed.core.context_parallel.get_batch_utils import _validate_actual_seq_len


def test_eod_accepts_dataset_integral_cu_seqlens():
    actual_seq_len = torch.tensor([83, 167], dtype=torch.int64)

    result = _validate_actual_seq_len(actual_seq_len)

    assert result is actual_seq_len


def test_eod_rejects_non_integral_cu_seqlens():
    with pytest.raises(AssertionError, match='integral dtype'):
        _validate_actual_seq_len(torch.tensor([84.0, 168.0]))


def test_p2p_eod_uses_exact_integer_local_endpoints(monkeypatch):
    args = Namespace(context_parallel_size=2, tensor_model_parallel_size=2)
    actual_seq_len = torch.tensor([8, 16], dtype=torch.int64)
    monkeypatch.setattr(get_batch_utils, 'pad_data', lambda endpoints, *_: endpoints)

    result = get_batch_utils._prepare_p2p_eod_actual_seq_len(actual_seq_len, {}, args)

    assert torch.equal(result, torch.tensor([4, 8], dtype=torch.int64))
    assert result.dtype == torch.int64


def test_p2p_eod_rejects_non_divisible_padded_endpoints(monkeypatch):
    args = Namespace(context_parallel_size=2, tensor_model_parallel_size=2)
    monkeypatch.setattr(get_batch_utils, 'pad_data', lambda endpoints, *_: endpoints)

    with pytest.raises(AssertionError, match='must be divisible'):
        get_batch_utils._prepare_p2p_eod_actual_seq_len(
            torch.tensor([7], dtype=torch.int64), {}, args
        )
