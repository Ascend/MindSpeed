from argparse import Namespace
import types

import torch
import torch_npu
import pytest
from pytest_mock import MockFixture

import mindspeed.megatron_adaptor


from mindspeed.core.pipeline_parallel.variable_seq_length.adaptor import (
    mindspeed_communicate,
    mindspeed_commuticate_shapes,
)
from mindspeed.core.pipeline_parallel.variable_seq_length.communicate import (
    communicate_impl,
    communicate_shapes_impl,
)


def test_mindspeed_communicate(mocker: MockFixture):
    mocker.patch(
        "mindspeed.core.pipeline_parallel.variable_seq_length.adaptor.communicate_impl",
        return_value=(1, 2),
    )
    ret = mindspeed_communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        config=None,
        wait_on_reqs=False,
    )
    assert ret == (1, 2)


def test_mindspeed_communicate_shapes(mocker: MockFixture):
    mocker.patch(
        "mindspeed.core.pipeline_parallel.variable_seq_length.adaptor.communicate_shapes_impl",
        return_value=(1, 2),
    )
    ret = mindspeed_commuticate_shapes(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        config=None,
    )
    assert ret == (1, 2)


@pytest.mark.parametrize(
    " config, expected",
    [
        (
            Namespace(
                use_ring_exchange_p2p=False,
                batch_p2p_comm=True,
                batch_p2p_sync=True,
            ),
            ([0, 0, 0]),
        ),
        (
            Namespace(
                use_ring_exchange_p2p=False,
                batch_p2p_comm=False,
                batch_p2p_sync=True,
            ),
            ([0, 0, 0]),
        ),
    ],
)
def test_communicate_shapes_impl(mocker: MockFixture, config, expected):
    mocker.patch("torch.cuda.synchronize")
    ret = communicate_shapes_impl(
        tensor_send_next=torch.tensor([1, 2, 3]),
        tensor_send_prev=torch.tensor([4, 5, 6]),
        recv_prev=True,
        recv_next=True,
        config=config,
        get_pipeline_model_parallel_group=lambda: None,
        get_pipeline_model_parallel_next_rank=lambda: 1,
        get_pipeline_model_parallel_prev_rank=lambda: 2,
        batched_p2p_ops=lambda **kwargs: [],
        p2p_ops=lambda **kwargs: [],
    )
    assert ret[0] == expected


@pytest.mark.parametrize(
    " config, expected",
    [
        (
            Namespace(
                use_ring_exchange_p2p=False,
                batch_p2p_comm=False,
                batch_p2p_sync=True,
                variable_seq_lengths=True,
                pipeline_dtype=torch.float32,
            ),
            (0, 0, []),
        ),
    ],
)
def test_communicate_impl(mocker: MockFixture, config, expected):
    mocker.patch("torch.cuda.synchronize")
    mocker.patch(
        "mindspeed.core.pipeline_parallel.variable_seq_length.communicate.communicate_shapes_impl",
        return_value=([0, 0, 0], [0, 0, 0]),
    )
    prev, next, reqs = communicate_impl(
        tensor_send_next=torch.tensor([1, 2, 3]),
        tensor_send_prev=torch.tensor([4, 5, 6]),
        recv_prev=True,
        recv_next=True,
        tensor_shape=[1, 2, 3],
        config=config,
        get_pipeline_model_parallel_group=lambda: [1],
        get_pipeline_model_parallel_next_rank=lambda: [1],
        get_pipeline_model_parallel_prev_rank=lambda: [2],
        batched_p2p_ops=lambda **kwargs: {},
        p2p_ops=lambda **kwargs: {},
        original_batched_p2p_ops=lambda **kwargs: {},
        original_p2p_ops=lambda **kwargs: {}
    )
    assert prev.sum() == expected[0]
    assert next.sum() == expected[1]
    assert len(reqs) == len(expected[2])
