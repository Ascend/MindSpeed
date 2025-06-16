# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.

from argparse import Namespace
import types

import torch
import torch_npu
import pytest
from pytest_mock import MockFixture

import mindspeed.megatron_adaptor


from mindspeed.core.pipeline_parallel.optimize_send_recv_comm.adaptor import (
    mindspeed_get_forward_backward_func,
    mindspeed_initialize_model_parallel_wrapper,
    mindspeed_destroy_model_parallel_wrapper,
    flexible_schedules,
)
from mindspeed.core.pipeline_parallel.optimize_send_recv_comm.parallel_state import (
    initialize_model_parallel_impl,
    get_pipeline_parallel_group_for_new_stream,
)

from mindspeed.core.pipeline_parallel.optimize_send_recv_comm.parallel_state import (
    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM,
)


def test_mindspeed_get_forward_backward_func():
    ret = mindspeed_get_forward_backward_func()
    assert (
        flexible_schedules.get_pipeline_parallel_group_for_new_stream
        == get_pipeline_parallel_group_for_new_stream
    )
    assert ret == flexible_schedules.forward_backward_pipelining_without_interleaving


def test_mindspeed_initialize_model_parallel_wrapper(mocker: MockFixture):
    mocker.patch(
        "mindspeed.core.pipeline_parallel.optimize_send_recv_comm.adaptor.initialize_model_parallel_impl",
    )
    func = lambda: None
    ret = mindspeed_initialize_model_parallel_wrapper(func)
    assert isinstance(ret, types.FunctionType)


def test_mindspeed_destroy_model_parallel_wrapper(mocker: MockFixture):
    func = lambda: None
    ret = mindspeed_destroy_model_parallel_wrapper(func)
    assert isinstance(ret, types.FunctionType)


def test_initialize_model_parallel_impl(mocker: MockFixture):
    mocker.patch(
        "torch.distributed.new_group",
        return_value=(1, 2),
    )
    mocker.patch(
        "torch.distributed.get_world_size",
        return_value=8,
    )
    mocker.patch(
        "torch.distributed.get_rank",
        return_value=1,
    )
    initialize_model_parallel_impl(get_nccl_options=lambda x, y: None)
    assert get_pipeline_parallel_group_for_new_stream() == (1, 2)
