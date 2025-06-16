# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.

from argparse import Namespace

import torch
import torch_npu
import pytest
from pytest_mock import MockFixture

import mindspeed.megatron_adaptor

from mindspeed.core.pipeline_parallel.multi_parameter.adaptor import (
    get_tensor_shapes_wrapper,
    get_forward_backward_func_wrapper,
    core_transformer_config_from_args_wrapper,
    forward_step_wrapper,
    forward_backward_pipelining_with_interleaving,
)

from mindspeed.core.pipeline_parallel.multi_parameter.communication import (
    backward_step_impl,
    recv_forward_or_backward,
    send_forward_or_backward,
    send_forward_and_backward,
)


def test_get_tensor_shapes_wrapper():
    fn = lambda: None
    wrapper_func = get_tensor_shapes_wrapper(fn)
    config = Namespace(
        pipeline_tensor_shapes=[
            {
                "shape": (1, 168),
                "dtype": torch.float32,
            },
            {
                "shape": (1, 168),
                "dtype": torch.float32,
            },
        ]
    )
    ret = wrapper_func(config=config)
    assert ret == [
        {
            "shape": (1, 168),
            "dtype": torch.float32,
        },
        {
            "shape": (1, 168),
            "dtype": torch.float32,
        },
    ]


func_1 = lambda: ([[0]], 1)
func_2 = lambda: ([0], 1)


@pytest.mark.parametrize(
    " func, expected",
    [
        (
            func_1,
            ([0], 1),
        ),
        (
            func_2,
            ([0], 1),
        ),
    ],
)
def test_forward_step_wrapper(func, expected):
    ret = forward_step_wrapper(func)
    assert ret() == expected


def test_get_forward_backward_func_wrapper(mocker: MockFixture):
    mocker.patch(
        "mindspeed.core.pipeline_parallel.multi_parameter.adaptor.get_pipeline_model_parallel_world_size",
        return_value=2,
    )
    mocker.patch(
        "mindspeed.core.pipeline_parallel.multi_parameter.adaptor.get_virtual_pipeline_model_parallel_world_size",
        return_value=2,
    )
    func = lambda: None
    wrapper_func = get_forward_backward_func_wrapper(func)
    assert wrapper_func() == forward_backward_pipelining_with_interleaving


def test_core_transformer_config_from_args_wrapper(mocker: MockFixture):
    func = lambda x: Namespace(deallocate_pipeline_outputs=True)
    args = mocker.MagicMock()
    args.use_multiparameter_pipeline_model_paralllel = True
    wrapper_func = core_transformer_config_from_args_wrapper(func)
    ret = wrapper_func(args)
    assert not ret.deallocate_pipeline_outputs


@pytest.mark.parametrize(
    "input_tensor, output_tensor, output_tensor_grad, expected",
    [
        (
            torch.tensor([1, 2, 3], device=torch.cuda.current_device()),
            torch.tensor([4, 5, 6], device=torch.cuda.current_device()),
            torch.tensor([7, 8, 9], device=torch.cuda.current_device()),
            torch.tensor([0, 0, 0], device=torch.cuda.current_device()),
        ),
        (
            torch.tensor([1, 2, 3], device=torch.cuda.current_device()),
            torch.tensor([4, 5, 6], device=torch.cuda.current_device()),
            None,
            torch.tensor([0, 0, 0], device=torch.cuda.current_device()),
        ),
        (
            None,
            torch.tensor([4, 5, 6], device=torch.cuda.current_device()),
            None,
            None,
        ),
    ],
)
def test_backward_step_impl(
    mocker: MockFixture,
    input_tensor,
    output_tensor,
    output_tensor_grad,
    expected,
):
    config = mocker.MagicMock()
    config.timers = None
    config.grad_scale_func = None
    mocker.patch("torch.autograd.backward")
    get_pipeline_model_parallel_world_size = lambda: 1
    is_pipeline_stage_after_split = lambda: True

    ret = backward_step_impl(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        output_tensor_grad=output_tensor_grad,
        is_encoder_and_decoder=False,
        config=config,
        get_pipeline_model_parallel_world_size=get_pipeline_model_parallel_world_size,
        is_pipeline_stage_after_split=is_pipeline_stage_after_split,
    )
    if ret is not None:
        assert (ret == expected).all()
    else:
        assert ret == expected


@pytest.mark.parametrize(
    "tensor_shapes, recv, expected",
    [
        (
            [None],
            lambda x, y: 1,
            [None],
        ),
        (
            [{"dtype": torch.float32, "shape": (1, 3)}],
            lambda x, y: 1,
            [1],
        ),
    ],
)
def test_recv_forward_or_backward(tensor_shapes, recv, expected):
    config = Namespace(
        pipeline_dtype=torch.float32,
    )
    ret = recv_forward_or_backward(
        tensor_shapes=tensor_shapes,
        config=config,
        recv=recv,
    )
    assert ret == expected


@pytest.mark.parametrize(
    "tensors, tensor_shapes, config, expected",
    [
        (
            None,
            [None],
            Namespace(pipeline_dtype=torch.float32),
            torch.float32,
        ),
        (
            torch.tensor([1, 2, 3]),
            [{"dtype": torch.float64, "shape": (1, 3)}],
            Namespace(pipeline_dtype=torch.float32),
            torch.float64,
        ),
    ],
)
def test_send_forward_or_backward(tensors, tensor_shapes, config, expected):
    send = lambda x, y: None
    send_forward_or_backward(
        tensors=tensors,
        tensor_shapes=tensor_shapes,
        config=config,
        send_forward_or_backward_=send,
    )
    assert config.pipeline_dtype == expected


@pytest.mark.parametrize(
    "tensors, tensor_shapes, expected",
    [
        (
            [torch.tensor([1, 2, 3])],
            [{"dtype": torch.float32, "shape": (1, 3)}],
            [torch.tensor([1, 2, 3])],
        ),
        (
            torch.tensor([1, 2, 3]),
            [{"dtype": torch.float32, "shape": (1, 3)}],
            [None],
        ),
    ],
)
def test_send_forward_and_backward(tensors, tensor_shapes, expected):
    send = lambda x, y, z: x
    config = Namespace(
        pipeline_dtype=torch.float32,
    )
    ret = send_forward_and_backward(
        tensors=tensors,
        tensor_shapes=tensor_shapes,
        config=config,
        send_forward_and_backward_=send,
    )

    if ret[0] is not None:
        assert (ret[0] == expected[0]).all()
    else:
        assert ret == expected
