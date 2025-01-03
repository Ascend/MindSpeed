import torch
import torch_npu
import pytest
import numpy as np
from unit_tests.common import DistributedTest


shape_list = [
    [2048, 4096, 8192],
    [3584, 45056, 4608],
    [1280, 32768, 8192],
    [4096, 23640, 1638],
    [5504, 32768, 8192],
    [5120, 32256, 7936],
    [6848, 32768, 8192],
    [3072, 32768, 8192],
    [16384, 23640, 4096]
]

total_shape_num = len(shape_list)
dtype_list = [torch.float16, torch.bfloat16]
trans_case_list = ["trans_none", "trans_B"]

test_ratios = 5e-3
min_normal = 6e-5

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def generate_gaussian(dim0, dim1, dtype):
    return torch.randn(dim0, dim1, dtype=dtype, device=torch.npu.current_device())


def generate_zeros(dim0, dim1, dtype):
    return torch.zeros(dim0, dim1, dtype=dtype, device=torch.npu.current_device())


def generate_input_tensors(ipt_m, k, n, dtype):
    torch.manual_seed(0)
    input1 = generate_gaussian(ipt_m, k, dtype)
    input2 = generate_gaussian(k, n, dtype)
    bias = generate_gaussian(1, n, dtype)
    return input1, input2, bias


def is_similar(actual_output, pred_output):
    torch.npu.synchronize()
    error_ratio = -1
    if torch.npu.current_device() == 0:
        actual_shape = list(actual_output.shape)
        actual_output = actual_output.to(torch.float32).cpu().numpy()
        pred_output = pred_output.to(torch.float32).cpu().numpy()
        diff = np.abs(actual_output - pred_output)
        limit_error = np.abs(actual_output * test_ratios)
        limit_error[limit_error < min_normal] = min_normal
        error_count = np.sum(diff > limit_error)
        if actual_shape[0] != 0 and actual_shape[1] != 0:
            error_ratio = error_count / actual_shape[0] / actual_shape[1]
        else:
            error_ratio = np.inf
    return error_ratio


class TestCoC(DistributedTest):
    world_size = 8

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_coc_matmul_all_reduce(self):
        from mindspeed.ops.lcal_functional import coc_ops
        for shape_idx in range(total_shape_num):
            for dtype in dtype_list:
                m, k, n = shape_list[shape_idx][0], shape_list[shape_idx][1], shape_list[shape_idx][2]
                input1, input2, bias = generate_input_tensors(m, k, n, dtype)

                for trans_case in trans_case_list:
                    trans_b = False if trans_case == "trans_none" else True
                    input_left = input1
                    input_right = input2.t().contiguous() if trans_b else input2

                    orig_output = torch.matmul(input_left, input_right.t()) if trans_b else torch.matmul(input_left,
                                                                                                         input_right)
                    torch.distributed.all_reduce(orig_output)
                    torch.npu.synchronize()
                    orig_output = orig_output + bias

                    output = generate_zeros(m, n, dtype)
                    coc_ops.matmul_all_reduce(input_left, input_right, output, bias)
                    torch.npu.synchronize()

                    err_rate = is_similar(orig_output, output)
                    assert err_rate < 5e-3

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_coc_all_gather_matmul(self):
        from mindspeed.ops.lcal_functional import coc_ops
        for shape_idx in range(total_shape_num):
            for dtype in dtype_list:
                m, k, n = shape_list[shape_idx][0], shape_list[shape_idx][1], shape_list[shape_idx][2]
                input1, input2, bias = generate_input_tensors(m // TestCoC.world_size, k, n, dtype)

                for trans_case in trans_case_list:
                    trans_b = False if trans_case == "trans_none" else True
                    input_left = input1
                    input_right = input2.t().contiguous() if trans_b else input2

                    orig_comm_output = generate_zeros(m, k, dtype)
                    torch.distributed._all_gather_base(orig_comm_output, input_left)
                    torch.npu.synchronize()
                    orig_output = torch.matmul(orig_comm_output, input_right.t()) if trans_b else torch.matmul(
                        orig_comm_output, input_right)
                    orig_output = orig_output + bias

                    output = generate_zeros(m, n, dtype)
                    coc_ops.all_gather_matmul(input_left, input_right, output, bias)

                    err_rate = is_similar(orig_output, output)
                    assert err_rate < 5e-3

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_coc_all_gather_matmul_v2(self):
        from mindspeed.ops.lcal_functional import coc_ops
        for shape_idx in range(total_shape_num):
            for dtype in dtype_list:
                m, k, n = shape_list[shape_idx][0], shape_list[shape_idx][1], shape_list[shape_idx][2]
                input1, input2, bias = generate_input_tensors(m // TestCoC.world_size, k, n, dtype)

                for trans_case in trans_case_list:
                    trans_b = False if trans_case == "trans_none" else True
                    input_left = input1
                    input_right = input2.t().contiguous() if trans_b else input2

                    orig_comm_output = generate_zeros(m, k, dtype)
                    torch.distributed._all_gather_base(orig_comm_output, input_left)
                    torch.npu.synchronize()
                    orig_output = torch.matmul(orig_comm_output, input_right.t()) if trans_b else torch.matmul(
                        orig_comm_output, input_right)
                    orig_output = orig_output + bias

                    output = generate_zeros(m, n, dtype)
                    comm_output = generate_zeros(m, k, dtype)
                    coc_ops.all_gather_matmul_v2(input_left, input_right, output, comm_output, bias)

                    err_rate = is_similar(orig_output, output)
                    err_rate_comm = is_similar(orig_comm_output, comm_output)
                    assert err_rate < 5e-3
                    assert err_rate_comm < 5e-3

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_coc_matmul_reduce_scatter(self):
        from mindspeed.ops.lcal_functional import coc_ops
        for shape_idx in range(total_shape_num):
            for dtype in dtype_list:
                m, k, n = shape_list[shape_idx][0], shape_list[shape_idx][1], shape_list[shape_idx][2]
                input1, input2, bias = generate_input_tensors(m, k, n, dtype)

                for trans_case in trans_case_list:
                    trans_b = False if trans_case == "trans_none" else True
                    input_left = input1
                    input_right = input2.t().contiguous() if trans_b else input2

                    orig_output = generate_zeros(m // TestCoC.world_size, n, dtype)
                    mm_output = torch.matmul(input_left, input_right.t()) if trans_b else torch.matmul(input_left,
                                                                                                       input_right)
                    torch.distributed._reduce_scatter_base(orig_output, mm_output)
                    torch.npu.synchronize()
                    orig_output = orig_output + bias

                    output = generate_zeros(m // TestCoC.world_size, n, dtype)
                    coc_ops.matmul_reduce_scatter(input_left, input_right, output, bias)
                    torch.npu.synchronize()

                    err_rate = is_similar(orig_output, output)
                    assert err_rate < 5e-3
