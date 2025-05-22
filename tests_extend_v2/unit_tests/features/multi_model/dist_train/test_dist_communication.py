# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Tuple, List
import pytest
import torch
from mindspeed import megatron_adaptor_v2
import torch.distributed
import mindspeed.core.multi_modal.dist_train.dist_communication as comm
from mindspeed.core.multi_modal.dist_train.dist_train_config import _set_config, _clear_dist_config
from mindspeed.core.multi_modal.dist_train.dist_parallel_state import initialize_model_parallel, reset_global_group_and_ranks
from mindspeed.core.multi_modal.dist_train.dist_ranks_match import clear_model_comm_ranks
from tests_extend.commons import set_random_seed
from tests_extend.unit_tests.common import DistributedTest
from tests_extend_v2.unit_tests.features.multi_model.dist_train.dist_train_config_utils import get_single_config, make_whole_config


TENSOR_SYNC_TOOL = comm.TensorSyncTool()
COLLECTED_DTYPES = TENSOR_SYNC_TOOL.type_to_int


class TestDistCommunicationTensorSyncTool:
    @pytest.mark.parametrize('tensor_shape', ((0,), (1024, 2, 8, 64)))
    def test_tensor_sync_tool_encode_decode_shape(self, tensor_shape):
        tensor = torch.empty(tensor_shape)
        header = TENSOR_SYNC_TOOL.encode_tensor_header(tensor)
        _, tensor_shape_, _ = TENSOR_SYNC_TOOL.decode_tensor_header(header)
        assert tensor_shape_ == list(tensor.shape), f'`{tensor_shape_=}` and `{list(tensor.shape)}` should be equal'

    @pytest.mark.parametrize('dtype', COLLECTED_DTYPES)
    def test_tensor_sync_tool_encode_decode_dtype(self, dtype):
        try:
            tensor = torch.empty((1, 2), dtype=dtype)
        except RuntimeError:
            return
        header = TENSOR_SYNC_TOOL.encode_tensor_header(tensor)
        dtype_, _, _ = TENSOR_SYNC_TOOL.decode_tensor_header(header)
        assert dtype_ == tensor.dtype, f'`{dtype_=}` and `{tensor.dtype}` should be equal'

    @pytest.mark.parametrize('requires_grad', (True, False))
    def test_tensor_sync_tool_encode_decode_requires_grad(self, requires_grad):
        tensor = torch.empty((1, 2), requires_grad=requires_grad)
        header = TENSOR_SYNC_TOOL.encode_tensor_header(tensor)
        _, _, requires_grad_ = TENSOR_SYNC_TOOL.decode_tensor_header(header)
        assert requires_grad_ == tensor.requires_grad, \
            f'`{requires_grad_=}` and `{tensor.requires_grad}` should be equal'


class TestDistCommunicationWithDistributed(DistributedTest):
    world_size = 4
    dtypes = (torch.float16, torch.float32, torch.float64)
    skip_flag = any(dtype not in COLLECTED_DTYPES for dtype in dtypes)

    @staticmethod
    def _test_dist_model_send_recv_header(params):
        dtype, shape, requires_grad = params
        src, dst = 0, (1, 2, 3)
        rank = torch.distributed.get_rank()
        set_random_seed(1234)

        if rank == src:
            t = torch.zeros(shape, dtype=dtype, device=torch.npu.current_device(), requires_grad=requires_grad)
            for dst_ in dst:
                comm._send_header(t, dst_)
        elif rank in dst:
            header = comm._recv_header(src)
            assert header == params, f'{header=} should be equal to {params}'

    @pytest.mark.skipif(dtypes[0] not in COLLECTED_DTYPES, reason='Some data types may not be supported by the device')
    def test_dist_model_send_recv_header_scene_dtypes_0_dim_3_req_grad_false(self):
        self._test_dist_model_send_recv_header((self.dtypes[0], [2, 2, 4], False))

    @pytest.mark.skipif(dtypes[1] not in COLLECTED_DTYPES, reason='Some data types may not be supported by the device')
    def test_dist_model_send_recv_header_scene_dtypes_1_dim_2_req_grad_true(self):
        self._test_dist_model_send_recv_header((self.dtypes[1], [1024, 2], True))

    @pytest.mark.skipif(skip_flag, reason='Some data types may not be supported by the device')
    @pytest.mark.parametrize('src_dst_ranks', (([0], [1, 2, 3]), ([0, 1], [2, 3]), ([0, 1, 2], [3])))
    def test_dist_model_send_recv_tensor(self, src_dst_ranks: Tuple[List, List]):
        src, dst = src_dst_ranks
        rank = torch.distributed.get_rank()
        set_random_seed(1234)

        def get_manual_tensor(dyn_delta):
            return torch.rand((1024, 2, 4 + dyn_delta), dtype=self.dtypes[0], device=torch.npu.current_device())

        if rank in src:
            ts = []
            idx = src.index(rank)
            for _ in range(idx):
                ts.append(get_manual_tensor(idx))
            t = get_manual_tensor(idx)
            comm._send_tensor(t, dst)
        elif rank in dst:
            ts_ans = []
            for idx in range(len(src)):
                t = get_manual_tensor(idx)
                ts_ans.append(t)
            ts = comm._recv_tensor(src)
            for t1, t2 in zip(ts_ans, ts):
                assert torch.isclose(t1, t2).all().item(), f'An incorrect tensor is received, {t1=}, {t2=}'

    @pytest.mark.skipif(skip_flag, reason='Some data types may not be supported by the device')
    def test_send_recv_src_send_tensor_list_dst_should_get_first_tensor(self):
        rank = torch.distributed.get_rank()
        device = torch.npu.current_device()
        set_random_seed(1234)
        tensor_list = [torch.rand((1024, 1024), dtype=self.dtypes[0], device=device) for _ in range(2)]
        tensor_list[1] += tensor_list[0] + torch.ones_like(tensor_list[0])

        if rank == 0:
            comm.send_recv(tensor_list, False, [1, 2, 3])
        else:
            tensor_ = comm.send_recv(None, True, [0])
            assert len(tensor_) == 1, 'Only a list with one tensor should be received.'
            assert torch.isclose(tensor_list[0], tensor_[0]).all().item(), 'An incorrect tensor is received.'

    @pytest.mark.skipif(skip_flag, reason='Some data types may not be supported by the device')
    @pytest.mark.parametrize('src_dst_ranks', (([0], [1, 2, 3]), ([0, 1, 2], [3]), ([0, 1], [2, 3])))
    def test_send_recv_with_single_tensor(self, src_dst_ranks):
        src, dst = src_dst_ranks
        rank = torch.distributed.get_rank()
        device = torch.npu.current_device()
        set_random_seed(1234)

        dtype, shape = self.dtypes[:3], ((2, 2, 4), (2, 16), (1, 4))
        tensor_list = [torch.rand(shape_, dtype=dtype_, device=device) for shape_, dtype_ in zip(shape, dtype)]

        if rank in src:
            send_tensor = tensor_list[rank]
            comm.send_recv(send_tensor, False, dst)
        elif rank in dst:
            recv_tensor = [tensor_list[i] for i in range(len(src))]
            recv_tensor_ = comm.send_recv(None, True, src)
            assert len(recv_tensor_) == len(recv_tensor), 'The length of the received list is inconsistent.'
            for t1, t2 in zip(recv_tensor_, recv_tensor):
                assert torch.isclose(t1, t2).all().item(), 'An incorrect tensor list is received.'

    def _get_manual_tensor_list(self, dst_size):
        all_dtype_list = ([self.dtypes[0], self.dtypes[1], self.dtypes[2]],
                          [self.dtypes[2], self.dtypes[1], self.dtypes[0]],
                          [self.dtypes[2], self.dtypes[0], self.dtypes[1]],)
        all_shape_list = ([(1,), (2,), (3,)],
                          [(3,), (2,), (1,)],
                          [(1,), (3,), (2,)],)
        if not (len(all_dtype_list) == len(all_shape_list) >= self.world_size - dst_size):
            raise ValueError('This method does not support src_world_size greater than 3.')
        device = torch.npu.current_device()
        tensor_list = [[torch.rand(shape, dtype=dtype, device=device) for shape, dtype in zip(shape_list, dtype_list)]
                       for shape_list, dtype_list in zip(all_shape_list, all_dtype_list)]
        return tensor_list, all_dtype_list, all_shape_list

    @pytest.mark.skipif(skip_flag, reason='Some data types may not be supported by the device')
    @pytest.mark.parametrize('src_dst_ranks', (([0], [1, 2, 3]), ([0, 1], [2, 3]), ([0, 1, 2], [3])))
    def test_send_tensor_list_and_recv_tensor_list(self, src_dst_ranks):
        src, dst = src_dst_ranks
        set_random_seed(1234)
        rank = torch.distributed.get_rank()

        tensor_list, all_dtype_list, _ = self._get_manual_tensor_list(len(dst))

        if rank in src:
            comm.send_tensor_list(tensor_list[rank], dst)
        elif rank in dst:
            recv_tensor_list = [[tensor_list_[i] for tensor_list_ in tensor_list[:len(src)]]
                                for i in range(len(all_dtype_list[0]))]
            recv_tensor_list_ = comm.recv_tensor_list(src)
            assert all(torch.isclose(tensor_list[i], tensor_list_[i]).all().item() for i in range(len(src))
                       for tensor_list, tensor_list_ in zip(recv_tensor_list, recv_tensor_list_)), \
                'An incorrect tensor list is received.'

    @staticmethod
    def _clear_dist_train_global_arguments():
        _clear_dist_config()
        clear_model_comm_ranks()
        reset_global_group_and_ranks()

    @pytest.mark.skipif(skip_flag, reason='Some data types may not be supported by the device dtypes')
    @pytest.mark.parametrize('src_dst_ranks', (([0], [1, 2, 3]), ([0, 1], [2, 3]), ([0, 1, 2], [3])))
    def test_send_recv_tensor_list_with_tensor_list(self, src_dst_ranks: Tuple):
        src, dst = src_dst_ranks
        set_random_seed(1234)
        rank = torch.distributed.get_rank()

        # Need init config because `send_recv_tensor_list` will call `get_global_pipeline_parallel_rank`
        self._clear_dist_train_global_arguments()
        config_list = [get_single_config('vit', 0, len(src)),
                       get_single_config('gpt', 1, len(dst))]
        _set_config(make_whole_config(config_list))
        initialize_model_parallel()

        tensor_list, all_dtype_list, _ = self._get_manual_tensor_list(len(dst))

        torch.distributed.barrier()
        if rank in src:
            send_tensor_list = tensor_list[rank]
            comm.send_recv_tensor_list(send_tensor_list, False, dst)
        elif rank in dst:
            recv_tensor_list_ans = [[tensor_list_[i] for tensor_list_ in tensor_list[:len(src)]]
                                    for i in range(len(all_dtype_list[0]))]
            recv_tensor_list = comm.send_recv_tensor_list(None, True, src)
            assert isinstance(recv_tensor_list, list), 'A list should be received'
            assert len(recv_tensor_list) == len(recv_tensor_list_ans), 'The length of the list is incorrect'
            for l1, l2 in zip(recv_tensor_list, recv_tensor_list_ans):
                for t1, t2 in zip(l1, l2):
                    assert torch.isclose(t1, t2).all().item(), 'Received incorrect tensor list'

    def _test_generate_send_recv_ops(self, is_fwd_only_pair, ans_ops):
        self._clear_dist_train_global_arguments()
        config_list = [get_single_config('vit', 0, 2, pp_size=2, forward_only=is_fwd_only_pair[0]),
                       get_single_config('gpt', 1, 2, pp_size=2, forward_only=is_fwd_only_pair[1])]
        _set_config(make_whole_config(config_list))
        initialize_model_parallel()

        rank = torch.distributed.get_rank()
        ops = comm.generate_send_recv_mask(rank)
        assert ans_ops == ops, f'{ans_ops} should be equal to {ops}'

    @staticmethod
    def _make_ops(ops_value):
        rank = torch.distributed.get_rank()
        ops = {'send_forward': ops_value[rank][0], 'send_backward': ops_value[rank][1],
               'recv_forward': ops_value[rank][2], 'recv_backward': ops_value[rank][3]}
        return ops

    def test_generate_send_recv_ops_is_fwd_only_true_true(self):
        #                    | dist_train_model_0             | dist_train_model_1            |
        # -------------------------------------------------------------------------------------
        # <is_fwd_only_pair> | True                           | True                          |
        #                    | first_stage | \ | last_stage   | first_stage | \ | last_stage  |
        # <ans>              | ans[0]      | \ | ans[1]       | ans[2]      | \ | ans[3]      |
        # send_forward:      | False       | \ | True         | False       | \ | True        |
        # send_backward:     | False       | \ | False        | False       | \ | False       |
        # recv_forward:      | False       | \ | False        | False       | \ | False       |
        # recv_backward:     | False       | \ | False        | False       | \ | False       |

        is_fwd_only_pair = (True, True)
        ans = ((False, False, False, False), (True, False, False, False),
               (False, False, True, False), (False, False, False, False))
        ans = self._make_ops(ans)
        self._test_generate_send_recv_ops(is_fwd_only_pair, ans)

    def test_generate_send_recv_ops_is_fwd_only_true_false(self):
        is_fwd_only_pair = (True, False)
        ans = ((False, False, False, False), (True, False, False, False),
               (False, False, True, False), (False, False, False, False))
        ans = self._make_ops(ans)
        self._test_generate_send_recv_ops(is_fwd_only_pair, ans)

    def test_generate_send_recv_ops_is_fwd_only_false_true(self):
        is_fwd_only_pair = (False, True)
        ans = ((False, False, False, False), (True, False, False, False),
               (False, False, True, False), (False, False, False, False))
        ans = self._make_ops(ans)
        self._test_generate_send_recv_ops(is_fwd_only_pair, ans)

    def test_generate_send_recv_ops_is_fwd_only_false_false(self):
        is_fwd_only_pair = (False, False)
        ans = ((False, False, False, False), (True, False, False, True),
               (False, True, True, False), (False, False, False, False))
        ans = self._make_ops(ans)
        self._test_generate_send_recv_ops(is_fwd_only_pair, ans)
