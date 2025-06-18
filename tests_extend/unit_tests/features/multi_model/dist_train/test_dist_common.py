# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import itertools
import json
import tempfile
import socket
from unittest.mock import patch, MagicMock, call

import unittest
import pytest
import torch
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu
from mindspeed.core.multi_modal.dist_train import utils
from mindspeed.core.multi_modal.dist_train import dist_communication as dc
from mindspeed.core.multi_modal.dist_train import dist_train_config as dist_config
from mindspeed.core.multi_modal.dist_train.dist_ranks_match import (
    generate_model_comm_ranks,
    _MODEL_COMM_RANKS
)


dc.init_tensor_sync_tool()


class TestTensorSyncTool(unittest.TestCase):
    def setUp(self):
        self.tool = dc.TensorSyncTool()

    def test_encode_decode_header(self):
        test_cases = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.zeros(5, requires_grad=True, dtype=torch.float16),
            torch.ones((2,), dtype=torch.int32),
        ]
        for tensor in test_cases:
            header = self.tool.encode_tensor_header(tensor)
            dtype, shape, requires_grad = self.tool.decode_tensor_header(header)

            self.assertEqual(dtype, tensor.dtype)
            self.assertEqual(shape, list(tensor.shape))
            self.assertEqual(requires_grad, tensor.requires_grad)

    def test_too_many_dims(self):
        tensor = torch.randn(1, 2, 3, 4, 5, 6, 7, 8)
        with self.assertRaises(ValueError):
            self.tool.encode_tensor_header(tensor)


class TestCommunicationFunctions(unittest.TestCase):
    def setUp(self):
        self.rank_patcher = patch(
            'mindspeed.core.multi_modal.dist_train.dist_communication.get_global_pipeline_parallel_rank'
        )
        self.mock_get_rank = self.rank_patcher.start()
        self.mock_get_rank.return_value = 0

    def tearDown(self):
        self.rank_patcher.stop()

    @patch('mindspeed.core.multi_modal.dist_train.dist_communication._send_tensor')
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication._recv_tensor')
    def test_send_recv(self, mock_recv, mock_send):
        tensor = torch.randn(3, 4)
        dc.send_recv(tensor, False, [1, 2])
        mock_send.assert_called_once_with(tensor, [1, 2])
        mock_recv.assert_not_called()

        mock_recv.reset_mock()
        mock_send.reset_mock()
        mock_recv.return_value = [torch.tensor([1])]
        result = dc.send_recv(None, True, [3])
        self.assertEqual(len(result), 1)
        mock_send.assert_not_called()
        mock_recv.assert_called_once_with([3])

        mock_recv.reset_mock()
        mock_send.reset_mock()
        dc.send_recv(tensor, True, [4])
        mock_send.assert_called_once_with(tensor, [4])
        mock_recv.assert_called_once_with([4])

        self.mock_get_rank.return_value = 1
        mock_recv.reset_mock()
        mock_send.reset_mock()
        dc.send_recv(tensor, True, [5])
        self.assertEqual(mock_send.call_args[0][1], [5])
        self.assertEqual(mock_recv.call_args[0][0], [5])

    @patch('mindspeed.core.multi_modal.dist_train.dist_communication.send_tensor_list')
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication.recv_tensor_list')
    def test_send_recv_tensor_list(self, mock_recv_list, mock_send_list):
        tensor_list = [torch.randn(2), torch.randn(3)]

        dc.send_recv_tensor_list(tensor_list, False, [1])
        mock_send_list.assert_called_once_with(tensor_list, [1])
        mock_recv_list.assert_not_called()

        mock_recv_list.return_value = [[torch.tensor([1]), torch.tensor([2])]]
        result = dc.send_recv_tensor_list(None, True, [2])
        self.assertEqual(len(result[0]), 2)
        mock_send_list.assert_called_once()

    @patch('torch.distributed.send')
    @patch('torch.distributed.recv')
    def test_send_recv_tensor_list_operations(self, mock_recv, mock_send):
        with patch(
                'mindspeed.core.multi_modal.dist_train.dist_communication._send_tensor'
        ) as mock_send_tensor, patch(
            'mindspeed.core.multi_modal.dist_train.dist_communication._recv_tensor'
        ) as mock_recv_tensor:
            tensors = [torch.randn(2), torch.randn(3)]
            dc.send_tensor_list(tensors, [1, 2])
            self.assertEqual(mock_send.call_count, 2)

            mock_recv.return_value = torch.tensor([2])
            mock_recv_tensor.side_effect = [
                [torch.tensor([1.0]), torch.tensor([2.0])],
                [torch.tensor([3.0]), torch.tensor([4.0])]
            ]
            result = dc.recv_tensor_list([1, 2])
            self.assertEqual(len(result), 0)

    @patch('mindspeed.core.multi_modal.dist_train.dist_communication._send_header')
    @patch('torch.distributed.send')
    def test_send_tensor(self, mock_send, mock_send_header):
        tensor = torch.randn(3, 4)
        dc._send_tensor(tensor, [1, 2])
        self.assertEqual(mock_send_header.call_count, 2)
        self.assertEqual(mock_send.call_count, 2)

    @patch('mindspeed.core.multi_modal.dist_train.dist_communication._recv_header')
    @patch('torch.distributed.recv')
    def test_recv_tensor(self, mock_recv, mock_recv_header):
        mock_recv_header.return_value = (torch.float32, [3, 4], False)
        result = dc._recv_tensor([1, 2])
        self.assertEqual(len(result), 2)
        self.assertEqual(mock_recv.call_count, 2)


class TestSendRecvMask(unittest.TestCase):
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication.get_dist_model_index')
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication.get_rank_number_to_model_index')
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication.get_dist_model_config')
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication._is_pipeline_first_stage')
    @patch('mindspeed.core.multi_modal.dist_train.dist_communication._is_pipeline_last_stage')
    def test_mask_generation(
            self, mock_last_stage, mock_first_stage, mock_get_config,
            mock_get_rank_map, mock_get_index
    ):
        mock_first_stage.return_value = True
        mock_last_stage.return_value = False
        mock_get_index.return_value = 2
        mock_get_rank_map.return_value = [0, 1, 2, 3]
        mock_get_config.return_value.forward_only = False

        mask = dc.generate_send_recv_mask()
        self.assertTrue(mask['recv_forward'])
        self.assertTrue(mask['send_backward'])
        self.assertFalse(mask['send_forward'])
        self.assertFalse(mask['recv_backward'])

        mock_first_stage.return_value = False
        mock_last_stage.return_value = True
        mask = dc.generate_send_recv_mask()
        self.assertTrue(mask['send_forward'])
        self.assertTrue(mask['recv_backward'])

        mock_get_config.return_value.forward_only = True
        mask = dc.generate_send_recv_mask()
        self.assertTrue(mask['send_forward'])
        self.assertFalse(mask['recv_backward'])


class TestDistRanksMatch:
    @pytest.fixture(autouse=True)
    def init_dist(self):
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "5489"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=1,
            rank=0
        )
        yield

        dist.destroy_process_group()
        torch.cuda.empty_cache()

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        _MODEL_COMM_RANKS.clear()
        yield
        _MODEL_COMM_RANKS.clear()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.get_all_config_size.return_value = 2
        return config

    @pytest.fixture
    def mock_dist(self):
        dist_mock = MagicMock()
        dist_mock.get_rank.return_value = 0
        return dist_mock

    def test_global_state_check(self, mock_config):
        _MODEL_COMM_RANKS[1] = [1, 2, 3]
        mock_config.get_all_config_size.return_value = 3

        with pytest.raises(RuntimeError, match="Get config size .* is not equal to 2"):
            generate_model_comm_ranks([], [], [], [])

        assert _MODEL_COMM_RANKS[1] == [1, 2, 3]

    def test_empty_tp_ranks(self):
        with pytest.raises(ValueError, match="tp ranks must not empty"):
            generate_model_comm_ranks([[1]], [], [[1]], [[1]])

        with pytest.raises(ValueError, match="tp ranks must not empty"):
            generate_model_comm_ranks([[1]], [[1]], [[1]], [])

    def test_complex_mapping(self, mock_config, mock_dist):
        pp_prev = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        tp_prev = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        pp_last = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        tp_last = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

        def mock_get_size_list(total, num, min_val=1):
            return [total // num] * num

        with patch('mindspeed.core.multi_modal.dist_train.dist_ranks_match.get_size_list',
                   new=mock_get_size_list):
            generate_model_comm_ranks(pp_prev, tp_prev, pp_last, tp_last)

        assert len(_MODEL_COMM_RANKS) == 9
        for _, v in _MODEL_COMM_RANKS.items():
            assert len(v) > 0
            assert None not in v

    def test_print_output(self, mock_config, mock_dist, capsys):
        pp_prev = [[0, 1]]
        tp_prev = [[0, 1]]
        pp_last = [[0, 1]]
        tp_last = [[0, 1]]

        mock_dist.get_rank.return_value = 1
        generate_model_comm_ranks(pp_prev, tp_prev, pp_last, tp_last)

        captured = capsys.readouterr()
        assert "rank=0" in captured.out
        assert "num_take_last" in captured.out

    def test_none_key_check(self, mock_config, mock_dist):
        pp_prev = [[0, 1], [2, 3]]
        tp_prev = [[0, 1], [2, 3]]
        pp_last = [[0, 1], [2, 3]]
        tp_last = [[None, 1], [2, 3]]

        generate_model_comm_ranks(pp_prev, tp_prev, pp_last, tp_last)
        assert any(len(v) > 2 for v in _MODEL_COMM_RANKS.values())


class TestDistConfig(unittest.TestCase):
    def setUp(self):
        dist_config._ALL_CONFIG = {}
        dist_config._RANK_NUMBER_TO_MODEL_INDEX = []
        dist_config._RANK_NUMBER_TO_MODEL_NAME = []
        dist_config._NUMBER_OF_MODELS = 0
        dist_config._USE_MULTIPARAM_SEND_RECV = False
        dist_config._ALL_DIST_MODEL_INDEX = []
        dist_config._ALL_DIST_MODEL_NAME = []
        dist_config._ALL_DIST_MODEL_CONFIG = []

    def create_valid_config(self):
        return {
            dist_config.CK.MODEL_NAME: "opensoraplan1.3",
            dist_config.CK.USE_MULTIPARAM_SEND_RECV: True,
            dist_config.CK.MODEL_CONFIG: [
                {
                    dist_config.CK.NAME: "vae",
                    dist_config.CK.MODEL_INDEX: 0,
                    dist_config.CK.WORLD_SIZE: 2,
                    dist_config.CK.MAIN_DP: True
                },
                {
                    dist_config.CK.NAME: "dit",
                    dist_config.CK.MODEL_INDEX: 1,
                    dist_config.CK.WORLD_SIZE: 2
                }
            ]
        }

    def test_model_config_init_invalid(self):
        with self.assertRaises(ValueError):
            dist_config.ModelConfig({dist_config.CK.WORLD_SIZE: -1}, 0)

        with self.assertRaises(ValueError):
            dist_config.ModelConfig({
                dist_config.CK.WORLD_SIZE: 4,
                dist_config.CK.TENSOR_MODEL_PARALLEL_SIZE: 0
            }, 0)

        with self.assertRaises(TypeError):
            dist_config.ModelConfig({
                dist_config.CK.WORLD_SIZE: 4,
                dist_config.CK.FORWARD_ONLY: "not-a-bool"
            }, 0)

    def test_check_config_missing_keys(self):
        config = self.create_valid_config()
        del config[dist_config.CK.MODEL_CONFIG]
        with self.assertRaises(KeyError):
            dist_config._check_config(config)

    def test_check_config_duplicate_names(self):
        config = self.create_valid_config()
        config[dist_config.CK.MODEL_CONFIG][1][dist_config.CK.NAME] = "vae"
        with self.assertRaises(ValueError):
            dist_config._check_config(config)

    def test_check_config_non_continuous_indexes(self):
        config = self.create_valid_config()
        config[dist_config.CK.MODEL_CONFIG][1][dist_config.CK.MODEL_INDEX] = 2
        with self.assertRaises(ValueError):
            dist_config._check_config(config)

    def test_check_config_invalid_model_name_sequence(self):
        config = self.create_valid_config()
        config[dist_config.CK.MODEL_CONFIG][0][dist_config.CK.NAME] = "dit"
        config[dist_config.CK.MODEL_CONFIG][1][dist_config.CK.NAME] = "vae"
        with self.assertRaises(ValueError):
            dist_config._check_config(config)

    def test_merge_dist_train_args_json(self):
        config = self.create_valid_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({dist_config.CK.DIST_CONFIG: config}, f)
            file_path = f.name

        try:
            dist_config.merge_dist_train_args(file_path)

            self.assertEqual(dist_config._NUMBER_OF_MODELS, 2)
            self.assertTrue(dist_config._USE_MULTIPARAM_SEND_RECV)
            self.assertEqual(dist_config.get_all_config_size(), 2)
        finally:
            os.unlink(file_path)

    def test_merge_dist_train_args_invalid_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            f.write("invalid content")
            f.flush()
            with self.assertRaises(TypeError):
                dist_config.merge_dist_train_args(f.name)

    def test_getters_with_global_state(self):
        config = self.create_valid_config()
        dist_config._set_config(config)

        self.assertEqual(len(dist_config.get_all_config()), 2)
        self.assertEqual(dist_config.get_all_config_size(), 2)
        self.assertEqual(dist_config.get_rank_number_to_model_index(), [0, 0, 1, 1])
        self.assertEqual(dist_config.get_rank_number_to_model_name(), ["vae", "vae", "dit", "dit"])

        self.assertEqual(dist_config.get_dist_model_name(rank=0), "vae")
        self.assertEqual(dist_config.get_dist_model_name(rank=2), "dit")
        self.assertEqual(dist_config.get_dist_model_name(global_index=0), "vae")

        vae_config = dist_config.get_dist_model_config(name="vae")
        self.assertEqual(vae_config.name, "vae")
        self.assertEqual(vae_config.world_size, 2)

        self.assertEqual(dist_config.get_dist_model_index(rank=0), 0)
        self.assertEqual(dist_config.get_dist_model_index(rank=2), 1)

        self.assertEqual(dist_config.get_dist_global_model_index(rank=0), 0)
        self.assertEqual(dist_config.get_dist_global_model_index(rank=2), 1)

        self.assertTrue(dist_config.is_use_multiparam_send_recv())
        self.assertFalse(dist_config.is_forward_only_model(name="vae"))

        dist_config._ALL_CONFIG["vae"].forward_only = True
        self.assertTrue(dist_config.is_forward_only_model(name="vae"))

    @patch('torch.distributed.get_rank', return_value=1)
    def test_getters_with_current_rank(self, mock_get_rank):
        config = self.create_valid_config()
        dist_config._set_config(config)

        self.assertEqual(dist_config.get_dist_model_name(), "vae")
        self.assertEqual(dist_config.get_dist_model_index(), 0)
        self.assertEqual(dist_config.get_dist_global_model_index(), 0)
        self.assertEqual(dist_config.get_dist_model_config().name, "vae")

    def test_validate_configs_world_size(self):
        class Args:
            world_size = 4

        config = self.create_valid_config()
        dist_config._set_config(config)

        dist_config.validate_configs_world_size(Args())

        Args.world_size = 3
        with self.assertRaises(ValueError):
            dist_config.validate_configs_world_size(Args())

    def test_error_handling_in_getters(self):
        with self.assertRaises(IndexError):
            dist_config.get_dist_model_name(rank=0)

        config = self.create_valid_config()
        dist_config._set_config(config)

        with self.assertRaises(IndexError):
            dist_config.get_dist_model_name(rank=10)

        with self.assertRaises(ValueError):
            dist_config.get_dist_model_name(global_index=10)

        with self.assertRaises(KeyError):
            dist_config.get_dist_model_config(name="invalid")

    def test_main_dp_validation(self):
        config = self.create_valid_config()

        config[dist_config.CK.MODEL_CONFIG][1][dist_config.CK.MAIN_DP] = True
        with self.assertRaises(ValueError):
            dist_config._check_config(config)

        config = self.create_valid_config()
        config[dist_config.CK.MODEL_CONFIG][0][dist_config.CK.MAIN_DP] = "not-bool"
        with self.assertRaises(TypeError):
            dist_config._check_config(config)


class TestDistTrainUtils:
    @pytest.fixture
    def mock_config(self):
        config1 = MagicMock(spec=dist_config.CK)
        config1.main_dp = True
        config1.world_size = 16
        config1.tensor_model_parallel_size = 2
        config1.pipeline_model_parallel_size = 2
        config1.context_parallel_size = 1

        config2 = MagicMock(spec=dist_config.CK)
        config2.main_dp = False
        config2.world_size = 8
        config2.tensor_model_parallel_size = 1
        config2.pipeline_model_parallel_size = 1
        config2.context_parallel_size = 1

        return {"config1": config1, "config2": config2}

    @pytest.fixture
    def mock_get_all_config(self, mock_config):
        with patch('mindspeed.core.multi_modal.dist_train.utils.get_all_config') as mock:
            mock.return_value = mock_config
            yield mock

    @pytest.fixture
    def mock_is_in_subworld(self):
        with patch('mindspeed.core.multi_modal.dist_train.utils.is_in_subworld') as mock:
            yield mock

    @pytest.fixture
    def mock_get_data_parallel_world_size(self):
        with patch('mindspeed.core.multi_modal.dist_train.utils.get_data_parallel_world_size') as mock:
            yield mock

    def test_get_global_data_parallel_size(self, mock_get_all_config):
        expected = 4
        result = utils.get_global_data_parallel_size()
        assert result == expected

    def test_get_global_data_parallel_size_no_main_dp(self, mock_get_all_config):
        for config in mock_get_all_config.return_value.values():
            config.main_dp = False
        with pytest.raises(AssertionError, match="No Main DP"):
            utils.get_global_data_parallel_size()

    def test_need_inner_data_parallel_not_in_vit_subworld(self, mock_is_in_subworld):
        mock_is_in_subworld.return_value = False
        result = utils.need_inner_data_parallel()
        assert result is False

    def test_need_inner_data_parallel_no_main_dp(self, mock_is_in_subworld, mock_get_all_config):
        mock_is_in_subworld.return_value = True
        for config in mock_get_all_config.return_value.values():
            config.main_dp = False
        result = utils.need_inner_data_parallel()
        assert result is False

    def test_need_inner_data_parallel_dp_ratio_leq_1(self, mock_is_in_subworld, mock_get_all_config, mock_get_data_parallel_world_size):
        mock_is_in_subworld.return_value = True
        mock_get_data_parallel_world_size.return_value = 4
        with patch('mindspeed.core.multi_modal.dist_train.utils.get_global_data_parallel_size') as mock:
            mock.return_value = 4
            result = utils.need_inner_data_parallel()
        assert result is False