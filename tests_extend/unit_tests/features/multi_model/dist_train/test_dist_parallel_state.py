# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from unittest.mock import patch, MagicMock
import pytest
import torch
import mindspeed.core.multi_modal.dist_train.dist_parallel_state as dps


class MockDistributed:
    is_initialized = True
    rank = 0
    world_size = 8

    @staticmethod
    def get_rank():
        return MockDistributed.rank

    @staticmethod
    def get_world_size():
        return MockDistributed.world_size

    @staticmethod
    def new_group(ranks, **kwargs):
        return MagicMock()


class TestDetachedSubWorld:
    def test_initialization(self):
        subworld = dps.DetachedSubWorld("test", 0, [0, 1, 2])
        assert subworld.name == "test"
        assert subworld.ranks == [0, 1, 2]
        assert subworld.start_rank == 0
        assert subworld.tensor_model_parallel_group is None

    def test_repr(self):
        subworld = dps.DetachedSubWorld("test", 0, [0])
        with patch("torch.distributed.get_process_group_ranks", return_value=[0]):
            assert "model=test" in repr(subworld)


class TestGlobalState:
    def test_set_global_group_by_subworld(self):
        subworld = dps.DetachedSubWorld("test", 0, [0])
        subworld.tensor_model_parallel_group = "tp_group"
        dps.set_global_group_and_ranks_by_subworld(subworld)
        assert dps._TENSOR_MODEL_PARALLEL_GROUP == "tp_group"


class TestInitializeModelParallel:
    @patch("torch.distributed.get_world_size", MockDistributed.get_world_size)
    @patch("torch.distributed.get_rank", MockDistributed.get_rank)
    @patch("torch.distributed.new_group", MockDistributed.new_group)
    @patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.get_dist_model_config")
    def test_initialize(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.tensor_model_parallel_size = 2
        mock_config.pipeline_model_parallel_size = 2
        mock_config.context_parallel_size = 1
        mock_config.world_size = 8
        mock_config.start_rank = 0
        mock_config.name = "model0"
        mock_config.main_dp = False
        mock_get_config.return_value = mock_config

        with patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.get_all_config_size", return_value=1), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.get_all_config",
                      return_value={0: mock_config}), \
                patch("mindspeed.core.multi_modal.dist_train.utils.get_global_data_parallel_size",
                      return_value=1):

            dps.initialize_model_parallel()

            assert "model0" in dps.ALL_SUB_WORLD
            assert len(dps.ALL_SUB_WORLD) == 1


class TestStateFunctions:
    @patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.set_global_group_and_ranks_by_subworld")
    @patch("mindspeed.core.multi_modal.dist_train.dist_train_config.get_dist_model_name", return_value="model0")
    @patch("torch.distributed.get_rank", MockDistributed.get_rank)  # 模拟分布式rank
    def test_subworld_decorator(self, mock_get_name, mock_set):
        @dps.subwrold_decorator
        def test_func():
            return True

        dps._CUR_SUB_WORLD = None
        assert test_func() is True
        mock_set.assert_called()

    @patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.subwrold_decorator")
    @patch("mindspeed.core.multi_modal.dist_train.dist_train_config.get_dist_model_name", return_value="model0")
    @patch("torch.distributed.get_rank", MockDistributed.get_rank)  # 模拟分布式rank
    def test_get_functions(self, mock_get_name, mock_decorator):
        dps._TENSOR_MODEL_PARALLEL_GROUP = "tp_group"
        dps._DATA_PARALLEL_GROUP = "dp_group"
        assert dps.get_tensor_model_parallel_group() is not None
        assert dps.get_data_parallel_group() is not None


class TestHelperFunctions:
    @patch("torch.distributed.get_rank", return_value=0)
    def test_is_in_subworld(self, mock_get_rank):
        dps.ALL_SUB_WORLD = {"model0": MagicMock(ranks=[0, 1, 2])}
        assert dps.is_in_subworld("model0") is True
        assert dps.is_in_subworld("invalid") is False

    @patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.get_args")
    def test_dist_train_checks(self, mock_get_args):
        mock_args = MagicMock()
        mock_args.dist_train = True
        mock_get_args.return_value = mock_args

        with patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.is_in_subworld", return_value=True):
            assert dps.is_use_dist_train_and_in_subworld("model0") is True


class TestExpertParallel:
    @patch("mindspeed.core.multi_modal.dist_train.dist_parallel_state.subwrold_decorator")
    @patch("mindspeed.core.multi_modal.dist_train.dist_train_config.get_dist_model_name", return_value="model0")
    @patch("torch.distributed.get_rank", MockDistributed.get_rank)
    def test_expert_functions(self, mock_get_name, mock_decorator):
        dps._EXPERT_MODEL_PARALLEL_GROUP = "expert_group"
        dps.get_expert_model_parallel_group()
        dps._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = 0
        assert dps.get_expert_model_parallel_world_size() == 0