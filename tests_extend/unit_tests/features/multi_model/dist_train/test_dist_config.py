# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch
from mindspeed import megatron_adaptor
from mindspeed.core.multi_modal.dist_train import dist_train_config as config
from mindspeed.core.multi_modal.dist_train.dist_parallel_state import initialize_model_parallel, \
    reset_global_group_and_ranks
from mindspeed.core.multi_modal.dist_train.dist_ranks_match import clear_model_comm_ranks
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.unit_tests.features.multi_model.dist_train.dist_train_config_utils import get_single_config, make_whole_config

CK = config.ContextKey()


class TestDistTrainConfigWithDistributed(DistributedTest):
    world_size = 4

    @staticmethod
    def set_normal_whole_config():
        """Set and return base two submodel config"""
        params = (["vit", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        config._set_config(configs)
        return configs

    def test_normal_scene(self):
        """Normal test"""
        config_ = self.set_normal_whole_config()
        all_config = config.get_all_config()
        assert len(all_config) == 2, "`get_all_config_size()` gets incorrect value."
        all_config = list(all_config.values())
        for i, cfg in enumerate(config_[CK.MODEL_CONFIG]):
            for key in cfg.keys():
                assert config_[CK.MODEL_CONFIG][i].get(key) == getattr(all_config[i], key), \
                    "`get_all_config()` gets incorrect value."

    def test_set_custom_key(self):
        """Testing for unknown key be set in dist-train model configuration."""
        k = "custom_new_key"
        v = "custom_new_value"
        params = (["vit", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        configs[k] = v
        config._clear_dist_config()
        with pytest.raises(KeyError) as exc_info:
            config._set_config(configs)
        assert exc_info.type is KeyError

    def test_model_name_missing(self):
        """Testing for necessary param not be set in dist-train model configuration."""
        params = (["vit", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = {CK.USE_MULTIPARAM_SEND_RECV: False, CK.MODEL_CONFIG: configs}
        config._clear_dist_config()
        with pytest.raises(KeyError) as exc_info:
            config._set_config(configs)
        assert "`model_name` key does not exist in DistTrain config" in str(exc_info.value)

    def test_model_config_missing(self):
        """Testing for necessary param not be set in dist-train model configuration."""
        configs = {CK.USE_MULTIPARAM_SEND_RECV: False, CK.MODEL_NAME: "internvl2"}
        config._clear_dist_config()
        with pytest.raises(KeyError) as exc_info:
            config._set_config(configs)
        assert "`model_config` key does not exist in DistTrain config" in str(exc_info.value)

    def test_model_config_empty(self):
        """Testing for necessary param not be set with value in dist-train model configuration."""
        configs = {CK.USE_MULTIPARAM_SEND_RECV: False, CK.MODEL_CONFIG: [], CK.MODEL_NAME: "internvl2"}
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "`model_config` must not be empty" in str(exc_info.value)

    def test_model_config_invalid_type(self):
        """Testing for `model_config` is not a list in dist-train model configuration."""
        configs = {CK.USE_MULTIPARAM_SEND_RECV: False, CK.MODEL_CONFIG: None, CK.MODEL_NAME: "internvl2"}
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`model_config` type must be list" in str(exc_info.value)

    def test_use_multiparam_send_recv_invalid_type(self):
        """Testing for abnormal `use_multiparam_send_recv` in dist-train model configuration."""
        params = (["vit", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        configs["use_multiparam_send_recv"] = None
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`use_multiparam_send_recv` value type must be bool" in str(exc_info.value)

    def test_model_name_invalid_type(self):
        """Testing for whole model name is `None` in dist-train model configuration."""
        params = (["vit", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        configs["model_name"] = None
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`model_name` value type must be string" in str(exc_info.value)

    def test_model_name_invalid_value(self):
        """Testing for submodel name not match whole model name in dist-train model configuration."""
        params = (["vit", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        configs["model_name"] = "llama"
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "`model_name` current not support" in str(exc_info.value)

    def test_set_model_invalid_key(self):
        """Testing for unknown submodel key be set in dist-train model configuration."""
        k = "custom_new_key"
        v = "custom_new_value"
        param1 = ["vit", 0, 2]
        param2 = ["gpt", 1, 2]
        config1 = get_single_config(*param1)
        config1[k] = v
        config2 = get_single_config(*param2)
        configs = make_whole_config([config1, config2])
        config._clear_dist_config()
        with pytest.raises(KeyError) as exc_info:
            config._set_config(configs)
        assert exc_info.type is KeyError

    def test_missing_model(self):
        """Testing for model config missing in dist-train model configuration."""
        param = ["vit", 0, 2]
        config1 = get_single_config(*param)
        configs = make_whole_config([config1])
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "`internvl2` model current only support" in str(exc_info.value)

    def test_model_index_invalid_value(self):
        """Testing for negative model_index in dist-train model configuration."""
        params = (["vit", -1, 2], ["gpt", 0, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "`model_index` must start from 0" in str(exc_info.value)

    def test_model_index_invalid_type(self):
        """Testing for invalid model_index in dist-train model configuration."""
        params = (["vit", None, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`model_index` value type must be int" in str(exc_info.value)

    def test_model_index_not_continuous(self):
        """Testing for discontinuous model_index in dist-train model configuration."""
        params = (["vit", 0, 2], ["gpt", 2, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
            reset_global_group_and_ranks()
            clear_model_comm_ranks()
            initialize_model_parallel()
            from megatron.training import get_args
            config.validate_configs_world_size(get_args())
        assert "`model_index` must be continuous" in str(exc_info.value)

    def test_model_config_name_invalid_value(self):
        """Testing for empty name when use single model in dist-train model configuration."""
        params = (["", -1, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "`name` is not a valid string" in str(exc_info.value)

    def test_model_config_name_invalid_type(self):
        """Testing for `None` name when use single model in dist-train model configuration."""
        params = ([None, -1, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`name` value type must be str" in str(exc_info.value)

    def test_model_config_name_same(self):
        """Testing for same name in dist-train model configuration."""
        params = (["gpt", 0, 2], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "`name` is duplicate in DistTrain config" in str(exc_info.value)

    def test_model_config_name_sequence_invalid(self):
        """Testing for name and model_index do not match in dist-train model configuration."""
        params = (["gpt", 0, 2], ["vit", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "sequence is incorrect" in str(exc_info.value)

    def test_world_size_invalid_type(self):
        """Testing for `None` world_size in dist-train model configuration."""
        params = (["vit", 0, None], ["gpt", 1, 4])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "should be greater than or equal to 0" in str(exc_info.value)

    def test_world_size_invalid_value(self):
        """Testing for negative world_size in dist-train model configuration."""
        params = (["vit", 0, -1], ["gpt", 1, 5])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "should be greater than or equal to 0" in str(exc_info.value)

    def test_world_size_value_overflow(self):
        """Testing for bigger world_size in dist-train model configuration."""
        params = (["vit", 0, 5], ["gpt", 1, 2])
        configs = [get_single_config(*param) for param in params]
        configs = make_whole_config(configs)
        config._clear_dist_config()
        with pytest.raises(RuntimeError) as exc_info:
            config._set_config(configs)
            reset_global_group_and_ranks()
            clear_model_comm_ranks()
            initialize_model_parallel()
            from megatron.training import get_args
            config.validate_configs_world_size(get_args())
        assert exc_info.type is RuntimeError

    def test_main_dp_duplicate(self):
        """Testing for abnormal num of main_dp be set to `True` in dist-train model configuration."""
        param1 = ["vit", 0, 2]
        param2 = ["gpt", 1, 2]
        config1 = get_single_config(*param1)
        config1["main_dp"] = True
        config2 = get_single_config(*param2)
        config2["main_dp"] = True
        configs = make_whole_config([config1, config2])
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "Only one `main_dp` can be true" in str(exc_info.value)

    def test_main_dp_invalid_type(self):
        """Testing for abnormal type of main_dp in dist-train model configuration."""
        param1 = ["vit", 0, 2]
        param2 = ["gpt", 1, 2]
        config1 = get_single_config(*param1)
        config1["main_dp"] = None
        config2 = get_single_config(*param2)
        configs = make_whole_config([config1, config2])
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`main_dp` value type must be bool" in str(exc_info.value)

    def test_parallel_size_divisible(self):
        """Testing for `world_size` is not divisible by parallel size in dist-train model configuration."""
        param1 = ["vit", 0, 2]
        param2 = ["gpt", 1, 2]
        config1 = get_single_config(*param1)
        config1["tensor_model_parallel_size"] = 2
        config1["pipeline_model_parallel_size"] = 2
        config2 = get_single_config(*param2)
        configs = make_whole_config([config1, config2])
        config._clear_dist_config()
        with pytest.raises(ValueError) as exc_info:
            config._set_config(configs)
        assert "should be divisible by" in str(exc_info.value)

    def test_forward_only_invalid_type(self):
        """Testing for abnormal type f `forward_only` in dist-train model configuration."""
        param1 = ["vit", 0, 2]
        param2 = ["gpt", 1, 2]
        config1 = get_single_config(*param1)
        config1["forward_only"] = None
        config2 = get_single_config(*param2)
        configs = make_whole_config([config1, config2])
        config._clear_dist_config()
        with pytest.raises(TypeError) as exc_info:
            config._set_config(configs)
        assert "`forward_only` value type must be bool" in str(exc_info.value)

    def test_forward_only_invalid_value(self):
        """Testing for get correct `forward_only` in dist-train model configuration."""
        param1 = ["vit", 0, 2]
        param2 = ["gpt", 1, 2]
        config1 = get_single_config(*param1)
        config1["forward_only"] = True
        config2 = get_single_config(*param2)
        config2["forward_only"] = False
        configs = make_whole_config([config1, config2])
        config._clear_dist_config()
        config._set_config(configs)
        rank = torch.distributed.get_rank()
        if rank in (0, 1):
            assert config.is_forward_only_model() is True, "forward_only is incorrect"
        else:
            assert config.is_forward_only_model() is False, "forward_only is incorrect"

    def test_get_dist_model_index(self):
        """Testing for get correct `model_index` in dist-train model configuration."""
        config_ = self.set_normal_whole_config()
        model_indexes = [
            cfg.get(CK.MODEL_INDEX)
            for cfg in config_[CK.MODEL_CONFIG]
            for _ in range(cfg[CK.WORLD_SIZE])
        ]
        rank = torch.distributed.get_rank()
        assert model_indexes[rank] == config.get_dist_model_index(), "Getting incorrect model_index."
        for i in range(0 - self.world_size, self.world_size):
            assert model_indexes[i] == config.get_dist_model_index(i), "Getting incorrect model_index."

    def test_get_dist_global_model_index(self):
        """Testing for get correct `global_model_index` in dist-train model configuration."""
        config_ = self.set_normal_whole_config()
        global_index = 0
        global_model_indexes = []
        for cfg in config_[CK.MODEL_CONFIG]:
            global_model_indexes.extend([global_index for _ in range(cfg[CK.WORLD_SIZE])])
            global_index += 1
        rank = torch.distributed.get_rank()
        assert global_model_indexes[rank] == config.get_dist_global_model_index(), \
            "Getting incorrect global_model_index."
        for i in range(0, self.world_size):
            assert global_model_indexes[i] == config.get_dist_global_model_index(i), \
                "Getting incorrect global_model_index."
