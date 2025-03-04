# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Dict
from dataclasses import dataclass
from mindspeed.multi_modal.dist_train.config.dist_train_config import ContextKey


CK = ContextKey()


@dataclass
class DistTrainModelConfig:
    name: str
    model_index: int
    world_size: int
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    main_dp: bool = False
    forward_only: bool = False


def get_single_config(*args, **kwargs) -> Dict:
    """ Get single dist_train model config """
    config = DistTrainModelConfig(*args, **kwargs)
    config_dict = {
        CK.NAME: config.name,
        CK.MODEL_INDEX: config.model_index,
        CK.WORLD_SIZE: config.world_size,
        CK.MAIN_DP: config.main_dp,
        CK.TENSOR_MODEL_PARALLEL_SIZE: config.tp_size,
        CK.PIPELINE_MODEL_PARALLEL_SIZE: config.pp_size,
        CK.CONTEXT_PARALLEL_SIZE: config.cp_size,
        CK.FORWARD_ONLY: config.forward_only,
    }
    return config_dict


def make_whole_config(configs, use_multiparam_send_recv: bool = False, model_name="internvl2"):
    """ Get whole dist_train config """
    if configs is None or configs == []:
        return None
    whole_config = {
        CK.USE_MULTIPARAM_SEND_RECV: use_multiparam_send_recv,
        CK.MODEL_CONFIG: configs,
        CK.MODEL_NAME: model_name
    }
    return whole_config
