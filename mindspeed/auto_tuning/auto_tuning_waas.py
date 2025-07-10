# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import codecs
import os
import pickle
import time
import uuid

from waas_sdk.waas_client import WaasClient
from waas_sdk.api.krb_options import KrbOptions
from waas_sdk.api.tls_options import TlsOptions
from waas_sdk.api.tuner_base_info import TunerBaseInfo
from waas_sdk.api.tuning_param import TuningParam
from waas_sdk.api.tuner_type import TunerType
from waas_sdk.api.tuning_options import TuningOptions
from waas_sdk.api.data_options import DataOptions

from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_executor import ExecutorFlag
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import (MindSpeedSettings as Settings, 
                                                                        MindSpeedSettingsPKL)
from mindspeed.auto_tuning.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_tuning.module.search.search_engine import search_demo
from mindspeed.auto_tuning.module.model_performance import ModelPerformance
from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.utils.utils import get_prof_dir, NumberConstant


def main_waas(pkl: MindSpeedSettingsPKL):
    at_uuid = str(uuid.uuid4())
    waas_client = WaasClient()

    krb_options = KrbOptions()
    krb_options.set_enable(False)
    waas_client.set_krb(krb_options)

    tls_options = TlsOptions()
    tls_options.set_enable(False)
    waas_client.set_tls(tls_options)

    waas_client.connect(Settings().waas_ip_addr, Settings().waas_ip_port, "AutoTuning")

    tuning_options = TuningOptions()
    tuning_options.set_request_timeout(60)
    tuning_client = waas_client.get_tuning(tuning_options)

    base_info = TunerBaseInfo.create(TunerType.PYTHON, "AutoTuning", "auto_tuning.AutoTuning")
    param = TuningParam()
    param.set_property("uuid", at_uuid)
    param.set_data(codecs.encode(pickle.dumps(pkl), "base64").decode())

    tuning_client.create_workspace("AutoTuning", base_info, param, -1)
    tuning_client.tune("AutoTuning", base_info, param)

    data_option = DataOptions()
    data_option.set_request_timeout(60)
    data_client = waas_client.get_kv_data_client(data_option)

    logger = get_logger("main")
    start_time = time.time()

    # Memory modeling
    MemoryModeling.set_model_cfg(Settings().model_cfg)

    count = 0
    static_list_data = str()
    while not static_list_data:
        time.sleep(1)
        count += 1
        if count > 10:
            raise RuntimeError("WAAS time out!")
        static_list_data = data_client.get(f"{at_uuid}_static_mem")
    static_list = pickle.loads(codecs.decode(static_list_data.encode(), "base64"))

    count = 0
    dynamic_list_data = str()
    while not dynamic_list_data:
        time.sleep(1)
        count += 1
        if count > 10:
            raise RuntimeError("WAAS time out!")
        dynamic_list_data = data_client.get(f"{at_uuid}_dynamic_mem")
    dynamic_list = pickle.loads(codecs.decode(dynamic_list_data.encode(), "base64"))

    logger.info("<==========Begin to profile static memory==========>")
    for cfg, filename in static_list:
        if not _check_file_exists(filename):
            Settings().executor.execute(
                filename,
                cfg=cfg,
                flag=ExecutorFlag.PARSE_MODEL
            )
    logger.info("<==========Finished profiling static memory==========>")
    logger.info("<==========Begin to profile dynamic memory==========>")
    for cfg in dynamic_list:
        if not _check_file_exists(get_prof_dir(cfg)):
            Settings().executor.execute(
                get_prof_dir(cfg),
                cfg=cfg
            )
    logger.info("<==========Finished profiling dynamic memory==========>")
    MemoryModeling.modeling(Settings().work_dir)
    model_parser_end_time = time.time()
    logger.info("Model parser cost time: %sms", str((model_parser_end_time - start_time) *\
                                                    NumberConstant.CONVERSION_TIME))

    count = 0
    profiling_cfg_data = str()
    while not profiling_cfg_data:
        time.sleep(1)
        count += 1
        if count > 10:
            raise RuntimeError("WAAS time out!")
        profiling_cfg_data = data_client.get(f"{at_uuid}_performance")
    profiling_cfg_list = pickle.loads(codecs.decode(profiling_cfg_data.encode(), "base64"))

    logger.info("profile_cfgs (tp, pp, dp, cp, ep, #layers, seq_len):")
    logger.info(",".join(
        str((cfg.tp,
             cfg.pp,
             cfg.dp,
             cfg.cp,
             cfg.ep,
             cfg.num_layers,
             cfg.seq_length))
        for cfg in profiling_cfg_list))

    generate_profiling_config_end_time = time.time()

    profiling_results = []
    logger.info("<==========Begin profiling==========>")
    logger.info("This process will run the script and get some profiling results.")
    logger.info("Please wait for a while.")
    count = 1
    for profiling_cfg in profiling_cfg_list:
        # tracking the order of profiling all over the list
        logger.info('<==========the %s/%s loop==========>', str(count), str(len(profiling_cfg_list)))
        logger.info("profile_db_configs (tp, pp, dp, cp, ep, #layers, seq_len):")
        logger.info(str([profiling_cfg.tp,
                         profiling_cfg.pp,
                         profiling_cfg.dp,
                         profiling_cfg.cp,
                         profiling_cfg.ep,
                         profiling_cfg.num_layers,
                         profiling_cfg.seq_length]))
        if not _check_file_exists(get_prof_dir(profiling_cfg)):
            Settings().executor.execute(
                get_prof_dir(profiling_cfg),
                cfg=profiling_cfg
            )

        profiling_node_parse = GatherNodeProfiling(os.path.join(Settings().work_dir, get_prof_dir(profiling_cfg)))
        profiling_res = profiling_node_parse.fuse_node_pkl()

        profiling_results.append([profiling_cfg, profiling_res])
        count += 1

    profiling_and_parser_end_time = time.time()

    # Performance Modeling
    model_performance = ModelPerformance(Settings(), Settings().model_cfg, Settings().work_dir)
    model_performance.get_profiling_info(profiling_results)

    final_cfgs, unsampled_profiling = search_demo(model_config=Settings().model_cfg,
                                                  perf_obj_function=model_performance.performance,
                                                  working_dir=Settings().work_dir)
    logger.info("model config is that:\n%s", str(Settings().model_cfg))

    search_cfg_end_time = time.time()
    logger.info(">>>>>> Generate profiling config cost time: %sms",
                str((generate_profiling_config_end_time - model_parser_end_time) *\
                    NumberConstant.CONVERSION_TIME))
    logger.info(">>>>>> Profiling and parser cost time: %sms",
                str((profiling_and_parser_end_time - generate_profiling_config_end_time) *\
                    NumberConstant.CONVERSION_TIME))
    logger.info(">>>>>> Search_cfg cost time: %sms",
                str((search_cfg_end_time - profiling_and_parser_end_time) *\
                    NumberConstant.CONVERSION_TIME))
    logger.info(">>>>>> Total cost time: %sms",
                str((search_cfg_end_time - start_time) *\
                    NumberConstant.CONVERSION_TIME))

    logger.info("<==========Final config generated==========>")
    logger.info("The recommended configs are:")
    for i, final_cfg in enumerate(final_cfgs):
        if final_cfg:
            logger.info("<==========Top #%s config==========>", str(i))
            logger.info("\n%s", str(final_cfg))
    logger.info("<==========Launch training==========>")


def _check_file_exists(filename: str) -> bool:
    return os.path.exists(os.path.join(Settings().work_dir, filename))
