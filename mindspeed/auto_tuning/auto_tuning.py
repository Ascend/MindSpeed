import logging
import os
import time
from argparse import Namespace

import torch.distributed as dist

from mindspeed.auto_tuning.config.generate_profiling_configs import generate_profiling_configs
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_executor import ExecutorFlag
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import (MindSpeedSettings as Settings,
                                                                        MindSpeedSettingsPKL)
from mindspeed.auto_tuning.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_tuning.module.search.search_engine import search_demo
from mindspeed.auto_tuning.module.model_performance import ModelPerformance
from mindspeed.auto_tuning.utils.file_utils import restricted_read
from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.utils.utils import get_prof_dir, NumberConstant


def auto_tuning(args: Namespace):
    logger = get_logger("main")
    Settings().init_settings(args)

    # Force refresh model args just in case model has been modified after previous run.
    logger.info("<==========Begin to parse args==========>")
    Settings().executor.execute(
        MindSpeedSettingsPKL.FILENAME,
        flag=ExecutorFlag.PARSE_ARGS
    )
    pkl: MindSpeedSettingsPKL = restricted_read(
        os.path.join(Settings().work_dir, MindSpeedSettingsPKL.FILENAME)
    )
    Settings().load_settings_from_pkl(pkl)
    dist.init_process_group(
        backend=dist.Backend.GLOO,
        world_size=Settings().nnodes,
        rank=Settings().node_rank
    )
    logger.info("<==========Finished parsing args==========>")

    if Settings().node_rank != 0:
        Settings().executor.wait()
        return

    if Settings().waas_enabled:
        try:
            from mindspeed.auto_tuning.auto_tuning_waas import main_waas
            main_waas(pkl)
            return
        except Exception as e:
            logger.info(str(e))

    _main()


def _main():
    logger = get_logger("main")
    start_time = time.time()

    # Memory modeling
    MemoryModeling.set_model_cfg(Settings().model_cfg)
    static_list, dynamic_list = MemoryModeling.generate_mem_modeling_profiling_list()
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
    logger.info("Model parser cost time: %sms", 
                str((model_parser_end_time - start_time) * NumberConstant.CONVERSION_TIME))

    profiling_cfg_list = generate_profiling_configs(Settings().model_cfg)

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

        profiling_node_parse = GatherNodeProfiling(os.path.join(Settings().work_dir,
                                                                get_prof_dir(profiling_cfg)))
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
                str((generate_profiling_config_end_time - model_parser_end_time) * \
                    NumberConstant.CONVERSION_TIME))
    logger.info(">>>>>> Profiling and parser cost time: %sms",
                str((profiling_and_parser_end_time - generate_profiling_config_end_time) * \
                    NumberConstant.CONVERSION_TIME))
    logger.info(">>>>>> Search_cfg cost time: %sms",
                str((search_cfg_end_time - profiling_and_parser_end_time) * \
                    NumberConstant.CONVERSION_TIME))
    logger.info(">>>>>> Total cost time: %sms",
                str((search_cfg_end_time - start_time) * \
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
