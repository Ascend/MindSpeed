import json
import logging
import os
import stat
import time
import pickle
from argparse import Namespace

from mindspeed.auto_tuning.utils.logger import init_logger, get_logger
from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_tuning.module.model_performance import ModelPerformance
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_tuning.module.search.search_engine import search_demo
from mindspeed.auto_tuning.utils.runner.model_executor import ExecutorFlag, ModelExecutor
from mindspeed.auto_tuning.utils.runner.torchrun_runner import TorchRunRunner
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.generate_profiling_configs import generate_profiling_configs
from mindspeed.auto_tuning.utils.utils import get_prof_dir
from mindspeed.auto_tuning.utils.restricted_unpickler import restricted_loads


def auto_tuning(args: Namespace, working_dir: str):
    init_logger(args.auto_tuning_log_level)
    logger = get_logger("main")
    start_time = time.time()
    executor = ModelExecutor(TorchRunRunner())

    # Force refresh model args just in case model has been modified after previous run.
    logger.info("<==========Begin to parse args==========>")
    executor.execute(working_dir, flag=ExecutorFlag.PARSE_ARGS)
    hardware_parse_path = os.path.join(working_dir, Hardware.HARDWARE_PARSE_FILENAME)
    args_parse_path = os.path.join(working_dir, ModelConfig.ARGS_PARSE_FILENAME)
    try:
        with open(hardware_parse_path, mode="rb") as file:
            hardware: Hardware = restricted_loads(file)  # type: ignore
        with open(args_parse_path, mode="rb") as file:
            model_config: ModelConfig = restricted_loads(file)  # type: ignore
    except pickle.UnpicklingError as e:
        logger.error(f"Incorrect pickle format. UnpicklingError: {e}")
        raise e
    Hardware().load(hardware)
    model_config.disable_cp_flag = False
    logger.info("<==========Finished parsing args==========>")

    # Memory modeling
    MemoryModeling.set_model_cfg(model_config)
    static_list, dynamic_list = MemoryModeling.generate_mem_modeling_profiling_list()
    logger.info("<==========Begin to profile static memory==========>")
    for cfg, filename in static_list:
        if not os.path.exists(os.path.join(working_dir, filename)):
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            pkl_filename = os.path.join(working_dir, f'ootb_{Hardware().node_rank}.pkl')
            with os.fdopen(os.open(pkl_filename, flags, mode=mode), 'wb') as f:
                pickle.dump(cfg, f)
            executor.execute(working_dir, output_filename=filename, cfg=cfg, flag=ExecutorFlag.PARSE_MODEL)
    logger.info("<==========Finished profiling static memory==========>")
    logger.info("<==========Begin to profile dynamic memory==========>")
    for cfg in dynamic_list:
        path = os.path.join(working_dir, get_prof_dir(cfg))
        if not os.path.exists(path):
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            pkl_filename = os.path.join(working_dir, f'ootb_{Hardware().node_rank}.pkl')
            with os.fdopen(os.open(pkl_filename, flags, mode=mode), 'wb') as f:
                pickle.dump(cfg, f)
            executor.execute(working_dir, output_filename=path, cfg=cfg, flag=ExecutorFlag.PROFILE)
    logger.info("<==========Finished profiling dynamic memory==========>")
    MemoryModeling.modeling(working_dir)
    model_parser_end_time = time.time()
    logger.info("Model parser cost time: %sms", str((model_parser_end_time - start_time) * 1000))

    hardware_config = Hardware()
    profiling_cfg_list = generate_profiling_configs(model_config)

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
        res_dir = f"{working_dir}/{get_prof_dir(profiling_cfg)}"
        if not os.path.exists(res_dir):
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            pkl_filename = os.path.join(working_dir, f'ootb_{Hardware().node_rank}.pkl')
            with os.fdopen(os.open(pkl_filename, flags, mode=mode), 'wb') as f:
                pickle.dump(profiling_cfg, f)
            executor.execute(working_dir, output_filename=res_dir, cfg=profiling_cfg, flag=ExecutorFlag.PROFILE)

        profiling_node_parse = GatherNodeProfiling(res_dir)
        profiling_res = profiling_node_parse.fuse_node_pkl()

        profiling_results.append([profiling_cfg, profiling_res])
        count += 1

    profiling_and_parser_end_time = time.time()

    # Performance Modeling
    model_performance = ModelPerformance(hardware_config, model_config, working_dir)
    model_performance.get_profiling_info(profiling_results)

    final_cfgs, unsampled_profiling = search_demo(model_config=model_config,
                                                  perf_obj_function=model_performance.performance,
                                                  working_dir=working_dir)
    logger.info("model config is that:\n%s", str(model_config))
    logger.info("hardware config is that:\n%s", str(hardware_config))

    search_cfg_end_time = time.time()
    logger.info(">>>>>> Generate profiling config cost time: %sms",
                str((generate_profiling_config_end_time - model_parser_end_time) * 1000))
    logger.info(">>>>>> Profiling and parser cost time: %sms",
                str((profiling_and_parser_end_time - generate_profiling_config_end_time) * 1000))
    logger.info(">>>>>> Search_cfg cost time: %sms",
                str((search_cfg_end_time - profiling_and_parser_end_time) * 1000))
    logger.info(">>>>>> Total cost time: %sms",
                str((search_cfg_end_time - start_time) * 1000))

    logger.info("<==========Final config generated==========>")
    logger.info("The recommended configs are:")
    for i, final_cfg in enumerate(final_cfgs):
        if final_cfg:
            logger.info("<==========Top #%s config==========>", str(i))
            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug("\n%s", str(final_cfg))
            else:
                logger.info("\n%s", ModelConfig.__str__(final_cfg))
    logger.info("<==========Launch training==========>")
