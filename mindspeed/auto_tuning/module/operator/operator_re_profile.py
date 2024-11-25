import os
import stat
import random
import pickle

from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.utils.runner.model_executor import ExecutorFlag, ModelExecutor
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_tuning.utils.runner.torchrun_runner import TorchRunRunner
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.utils.utils import get_prof_dir
from mindspeed.auto_tuning.module.operator.operator import OperatorPerformance
from mindspeed.auto_tuning.module.operator.operator_database import OperatorHistory


logger = get_logger('operator_re_profile')


def search_operator(working_dir, search_cfg, communication, profile_count,
                    scale_flag=False):
    # After a certain amount of profiling, the rest operators have not been found will be predicted using
    # regression method.
    executor = ModelExecutor(TorchRunRunner())
    profiling_results = []
    search_cfg_list = [search_cfg]
    model_config = communication.model_cfg
    seed = 1234
    random.seed(seed)
    unsampled_profiling_info = []
    if len(search_cfg_list) > 9:
        sampled_profiling_info = random.sample(search_cfg_list, min(9, len(search_cfg_list)))
        unsampled_profiling_info = list(set(search_cfg_list) - set(sampled_profiling_info))
    else:
        sampled_profiling_info = [search_cfg]
    for profiling_config in sampled_profiling_info:
        if scale_flag:
            profiling_config = scale_para(model_config, communication, profiling_config)
        re_profiling_config = SearchConfig()
        re_profiling_config.copy_from_config(model_config)
        re_profiling_config.num_layers = profiling_config.pipeline_model_parallel_size
        re_profiling_config.seq_length = profiling_config.seq_length
        re_profiling_config.tensor_model_parallel_size = profiling_config.tensor_model_parallel_size
        re_profiling_config.pipeline_model_parallel_size = profiling_config.pipeline_model_parallel_size
        re_profiling_config.data_parallel_size = profiling_config.data_parallel_size
        re_profiling_config.context_parallel_size = profiling_config.context_parallel_size
        re_profiling_config.expert_model_parallel_size = profiling_config.expert_model_parallel_size
        re_profiling_config.prepare_for_profiling()

        from mindspeed.auto_tuning.module.hardware import Hardware
        res_dir = os.path.join(working_dir, get_prof_dir(re_profiling_config, re_profile=True))
        if not os.path.exists(res_dir):
            profile_count[0] += 1
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            pkl_filename = os.path.join(working_dir, f'ootb_{Hardware().node_rank}.pkl')
            with os.fdopen(os.open(pkl_filename, flags, mode=mode), 'wb') as f:
                pickle.dump(re_profiling_config, f)
            executor.execute(working_dir=working_dir, output_filename=res_dir, cfg=re_profiling_config,
                             flag=ExecutorFlag.PROFILE)
        profiling_node_parse = GatherNodeProfiling(res_dir)
        profiling_res = profiling_node_parse.fuse_node_pkl()

        re_profiling_config.jit_compile = search_cfg.jit_compile
        profiling_results.append([re_profiling_config, profiling_res])

        operator_list = OperatorPerformance(model_config, working_dir=working_dir)
        operator_not_found = operator_list.origin_profile_data_list.get_profinfo_list_from_profiling(
            profiling_res.forward.operator_info[-1],
            forwardflag=1)
        operator_not_found_part2 = operator_list.origin_profile_data_list.get_profinfo_list_from_profiling(
            profiling_res.backward.operator_info[-1],
            forwardflag=0)
        operator_not_found.extend(operator_not_found_part2)
        logger.debug(f'Total number of operator re profiling is {len(operator_not_found)}')
        operator_history_list = []
        for operator in operator_not_found:
            operator_history = OperatorHistory(types=operator.type,
                                               accelerator_core=operator.accelerator_core,
                                               input_shape=operator.input_shapes.replace('"', ''),
                                               output_shape=operator.output_shapes.replace('"', ''),
                                               duration=operator.duration_us,
                                               device=Hardware().device_type,
                                               jit=int(model_config.jit_compile),
                                               cann="8.0.RC2.alpha002",
                                               driver="24.1.rc2.b030",
                                               dtype=model_config.dtype.value[0])
            operator_history_list.append(operator_history.convert_to_dict())
        operator_list.db.operator_history_dao.insert_history(operator_history_list)
        operator_list.db.operator_profiling_dao.insert_history(operator_history_list)
    return unsampled_profiling_info


def generate_scale_config(model_config):
    scale_config = model_config.copy()
    scale_config.num_layers = 256

    # parameter need to be adjusted
    scale_config.tensor_model_parallel_size = 64
    scale_config.num_attention_heads = 512
    scale_config.hidden_size = 65536
    scale_config.ffn_hidden_size = 229376
    scale_config.context_parallel_size = 32
    scale_config.seq_length = 131072
    scale_config.max_position_embeddings = 131072
    scale_config.expert_model_parallel_size = 32
    scale_config.num_experts = 32
    scale_config.pipeline_model_parallel_size = 16
    scale_config.normalize()
    return scale_config


def scale_para(model_config, communication, search_cfg, test=False):
    # load base parallel model config
    tp = search_cfg.tensor_model_parallel_size
    cp = search_cfg.context_parallel_size
    pp = search_cfg.pipeline_model_parallel_size
    ep = search_cfg.expert_model_parallel_size
    dp = search_cfg.data_parallel_size

    if pp % 2 != 0 and pp != 1:
        logger.warning('warning: pp value set is not even.')

    # load hardware config
    # use test because in a mock situation, we do not have the real device number
    if not test:
        num_nodes = communication.hardware.num_nodes
        num_devices = communication.hardware.num_devices
    else:
        num_nodes = 8
        num_devices = 2 * 8
    num_devices_ootb = 16

    if not test:
        # load model config
        num_layers = communication.model_cfg.num_layers
        num_attention_heads = communication.model_cfg.num_attention_heads
        hidden_size = communication.model_cfg.hidden_size
        ffn_hidden_size = communication.model_cfg.ffn_hidden_size
        num_experts = communication.model_cfg.num_experts
        sequence_length = communication.model_cfg.seq_length
    else:
        # for test only test whether the function works fine
        num_layers = model_config.num_layers
        num_attention_heads = model_config.num_attention_heads
        hidden_size = model_config.hidden_size
        ffn_hidden_size = model_config.ffn_hidden_size
        num_experts = model_config.num_experts
        sequence_length = model_config.seq_length

    scale_factor = 2  # here use default tp value 8 or 4
    # directly scale pp down to 1
    pp_scale_factor = pp
    scale_tp, scale_cp, scale_pp, scale_ep, scale_dp = tp, cp, pp, ep, dp
    scale_num_layers = num_layers
    scale_num_attention_heads = num_attention_heads
    scale_hidden_size = hidden_size
    scale_ffn_hidden_size = ffn_hidden_size
    scale_num_experts = num_experts
    scale_sequence_length = sequence_length
    scale_space = scale_tp * scale_cp * scale_pp
    if pp >= 2:
        scale_pp //= pp_scale_factor
        scale_num_layers //= num_layers
        scale_space = scale_tp * scale_cp * scale_pp
    logger.debug(f"Search configs is\n{search_cfg}")

    while scale_space > num_devices_ootb:
        logger.debug(f'the scale space is {scale_space}, the scale_tp is {scale_tp}, the scale_cp is {scale_cp}, '
                     f'the scale_pp is {scale_pp}, the scale_ep is {scale_ep}')
        if scale_cp >= 4:
            scale_cp //= scale_factor
            scale_sequence_length //= scale_factor
            scale_space = scale_tp * scale_cp * scale_pp
            continue
        if scale_tp >= 4:
            scale_tp //= scale_factor
            scale_num_attention_heads //= scale_factor
            scale_hidden_size //= scale_factor
            scale_ffn_hidden_size //= scale_factor
            scale_space = scale_tp * scale_cp * scale_pp
            continue

    scale_dp = num_devices_ootb // (scale_tp * scale_cp * scale_pp)
    while scale_dp * scale_cp < scale_ep:
        scale_ep //= scale_factor
        scale_num_experts //= scale_factor

    # set up config group
    before_scale = SearchConfig()
    before_scale.copy_from_config(model_config)
    before_scale.tensor_model_parallel_size = scale_tp
    before_scale.context_parallel_size = scale_cp
    before_scale.pipeline_model_parallel_size = scale_pp
    before_scale.num_layers = scale_num_layers
    before_scale.num_attention_heads = scale_num_attention_heads
    before_scale.expert_model_parallel_size = scale_ep
    before_scale.hidden_size = scale_hidden_size
    before_scale.ffn_hidden_size = scale_ffn_hidden_size
    before_scale.num_experts = scale_num_experts
    before_scale.seq_length = scale_sequence_length
    before_scale.data_parallel_size = scale_dp
    return before_scale
