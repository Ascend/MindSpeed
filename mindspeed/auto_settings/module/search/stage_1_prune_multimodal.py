from typing import List

from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.utils.logger import get_logger
import copy


def stage_1_discrete_search_space_prune(
        mcfg: ModelConfig, world_size
) -> List[SearchConfig]:
    """
    Stage 1 prune is without any modeling.
    This function prunes the search space for a distributed training job based on given constraints.

    Parameters:
    layer_number (int): The total number of layers.
    total_device_number (int): The total number of devices.
    micro_batch_number (int): The number of micro-batches.
    expert_number (int): The number of experts.
    pod_limit (int, optional): The maximum number of devices in a super pod. Default is 0.
    model_in_pod (bool, optional): If True, the product of tp and pp should be less than or equal to pod_limit. Default is False.
    device_fluctuation_ratio (float, optional): The ratio of device fluctuation. Must be between 0 and 1. Default is 0.

    Returns:
    list of dict: A list of valid configurations (tp, cp, pp, dp, ep, zero which stored as a dict) that satisfy all constraints.
    """

    num_devices = world_size
    logger = get_logger("stage_1_discrete_search_space_prune")


    valid_configs: List[SearchConfig] = list()
    # Iterate over all possible combinations of tp, cp, pp, dp, ep and zero
    # Prune tp based on device_type, tp = 1 or 8 only if running on 910B

    for pp in range(1, num_devices + 1):
        # Check if layer_number is divisible by pp
        pp_list = uneven_pp(mcfg.num_layers, pp)
        for dp in range(1, num_devices // pp + 1):
            # Check device number compatibility
            if pp * dp != num_devices:
                continue
                # Search domain drops the last possible value (layer_number // pp)
                # due to the constraint $layers_per_vpp * pp != layer_number$
            for mbs in [1, 2]:
                num_micro_batches = mcfg.global_batch_size // mbs
                if num_micro_batches % dp != 0:
                    continue
                if num_micro_batches // (pp * dp) <= 1:
                    continue
                
                cfg = copy.deepcopy(SearchConfig())
                cfg.copy_from_config(mcfg)
                cfg.tensor_model_parallel_size = 1
                cfg.world_size = num_devices
                cfg.context_parallel_size = 1
                cfg.pipeline_model_parallel_size = pp
                cfg.pp_list = pp_list
                cfg.micro_batch_size = mbs
                cfg.normalize()
                valid_configs.append(cfg)
                logger.debug(f"pp_list:{pp_list}")
    logger.info(f"valid_configs len: {len(valid_configs)}")
    return valid_configs


def uneven_pp(num_layers, pp):
    pp_list = []
    if pp < 0 or pp > num_layers:
        return pp_list
    min_pp = int(num_layers / pp)
    remainder = num_layers % pp

    pp_list = [min_pp] * pp
    for item in range(remainder):
        pp_list[item + 1] += 1
    return pp_list