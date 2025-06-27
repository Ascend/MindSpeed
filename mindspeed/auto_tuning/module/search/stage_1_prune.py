from typing import List

from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import MindSpeedSettings as Settings
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig


def stage_1_discrete_search_space_prune(
        mcfg: ModelConfig
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

    num_devices = Settings().search_world_size
    device_type = Settings().device_type

    valid_configs: List[SearchConfig] = list()

    # Iterate over all possible combinations of tp, cp, pp, dp, ep and zero
    # Prune tp based on device_type, tp = 1 or 8 only if running on 910B
    for tp in [2 ** i for i in range(1, Settings().devices_per_node.bit_length())]:

        for cp in range(1, num_devices // tp + 1):

            if mcfg.seq_length % cp != 0:
                continue

            # Check cp long sequence based on device_type
            if cp > 1:
                if ("910B" in device_type) and \
                        ((mcfg.seq_length // cp) < 8 * 1024):
                    continue
                if ("910_93" in device_type) and \
                        ((mcfg.seq_length // cp) < 4 * 1024):
                    continue

            for pp in range(1, num_devices // (tp * cp) + 1):

                # Check if layer_number is divisible by pp
                if mcfg.num_layers % pp != 0:
                    continue

                for dp in range(1, num_devices // (tp * cp * pp) + 1):

                    # Check device number compatibility
                    if tp * cp * pp * dp != num_devices:
                        continue

                    ep_search_domain = [None]
                    # Search ep only if is moe
                    if mcfg.num_experts:
                        ep_search_domain = list(range(1, min(cp * dp, mcfg.num_experts) + 1))
                    for ep in ep_search_domain:

                        if mcfg.num_experts and ep:
                            if (cp * dp) % ep != 0:
                                continue

                            extend_ep = tp * ep if mcfg.moe_tp_extend_ep else ep
                            if mcfg.num_experts % extend_ep != 0:
                                continue

                        layers_per_vpp_search_domain = [None]
                        # Search vpp only if pp is enabled
                        if pp > 1:
                            # Search domain drops the last possible value (layer_number // pp)
                            # due to the constraint $layers_per_vpp * pp != layer_number$
                            layers_per_vpp_search_domain += \
                                [x for x in range(1, mcfg.num_layers // pp)]
                        for layers_per_vpp in layers_per_vpp_search_domain:

                            # Check if $layers_per_vpp$ not None and $layers_per_vpp * pp | layer_number$
                            if layers_per_vpp and \
                                    mcfg.num_layers % (layers_per_vpp * pp) != 0:
                                continue

                            for mbs in [1, 2]:
                                num_micro_batches = mcfg.global_batch_size // mbs
                                if num_micro_batches % dp != 0:
                                    continue
                                if num_micro_batches // (pp * dp) <= 1:
                                    continue
                                cfg = SearchConfig()
                                cfg.copy_from_config(mcfg)
                                cfg.world_size = num_devices
                                cfg.tensor_model_parallel_size = tp
                                cfg.context_parallel_size = cp
                                cfg.pipeline_model_parallel_size = pp
                                cfg.num_layers_per_virtual_pipeline_stage = layers_per_vpp
                                cfg.use_distributed_optimizer = dp * cp // (ep or 1) > 1
                                cfg.micro_batch_size = mbs
                                if mcfg.is_moe():
                                    cfg.expert_model_parallel_size = ep or 1
                                cfg.normalize()

                                valid_configs.append(cfg)

    return valid_configs
