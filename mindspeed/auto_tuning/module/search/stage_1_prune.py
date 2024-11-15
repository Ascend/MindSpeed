from typing import List
from dataclasses import replace

from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig


def stage_1_discrete_search_space_prune(
        mcfg: ModelConfig,
        pod_limit=0,
        model_in_pod=False,
        device_fluctuation_down_ratio=0
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

    num_devices = mcfg.global_world_size
    device_type = Hardware().device_type

    valid_configs: List[SearchConfig] = list()

    # Iterate over all possible combinations of tp, cp, pp, dp, ep and zero
    # Prune tp based on device_type, tp = 1 or 8 only if running on 910B
    tp_search_list = [2 ** i for i in range(num_devices + 1)]
    if "910B" in device_type:
        tp_search_list = [1, 8]
    for tp in tp_search_list:

        # Check if tp is less than or equal to pod_limit
        if 0 < pod_limit < tp:
            continue

        for cp in range(1, num_devices // tp + 1):

            # Check cp long sequence based on device_type
            if cp > 1:
                if ("910B" in device_type) and \
                        ((mcfg.seq_length // cp) < 8 * 1024):
                    continue
                if ("910_9" in device_type) and \
                        ((mcfg.seq_length // cp) < 4 * 1024):
                    continue

            for pp in range(1, num_devices // (tp * cp) + 1):

                # Check if tp * pp is less than or equal to pod_limit
                if model_in_pod and tp * pp > pod_limit:
                    continue
                # Check if layer_number is divisible by pp
                if mcfg.num_layers % pp != 0:
                    continue

                for dp in range(1, num_devices // (tp * cp * pp) + 1):

                    # Check device number compatibility
                    if device_fluctuation_down_ratio > 0:
                        if not ((1 - device_fluctuation_down_ratio) * num_devices < tp * cp * pp * dp <= num_devices):
                            continue
                    else:
                        if tp * cp * pp * dp != num_devices:
                            continue
                    # Check if micro_batch_number is divisible by dp
                    if mcfg.num_micro_batches % dp != 0:
                        continue
                    # Check if micro_batch_number / (pp * dp) is greater than 1
                    if mcfg.num_micro_batches // (pp * dp) <= 1:
                        continue

                    num_experts = mcfg.num_experts if mcfg.num_experts else 1
                    for ep in range(1, min(cp * dp, num_experts) + 1):

                        # Check if (ep | cp * dp) and (ep | expert_number)
                        if ((cp * dp) % ep != 0) or (num_experts % ep != 0):
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
                                cfg_zero0 = SearchConfig()
                                cfg_zero0.copy_from_config(mcfg)
                                cfg_zero0.tensor_model_parallel_size = tp
                                cfg_zero0.context_parallel_size = cp
                                cfg_zero0.pipeline_model_parallel_size = pp
                                cfg_zero0.num_layers_per_virtual_pipeline_stage = \
                                    layers_per_vpp
                                cfg_zero0.use_distributed_optimizer = False
                                cfg_zero0.micro_batch_size = mbs
                                if mcfg.is_moe():
                                    cfg_zero0.expert_model_parallel_size = ep
                                cfg_zero0.normalize()

                                valid_configs.append(cfg_zero0)

                                # When (dp * cp > 1), zero can be 1; add this config to the list
                                if dp * cp > 1:
                                    cfg_zero1 = replace(cfg_zero0,
                                                        use_distributed_optimizer=True)
                                    valid_configs.append(cfg_zero1)

    return valid_configs
