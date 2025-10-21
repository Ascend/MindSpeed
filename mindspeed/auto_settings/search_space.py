"""
创建搜索空间
"""
import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import replace

from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.config.search_config import ExecutorFlag, DISABLE_CP
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.utils.singleton import Singleton
from mindspeed.auto_settings.utils.utils import get_prof_dir
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.file_utils import check_file_size
from mindspeed.auto_settings.utils.utils import (
    get_tp_for_profiling, get_seq_length_for_profiling, get_num_experts_for_profiling
)


class SearchSpace(metaclass=Singleton):
    PP4_FILENAME = "auto_settings_static_model_pp4.json"
    EXPERT2_FILENAME = "auto_settings_static_model_expert2.json"
    TP2_FILENAME = "auto_settings_static_model_tp2.json"
    GEN_CFGS_FILENAME = "profile/profiling_configs.json"
    BASELINE_SEQLEN = 4096

    def __init__(self):
        self.logger = get_logger("search_spaces")

    def generate_dynamic_mem_profiling_list(self) -> List[Tuple[SearchConfig, str]]:
        result: List[Tuple[SearchConfig, str]] = list()

        configs = []
        model_config = get_model_config()
        baseline_cfg = SearchConfig()
        baseline_cfg.copy_from_config(model_config)
        baseline_cfg.tensor_model_parallel_size = 4
        baseline_cfg.context_parallel_size = 1
        baseline_cfg.pipeline_model_parallel_size = 1
        baseline_cfg.num_layers = 1
        baseline_cfg.seq_length = self.BASELINE_SEQLEN
        if model_config.is_moe():
            baseline_cfg.num_experts = 16
            baseline_cfg.expert_model_parallel_size = 1
        configs.append(baseline_cfg)

        tp8_cfg = replace(baseline_cfg, tensor_model_parallel_size=8)
        configs.append(tp8_cfg)
        seq8k_cfg = replace(baseline_cfg, seq_length=2 * self.BASELINE_SEQLEN)
        configs.append(seq8k_cfg)

        for cfg in configs:
            cfg.prepare_for_profiling()
            result.append((cfg, get_prof_dir(cfg=cfg)))

        return result

    def generate_static_mem_profiling_list(self) -> List[Tuple[SearchConfig, str]]:
        result: List[Tuple[SearchConfig, str]] = list()

        model_config = get_model_config()
        baseline_tp = get_tp_for_profiling()
        pp4_cfg = SearchConfig()
        pp4_cfg.copy_from_config(model_config)
        pp4_cfg.tensor_model_parallel_size = baseline_tp
        pp4_cfg.context_parallel_size = 1
        pp4_cfg.pipeline_model_parallel_size = 4
        pp4_cfg.num_layers = 4
        pp4_cfg.untie_embeddings_and_output_weights = True
        pp4_cfg.seq_length = 4096
        if model_config.is_moe():
            pp4_cfg.num_experts = 1
            pp4_cfg.expert_model_parallel_size = 1
            pp4_cfg.moe_tp_extend_ep = False
        pp4_cfg.profile_type = ExecutorFlag.PARSE_MODEL
        result.append((pp4_cfg, self.PP4_FILENAME))

        if model_config.is_moe():
            expert2_cfg = replace(
                pp4_cfg,
                pipeline_model_parallel_size=1,
                num_layers=1,
                num_experts=2,
                expert_model_parallel_size=1
            )
            result.append((expert2_cfg, self.EXPERT2_FILENAME))

        tp2_cfg = replace(
            pp4_cfg,
            tensor_model_parallel_size=baseline_tp * 2,
            pipeline_model_parallel_size=1,
            num_layers=1,
            untie_embeddings_and_output_weights=False
        )
        result.append((tp2_cfg, self.TP2_FILENAME))

        for cfg, _ in result:
            cfg.prepare_for_profiling()

        return result

    def generate_profiling_configs(self) -> List[Tuple[SearchConfig, str]]:
        profile_cfgs: List[Tuple[SearchConfig, str]] = list()

        model_config = get_model_config()
        base_tp = get_tp_for_profiling()
        base_seq_len = get_seq_length_for_profiling(model_config)
        base_num_experts = get_num_experts_for_profiling(model_config)

        with Path(__file__).parent.joinpath(self.GEN_CFGS_FILENAME).open(encoding="utf-8") as f:
            check_file_size(f)
            cfgs = json.load(f)

        for cfg in cfgs:
            if "skip" in cfg.get("name", ""):
                self.logger.debug(f"{cfg} asked to skip.")
                continue

            if DISABLE_CP and cfg.get("cp", 1) > 1:
                self.logger.debug(f"Not searching cp, dropped {cfg}.")
                continue

            gen_cfg = SearchConfig()
            gen_cfg.copy_from_config(model_config)

            tp = cfg.get("tp", "default")
            if tp == "default":
                gen_cfg.tensor_model_parallel_size = base_tp
            elif tp.startswith("mul_t_by="):
                gen_cfg.tensor_model_parallel_size = base_tp * int(tp.strip().split("=")[1])
            else:
                raise ValueError(f"Not supporting value on tp field: {tp} of {cfg}.")

            gen_cfg.context_parallel_size = cfg.get("cp", 1)
            gen_cfg.pipeline_model_parallel_size = cfg.get("pp", 1)
            gen_cfg.num_layers = cfg.get("pp", 1)
            gen_cfg.use_ascend_mc2 = cfg.get("mc2", False)

            seq = cfg.get("seq", "default")
            if seq == "default":
                gen_cfg.seq_length = base_seq_len
            elif seq.startswith("slice_seq_by="):
                slice_rate = int(seq.strip().split("=")[1])
                if base_seq_len // slice_rate > 2 * 1024:
                    gen_cfg.seq_length = base_seq_len // slice_rate
                else:
                    gen_cfg.seq_length = base_seq_len * slice_rate
            else:
                raise ValueError(f"Not supporting value on seq field: {seq} of {cfg}.")

            if model_config.is_moe():
                num_experts = cfg.get("experts", "default")
                if num_experts == "default":
                    gen_cfg.num_experts = base_num_experts
                else:
                    raise ValueError(f"Not supporting value on experts field: {num_experts} of {cfg}.")
                gen_cfg.expert_model_parallel_size = cfg.get("ep", 1)

            if gen_cfg.seq_length // gen_cfg.cp <= 2 * 1024:
                self.logger.debug(f"Seq per cp too small, dropped {cfg}.")
                continue

            gen_cfg.prepare_for_profiling()
            if gen_cfg.ep and gen_cfg.ep > gen_cfg.dp * gen_cfg.cp:
                raise ValueError(f"ep > dp * cp of {cfg}.")

            profile_cfgs.append((gen_cfg, get_prof_dir(cfg=gen_cfg)))

        return profile_cfgs

    def build_pre_search_spaces(self):
        """
        创建预置的搜索空间
        """
        static_list = self.generate_static_mem_profiling_list()
        dynamic_list = self.generate_dynamic_mem_profiling_list()
        common_list = self.generate_profiling_configs()
        result = static_list + dynamic_list + common_list
        self.logger.info("profile_cfgs (tp, pp, dp, cp, ep, #layers, seq_len):")
        self.logger.info(",".join(
            str((cfg.tp,
                 cfg.pp,
                 cfg.dp,
                 cfg.cp,
                 cfg.ep,
                 cfg.num_layers,
                 cfg.seq_length))
            for cfg, _ in result))
        return result

    def build_search_spaces(self) -> List[SearchConfig]:

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
        mcfg = get_model_config()
        system_config = get_system_config()
        num_devices = system_config.search_world_size
        device_type = system_config.device_type

        valid_configs: List[SearchConfig] = list()

        # Iterate over all possible combinations of tp, cp, pp, dp, ep and zero
        # Prune tp based on device_type, tp = 1 or 8 only if running on 910B
        for tp in [2 ** i for i in range(system_config.devices_per_node.bit_length())]:

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
                                    # post init vps pps
                                    cfg.post_init()

                                    valid_configs.append(cfg)

        return valid_configs
