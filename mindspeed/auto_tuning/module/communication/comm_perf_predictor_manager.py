# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.communication import communication_profile
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import MindSpeedSettings as Settings
from mindspeed.auto_tuning.module.communication.cp_comm_perf_predictor import CpCommPerfPredictor
from mindspeed.auto_tuning.module.communication.dp_comm_perf_predictor import DpCommPerfPredictor
from mindspeed.auto_tuning.module.communication.ep_comm_perf_predictor import EpCommPerfPredictor
from mindspeed.auto_tuning.module.communication.mc2_comm_perf_predictor import Mc2CommPerfPredictor
from mindspeed.auto_tuning.module.communication.pp_comm_perf_predictor import PpCommPerfPredictor
from mindspeed.auto_tuning.module.communication.tp_comm_perf_predictor import TpCommPerfPredictor
from mindspeed.auto_tuning.module.communication.comm_hardware.comm_hardware_info import CommHardInfo, HccsDev


class CommPerfPredictorManager(object):
    """CommPerfPredictorManager modeling."""

    def __init__(self, hardware=None, model_cfg=None):
        self.hardware = hardware
        self.model_cfg = model_cfg

        hard_info = CommHardInfo(self.hardware.device_type)

        self.tp_predictor = TpCommPerfPredictor(hard_info)
        self.cp_predictor = CpCommPerfPredictor(hard_info)
        self.dp_predictor = DpCommPerfPredictor(hard_info)
        self.pp_predictor = PpCommPerfPredictor(hard_info)
        self.ep_predictor = EpCommPerfPredictor(hard_info)

        self.config_list = []

    def communication_modeling(self, profiling_results):
        if Settings().model_cfg.seq_length >= 8 * 1024:
            self.cp_predictor.is_cp_modeling = True
        if Settings().model_cfg.num_experts:
            self.ep_predictor.is_moe = True
        self.adapt_to_profile_info(profiling_results)
        self.info_to_modeling()

    def communication_roce_predict_fix(self):
        if "910_93" in self.hardware.device_type:
            if os.getenv("HCCL_INTER_HCCS_DISABLE", None):
                self.tp_predictor.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93.value
                self.cp_predictor.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93.value
                self.dp_predictor.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93.value
                self.pp_predictor.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93.value
                self.ep_predictor.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93.value

    def adapt_to_profile_info(self, profiling_results):
        for index, (config, model) in enumerate(profiling_results):
            # Load the profile information in a group of configuration files.
            total_profile_time_info = communication_profile.TotalProfileTimeInfo()

            self.config_list.append(config)

            self.get_profile_info(model, total_profile_time_info, config, profiling_results, index)
            # Force to run only one floor
            config.use_ascend_mc2 = 0
            if config.use_ascend_mc2:
                self.mc2_predictor.receive_samples_from_profiling(
                    index, config, total_profile_time_info.mc2_profile_time_info
                )
            else:
                self.tp_predictor.receive_samples_from_profiling(
                    index, config, total_profile_time_info.tp_profile_time_info
                )

            self.dp_predictor.receive_samples_from_profiling(
                index, config, total_profile_time_info.dp_profile_time_info
            )

            self.cp_predictor.receive_samples_from_profiling(
                index, config, total_profile_time_info.cp_profile_time_info
            )
            self.ep_predictor.receive_samples_from_profiling(
                index, config, total_profile_time_info.ep_profile_time_info
            )
            self.pp_predictor.receive_samples_from_profiling(
                index, config, total_profile_time_info.pp_profile_time_info
            )

    def info_to_modeling(self):
        self.tp_predictor.fit()
        self.tp_predictor.debug()

        self.dp_predictor.fit()
        self.dp_predictor.debug(self.config_list)

        self.cp_predictor.fit()
        self.cp_predictor.debug(self.config_list)

        self.ep_predictor.fit()
        self.ep_predictor.debug(self.config_list)

        self.pp_predictor.fit()
        self.pp_predictor.debug(self.config_list)

    def get_profile_info(
        self, model, total_profile_time_info, config: SearchConfig, profiling_results, index
    ):
        tensor_hcom_info = model.tensor_parallel_comm
        data_hcom_info = model.data_parallel_comm
        pipeline_hcom_info = model.pipeline_parallel_comm
        context_hcom_info = model.context_parallel_comm
        expert_hcom_info = model.expert_parallel_comm

        config.use_ascend_mc2 = 0
        if config.use_ascend_mc2:
            self.mc2_predictor.get_communication_info_from_profile(
                total_profile_time_info.mc2_profile_time_info, profiling_results, index
            )
        for stage_id, stage_id_tensor_hcom_info in enumerate(tensor_hcom_info):
            # ["tp_x"] regression
            if config.tp > 1:
                if stage_id == 0 and len(tensor_hcom_info) > stage_id:
                    self.tp_predictor.get_communication_info_from_profile(
                        total_profile_time_info.tp_profile_time_info, tensor_hcom_info[stage_id]
                    )
            # para_list.cp_x regression
            if stage_id == 0 and len(context_hcom_info) > stage_id:
                self.cp_predictor.get_communication_info_from_profile(
                    total_profile_time_info.cp_profile_time_info,
                    context_hcom_info[stage_id],
                    model,
                    config.cp,
                )
            if config.pp > 1:
                if stage_id == 0 and len(pipeline_hcom_info) > stage_id:
                    self.pp_predictor.get_communication_info_from_profile(
                        total_profile_time_info.pp_profile_time_info,
                        pipeline_hcom_info[stage_id],
                        config.pp,
                    )
            # para_list.dp_x regression
            if stage_id == 0 and len(data_hcom_info) > stage_id:
                self.dp_predictor.get_communication_info_from_profile(
                    total_profile_time_info.dp_profile_time_info, data_hcom_info[stage_id]
                )
            # para_list.ep_x regression
            if stage_id == 0 and len(expert_hcom_info) > stage_id:
                self.ep_predictor.get_communication_info_from_profile(
                    total_profile_time_info.ep_profile_time_info, expert_hcom_info[stage_id]
                )
