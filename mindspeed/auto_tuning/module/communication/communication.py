from mindspeed.auto_tuning.module.communication import communication_profile
from mindspeed.auto_tuning.module.communication.communication_model_tp import TpModel
from mindspeed.auto_tuning.module.communication.communication_model_cp import CpModel
from mindspeed.auto_tuning.module.communication.communication_model_dp import DpModel
from mindspeed.auto_tuning.module.communication.communication_model_pp import PpModel
from mindspeed.auto_tuning.module.communication.communication_model_ep import EpModel
from mindspeed.auto_tuning.module.communication.communication_model_mc2 import Mc2Model


class Communication(object):
    """Communication modeling."""

    def __init__(self, hardware=None, model_cfg=None):
        self.hardware = hardware
        self.model_cfg = model_cfg

        self.hccs_dev_num_910_9 = 384
        self.hccs_dev_num_910b = 8
        self.hccs_dev_num = 0
        if "910_9" in self.hardware.device_type:
            self.hccs_dev_num = self.hccs_dev_num_910_9
        if "910B" in self.hardware.device_type:
            self.hccs_dev_num = self.hccs_dev_num_910b

        self.tp_model = TpModel(self.hccs_dev_num)
        self.cp_model = CpModel(self.hccs_dev_num)
        self.dp_model = DpModel(self.hccs_dev_num)
        self.pp_model = PpModel(self.hccs_dev_num)
        self.ep_model = EpModel(self.hccs_dev_num)
        self.mc2_model = Mc2Model(self.hccs_dev_num)

        self.config_list = []

    def communication_modeling(self, profiling_results):
        self.adapt_to_profile_info(profiling_results)
        self.info_to_modeling()

    def adapt_to_profile_info(self, profiling_results):
        for index, (config, model) in enumerate(profiling_results):
            # Reads profile information in a group of configuration files.
            total_profile_time_info = communication_profile.TotalProfileTimeInfo()

            self.config_list.append(config)

            self.get_profile_info(model, total_profile_time_info, config, profiling_results, index)
            # Now force to run only one floor

            if config.use_ascend_mc2:
                self.mc2_model.get_comm_info_list(
                    total_profile_time_info.mc2_profile_time_info, config)
            else:
                self.tp_model.get_comm_info_list(
                    total_profile_time_info.tp_profile_time_info, config)
            self.dp_model.get_comm_info_list(
                total_profile_time_info.dp_profile_time_info, config)
            self.cp_model.get_comm_info_list(
                total_profile_time_info.cp_profile_time_info, config)
            self.ep_model.get_comm_info_list(
                total_profile_time_info.ep_profile_time_info, config)
            self.pp_model.get_comm_info_list(
                total_profile_time_info.pp_profile_time_info, config)

    def info_to_modeling(self):
        self.tp_model.modeling()
        self.tp_model.print_modeling(self.config_list)
        self.mc2_model.modeling()
        self.mc2_model.print_modeling(self.config_list)
        self.dp_model.modeling()
        self.dp_model.print_modeling(self.config_list)
        self.cp_model.modeling()
        self.cp_model.print_modeling(self.config_list)
        self.ep_model.modeling()
        self.ep_model.print_modeling(self.config_list)
        self.pp_model.modeling()
        self.pp_model.print_modeling(self.config_list)

    def get_profile_info(self, model, total_profile_time_info, config, profiling_results, index):
        tensor_hcom_info = model.tensor_parallel_comm
        data_hcom_info = model.data_parallel_comm
        pipeline_hcom_info = model.pipeline_parallel_comm
        context_hcom_info = model.context_parallel_comm
        expert_hcom_info = model.expert_parallel_comm
        if config.use_ascend_mc2:
            self.mc2_model.get_communication_info_from_profile(total_profile_time_info.mc2_profile_time_info,
                                                               profiling_results,
                                                               index)
        for stage_id, stage_id_tensor_hcom_info in enumerate(tensor_hcom_info):
            # ["tp_x"] regression
            if stage_id == 0 and len(tensor_hcom_info) > stage_id:
                self.tp_model.get_communication_info_from_profile(
                    total_profile_time_info.tp_profile_time_info, tensor_hcom_info[stage_id])
            # para_list.cp_x regression
            if stage_id == 0 and len(context_hcom_info) > stage_id:
                self.cp_model.get_communication_info_from_profile(
                    total_profile_time_info.cp_profile_time_info, context_hcom_info[stage_id], model, config.cp)
            if config.pp > 1:
                if stage_id == 0 and len(pipeline_hcom_info) > stage_id:
                    self.pp_model.get_communication_info_from_profile(
                        total_profile_time_info.pp_profile_time_info, pipeline_hcom_info[stage_id], config.pp)
            # para_list.dp_x regression
            if stage_id == len(tensor_hcom_info) - 1 and len(data_hcom_info) > stage_id:
                self.dp_model.get_communication_info_from_profile(
                    total_profile_time_info.dp_profile_time_info, data_hcom_info[stage_id])
            # para_list.ep_x regression
            if stage_id == 0 and len(expert_hcom_info) > stage_id:
                self.ep_model.get_communication_info_from_profile(
                    total_profile_time_info.ep_profile_time_info, expert_hcom_info[stage_id])
