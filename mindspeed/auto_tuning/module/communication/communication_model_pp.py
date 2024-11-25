from mindspeed.auto_tuning.module.communication.communication_model import CommunicationModel


class PpModel(CommunicationModel):
    def __init__(self, hccs_dev_num):
        super(PpModel, self).__init__(hccs_dev_num)

    def get_communication_info_from_profile(self, pp_profile_time_info, hcom_info_tage_id, pp):
        last_pp_start_time = 0
        total_pp_time = 0
        for i in range(0, pp - 1):
            key = list(hcom_info_tage_id.details[i].keys())[0]
            total_pp_time += hcom_info_tage_id.details[i][key]['Elapse Time(ms)']
            if last_pp_start_time == 0:
                last_pp_start_time = hcom_info_tage_id.details[i][key]['Start Timestamp(us)']
        pp_profile_time_info.each_pp_time = total_pp_time / (pp - 1)

    def get_comm_info_list(self, pp_profile_time_info, config):
        tp = config.tp
        cp = config.cp
        pp = config.pp
        dp = config.dp
        layers_per_vpp = config.layers_per_vpp if config.layers_per_vpp else 1
        comm_x = 1 / (layers_per_vpp * tp * cp)
        iv_list = [comm_x, 0, 0] # PP does not need to consider cross modeling.
        comm_time = pp_profile_time_info.each_pp_time
        self.main_domain.max_domain = pp * dp * cp * tp
        self.main_domain.min_domain = pp * dp * cp * tp
        if pp > 1:
            self.main_domain.append_time_in_domain(self.comm, iv_list, comm_time)
            # PPtime indicates the time consumed by each PP communication.

    def modeling(self):
        self.comm.modeling()
        if self.comm.hccs_w == 0:
            self.comm.hccs_w = self.comm.roce_w

    def print_modeling(self, config_list):
        self.logger.debug(f"******************   pp(ms)   ***********************")
        if self.main_domain.roce_comm_exist:
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<8}\t{8:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'vp', 
                  'cp', 'ep', 'pp_x', 'pp_time', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].pp > 1:
                    if self.comm.roce_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          str(config_list[i].layers_per_vpp), config_list[i].cp, config_list[i].ep,
                                          round(self.comm.roce_x_list[index][0], 3), round(
                                              self.comm.roce_time_list[index][0], 2),
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('pp_w', 'pp_b', chr(12288)))
            self.logger.debug(tplt.format(round(self.comm.roce_w, 3), 
                  round(self.comm.roce_b, 3), chr(12288)))
            self.logger.debug(f"-----------")
        if self.main_domain.hccs_comm_exist:
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<8}\t{8:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'vp', 'cp', 
                  'ep', 'pp_HCCS_x', 'pp_HCCS_time', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].pp > 1:
                    if self.comm.hccs_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          str(config_list[i].layers_per_vpp), config_list[i].cp, config_list[i].ep,
                                          round(
                                              self.comm.hccs_x_list[index][0], 3),
                                          round(self.comm.hccs_time_list[index][0], 2), chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('pp_HCCS_w', 'pp_HCCS_b', chr(12288)))
            self.logger.debug(tplt.format(round(self.comm.hccs_w, 3), round(self.comm.hccs_b, 3), 
                              chr(12288)))
            self.logger.debug(f"-----------")
        self.logger.debug(f"\n\n\n")

    def performance(self, search_cfg):
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        vp = search_cfg.num_layers // (
            pp * search_cfg.num_layers_per_virtual_pipeline_stage) if search_cfg.num_layers_per_virtual_pipeline_stage else 1
        cp = search_cfg.context_parallel_size

        pp_time = 0.0
        comm_x = (1 / (vp * tp * cp))
        iv_list = [comm_x, 0, 0] # PP does not need to consider cross modeling.
        self.main_domain.max_domain = pp * dp * cp * tp
        self.main_domain.min_domain = pp * dp * cp * tp
        if pp > 1:
            each_pp_time = self.main_domain.cal_time_in_domain(self.comm, iv_list)
            each_pp_time = each_pp_time * 2  # Multiply send and receive by 2.
            pp_time = each_pp_time * (pp * vp - 1) * 2
        return pp_time
