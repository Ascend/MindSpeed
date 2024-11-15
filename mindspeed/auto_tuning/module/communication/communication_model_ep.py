from mindspeed.auto_tuning.module.communication.communication_model import CommunicationModel


class EpModel(CommunicationModel):
    def __init__(self, hccs_dev_num):
        super(EpModel, self).__init__(hccs_dev_num)

    def get_communication_info_from_profile(self, ep_profile_time_info, hcom_info_tage_id):
        ep_profile_time_info.total_comm_time += hcom_info_tage_id.total_time_ms
        ep_profile_time_info.wait_comm_time += hcom_info_tage_id.wait_time_ms
        ep_profile_time_info.min_time += hcom_info_tage_id.min_comm_time_ms

    def get_comm_info_list(self, ep_profile_time_info, config):
        tp = config.tp
        cp = config.cp
        ep = config.ep
        pp = config.pp
        s = config.seq_length / 1000
        experts = config.num_experts if config.num_experts else 1

        if ep and ep > 1:
            comm_x = experts * s * (ep - 1) * pp / ep / tp / cp
            K = ep * tp / self.hccs_dev_num
            comm_y = experts * s * (K) * pp / ep / tp / cp
            comm_z = experts * s * (K - 1) / K * pp / ep / tp / cp
            iv_list = [comm_x, comm_y, comm_z]
            comm_time = ep_profile_time_info.min_time
            self.main_domain.max_domain = ep * tp
            self.main_domain.min_domain = tp
            self.main_domain.append_time_in_domain(self.comm, iv_list, comm_time)

    def modeling(self):
        self.comm.modeling()
    
    def print_modeling(self, config_list):
        self.logger.debug(f"******************   ep(ms)   ***********************")
        if self.main_domain.roce_comm_exist:
            self.logger.debug(f"roce")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 
                  'ep', 'ep_roce_time', 'ep_roce_x', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].ep > 1:
                    if self.comm.roce_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          round(self.comm.roce_time_list[index][0], 2), round(
                                              self.comm.roce_x_list[index][0], 3),
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"--------------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('ep_w', 'ep_b', chr(12288)))
            self.logger.debug(tplt.format(round(self.comm.roce_w, 3), 
                  round(self.comm.roce_b, 3), chr(12288)))
            self.logger.debug(f"--------------")
        if self.main_domain.hccs_comm_exist:
            self.logger.debug(f"hccs")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 
                  'ep', 'ep_hccs_time', 'ep_hccs_x', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].ep > 1:
                    if self.comm.hccs_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          round(
                                              self.comm.hccs_time_list[index][0], 2),
                                          round(self.comm.hccs_x_list[index][0], 3), chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('ep_HCCS_w', 'ep_HCCS_b', chr(12288)))
            self.logger.debug(tplt.format(round(self.comm.hccs_w, 3), round(self.comm.hccs_b, 3), 
                              chr(12288)))
            self.logger.debug(f"-----------")
        if self.main_domain.cross_comm_exist:
            self.logger.debug(f"cross")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 
                  'ep', 'ep_cross_time', 'ep_cross_x', 'ep_cross_y', chr(12288)))
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8.2f}\t{7:<8.2f}\t{8:<8.2f}"
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].ep > 1:
                    if self.comm.cross_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          self.comm.cross_time_list[index][0],
                                          self.comm.cross_x_list[index][0], self.comm.cross_y_list[index][0], 
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format(round(self.comm.hccs_w, 3), round(self.comm.roce_w, 3), 
                              chr(12288)))
            self.logger.debug(f"-----------")
        self.logger.debug(f"\n\n\n")

    def performance(self, search_cfg):
        tp = search_cfg.tensor_model_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size
        s = search_cfg.seq_length / 1000
        ep_time = 0.0
        experts = search_cfg.num_experts if search_cfg.num_experts else 1
        comm_x = experts * s * (ep - 1) * pp / ep / tp / cp
        K = ep * tp / self.hccs_dev_num
        comm_y = experts * s * (K) * pp / ep / tp / cp
        comm_z = experts * s * (K - 1) / K * pp / ep / tp / cp
        iv_list = [comm_x, comm_y, comm_z]
        self.main_domain.max_domain = ep * tp
        self.main_domain.min_domain = tp
        if ep and ep > 1:
            ep_time = self.main_domain.cal_time_in_domain(self.comm, iv_list)
        return ep_time
