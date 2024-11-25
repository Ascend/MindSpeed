from mindspeed.auto_tuning.module.communication.communication_model \
    import CommunicationModel, CommunicationList, Domain


class DpModel(CommunicationModel):
    def __init__(self, hccs_dev_num):
        super(DpModel, self).__init__(hccs_dev_num)
        # Profile modeling data table

        self.attention = CommunicationList()
        self.attention_reducescatter = CommunicationList()
        self.attention_allgather = CommunicationList()

        self.mlp_domain = Domain(hccs_dev_num)
        self.zero_comm = CommunicationList()
        self.zero = CommunicationList()
        self.zero_reducescatter = CommunicationList()
        self.zero_allgather = CommunicationList()

    def get_communication_info_from_profile(self, dp_profile_time_info, hcom_info_tage_id):
        dp_profile_time_info.total_comm_time += hcom_info_tage_id.total_time_ms
        dp_profile_time_info.total_mlpzero_time += hcom_info_tage_id.mlp_zero_time_ms
        dp_profile_time_info.total_otherzero_time += hcom_info_tage_id.total_time_ms - hcom_info_tage_id.mlp_zero_time_ms
        dp_profile_time_info.mlp_ag_time += hcom_info_tage_id.mlp_ag_time_ms
        dp_profile_time_info.mlp_rs_time += hcom_info_tage_id.mlp_rs_time_ms
        dp_profile_time_info.other_ag_time += hcom_info_tage_id.other_ag_time_ms
        dp_profile_time_info.other_rs_time += hcom_info_tage_id.other_rs_time_ms

    def get_comm_info_list(self, dp_profile_time_info, config):
        tp = config.tp
        cp = config.cp
        dp = config.dp
        ep = config.ep
        pp = config.pp
        zero = config.zero1
        experts = config.num_experts if config.num_experts else 1

        # attention
        if dp * cp > 1:
            comm_x = (dp * cp - 1) / (tp * pp)
            K = dp * cp * tp / self.hccs_dev_num
            comm_y = (K) / (tp * pp)
            comm_z = (K - 1) / (tp * pp)
            iv_list = [comm_x, comm_y, comm_z]
            comm_time = dp_profile_time_info.total_otherzero_time
            reducescatter_time = dp_profile_time_info.other_rs_time
            allgather_time = dp_profile_time_info.other_ag_time
            dp_total_time = dp_profile_time_info.total_comm_time
            self.main_domain.max_domain = dp * cp * tp
            self.main_domain.min_domain = cp * tp
            self.main_domain.append_time_in_domain(self.attention, iv_list, comm_time)
            self.main_domain.append_time_in_domain(self.attention_reducescatter, iv_list, reducescatter_time)
            self.main_domain.append_time_in_domain(self.attention_allgather, iv_list, allgather_time)
            self.main_domain.append_time_in_domain(self.comm, iv_list, dp_total_time)
        # MLP
            mlp_x = experts * (dp * cp / ep - 1) / tp / pp
            comm_time = dp_profile_time_info.total_mlpzero_time
            reducescatter_time = dp_profile_time_info.mlp_rs_time
            allgather_time = dp_profile_time_info.mlp_ag_time
            mlp_x = experts * (dp * cp / ep - 1) / tp / pp
            K = dp * cp * tp / ep / self.hccs_dev_num
            mlp_y = experts * (K) / (tp * pp)
            mlp_z = experts * (K - 1) / (tp * pp)
            iv_list = [mlp_x, mlp_y, mlp_z]
            self.mlp_domain.max_domain = dp * cp * tp
            self.mlp_domain.min_domain = cp * tp * ep
            self.mlp_domain.append_time_in_domain(self.zero, iv_list, comm_time)
            self.mlp_domain.append_time_in_domain(self.zero_reducescatter, iv_list, reducescatter_time)
            self.mlp_domain.append_time_in_domain(self.zero_allgather, iv_list, allgather_time)
            self.mlp_domain.append_time_in_domain(self.zero_comm, iv_list, dp_total_time)

    def modeling(self):
        self.attention.modeling()
        self.attention_reducescatter.modeling()
        self.attention_allgather.modeling()
        self.zero.modeling()
        self.zero_reducescatter.modeling()
        self.zero_allgather.modeling()
        
    def print_modeling(self, config_list):
        self.logger.debug(f"******************   dp(ms)   ***********************")
        attention = [
            self.comm,
            self.attention,
            self.attention_reducescatter,
            self.attention_allgather,
        ] 
        self.logger.debug(f"attention time :")
        self.print_modeling_unit(config_list, attention, self.main_domain)
        self.logger.debug(f"\n\n")

        mlp = [
            self.zero_comm,
            self.zero,
            self.zero_reducescatter,
            self.zero_allgather,
        ]
        self.logger.debug(f"mlp time :")
        self.print_modeling_unit(config_list, mlp, self.mlp_domain)
        self.logger.debug(f"\n\n\n")

    def print_modeling_unit(self, config_list, info_list, domain):
        if domain.roce_comm_exist:
            self.logger.debug(f"  roce")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}\t{9:<8}\t{10:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'dp_time',
            'x', 'time', 'ag_time', 'rs_time', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].dp * config_list[i].cp > 1:
                    if info_list[1].roce_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          round(info_list[0].roce_time_list[index][0], 2),
                                          round(info_list[1].roce_x_list[index][0], 3),
                                          round(info_list[1].roce_time_list[index][0], 2),
                                          round(info_list[2].roce_time_list[index][0], 3),
                                          round(info_list[3].roce_time_list[index][0], 2),
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}\t{4:<9}\t{5:<9}"
            self.logger.debug(tplt.format('time_w', 'time_b', 'rs_w', 'rs_b', 'ag_w', 'ag_b', chr(12288)))
            self.logger.debug(tplt.format(round(info_list[1].roce_w, 2), round(info_list[1].roce_b, 2),
                              round(info_list[2].roce_w, 2),
                              round(info_list[2].roce_b, 2),
                              round(info_list[3].roce_w, 2),
                              round(info_list[3].roce_b, 2), chr(12288)))
            self.logger.debug(f"----------------------")
        if domain.hccs_comm_exist:
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}\t{9:<8}\t{10:<8}"
            self.logger.debug(f"  hccs")
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'dp_time', 
            'x', 'time', 'ag_time', 'rs_time', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].dp * config_list[i].cp > 1:
                    if info_list[1].hccs_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          round(info_list[0].hccs_time_list[index][0], 2),
                                          round(info_list[1].hccs_x_list[index][0], 3),
                                          round(info_list[1].hccs_time_list[index][0], 2),
                                          round(info_list[2].hccs_time_list[index][0], 3),
                                          round(info_list[3].hccs_time_list[index][0], 2),
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}\t{4:<9}\t{5:<9}"
            self.logger.debug(tplt.format('dp_w', 'dp_b', 'rs_w', 'rs_b', 'ag_w', 'ag_b', chr(12288)))
            self.logger.debug(tplt.format(round(info_list[1].hccs_w, 2), round(self.attention.hccs_b, 2),
                              round(info_list[2].hccs_w, 2),
                              round(info_list[2].hccs_b, 2),
                              round(info_list[3].hccs_w, 2),
                              round(info_list[3].hccs_b, 2), chr(12288)))
            self.logger.debug(f"----------------------")
        if domain.cross_comm_exist:
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}\t{9:<8}\t{10:<8}\t{11:<8}"
            self.logger.debug(f"  cross")
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'dp_time', 'dp_x', 'dp_y', 'total_time', 'ag_time',
                'rs_time', chr(12288)))
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].dp * config_list[i].cp > 1:
                    if info_list[1].cross_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          round(info_list[0].cross_time_list[index][0], 2),
                                          round(info_list[1].cross_x_list[index][0], 3),
                                          round(info_list[1].cross_y_list[index][0], 3),
                                          round(info_list[1].cross_time_list[index][0], 2),
                                          round(info_list[2].cross_time_list[index][0], 3),
                                          round(info_list[3].cross_time_list[index][0], 3),
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"----------------------")

    def performance(self, search_cfg):
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size if search_cfg.expert_model_parallel_size else 1
        zero = search_cfg.use_distributed_optimizer
        experts = search_cfg.num_experts if search_cfg.num_experts else 1

        dp_time = 0.0
        comm_time = 0.0
        mlp_time = 0.0
        overlap_time = 0.0
        other_reducescatter = 0.0
        other_allgather = 0.0
        zero_reducescatter = 0.0
        zero_allgather = 0.0
        if dp * cp > 1:
            # attention：
            self.main_domain.max_domain = dp * cp * tp
            self.main_domain.min_domain = cp * tp
            comm_x = (dp * cp - 1) / tp / pp
            K = dp * cp * tp / self.hccs_dev_num
            comm_y = (K) / (tp * pp)
            comm_z = (K - 1) / (tp * pp)
            iv_list = [comm_x, comm_y, comm_z]
            comm_time = self.main_domain.cal_time_in_domain(self.attention, iv_list)
            other_reducescatter = self.main_domain.cal_time_in_domain(self.attention_reducescatter, iv_list)
            other_allgather = self.main_domain.cal_time_in_domain(self.attention_allgather, iv_list)

            # mlp
            self.mlp_domain.max_domain = dp * cp * tp
            self.mlp_domain.min_domain = cp * tp * ep
            mlp_x = experts * (dp * cp / ep - 1) / tp / pp
            K = dp * cp * tp / ep / self.hccs_dev_num
            mlp_y = experts * (K) / (tp * pp)
            mlp_z = experts * (K - 1) / (tp * pp)
            mlp_iv_list = [mlp_x, mlp_y, mlp_z]
            mlp_time = self.mlp_domain.cal_time_in_domain(self.zero, mlp_iv_list)
            zero_reducescatter = self.mlp_domain.cal_time_in_domain(self.zero_reducescatter, mlp_iv_list)
            zero_allgather = self.mlp_domain.cal_time_in_domain(self.zero_allgather, mlp_iv_list)
            if zero:
                if pp > 1:
                    overlap_time += (pp - 1) / pp * (other_reducescatter + zero_reducescatter)
                if pp > 2:
                    overlap_time += (pp - 2) / pp * (other_allgather + zero_allgather)
        dp_time = comm_time + mlp_time - overlap_time
        # dp_time here is the total gbs time effect
        return dp_time
