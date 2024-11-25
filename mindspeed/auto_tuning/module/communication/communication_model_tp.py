from mindspeed.auto_tuning.module.communication.communication_model import CommunicationModel


class TpModel(CommunicationModel):
    def __init__(self, hccs_dev_num):
        super(TpModel, self).__init__(hccs_dev_num)
        # Profile modeling data table
        self.tp_comm_total_time_list = []
        self.tp_comm_wait_time_list = []
        self.tp_comm_overlap_time_list = []

        self.tp_hccs_overlap_w = 0
        self.tp_hccs_overlap_b = 0

    def get_communication_info_from_profile(self, tp_profile_time_info, hcom_info_tage_id):
        tp_profile_time_info.total_comm_time += hcom_info_tage_id.total_time_ms
        tp_profile_time_info.wait_comm_time += hcom_info_tage_id.wait_time_ms
        tp_profile_time_info.overlap_comm_time += hcom_info_tage_id.overlap_time_ms

    def get_comm_info_list(self, tp_profile_time_info, config):
        tp = config.tp
        cp = config.cp
        pp = config.pp
        s = config.seq_length / 1000
        total_time = tp_profile_time_info.total_comm_time
        wait_time = tp_profile_time_info.wait_comm_time
        overlap_time = tp_profile_time_info.overlap_comm_time

        comm_x = (s / (tp * cp))
        if pp == 1:
            # The last forward allgather is not calculated. The first two reverse allgathers plus the last allgather
            # are not calculated.
            # When the PP function is disabled, there are 18 communications in the TP domain. Therefore, four loss
            # communications need to be excluded.
            comm_time = (total_time - wait_time) * 14 / 18 / pp
            self.tp_comm_overlap_time_list.append([overlap_time * 2 / 3 / pp])
        else:
            # When PP is enabled, there are 15 communications in the TP domain, and one loss communication needs to
            # be excluded.
            comm_time = (total_time - wait_time) * 14 / 15 / pp
            self.tp_comm_overlap_time_list.append([overlap_time / pp])
        self.comm.append_hccs([comm_x], comm_time)
        self.tp_comm_total_time_list.append([total_time])
        self.tp_comm_wait_time_list.append([wait_time])

    def modeling(self):
        self.comm.hccs_w, self.comm.hccs_b = self.comm.linear_x_y(
            self.comm.hccs_x_list, self.comm.hccs_time_list)
        self.tp_hccs_overlap_w, self.tp_hccs_overlap_b = self.comm.linear_x_y(
            self.comm.hccs_x_list, self.tp_comm_overlap_time_list)
        return

    def print_modeling(self, config_list):
        self.logger.debug(f"******************profile info list***********************")
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}\t{9:<8}\t{10:<8}\t{11:<8}"
        self.logger.debug(f"******************   tp(ms)   ***********************")
        self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'tp_time', 'tp_x', 'overlap_time', 'total_time',
                          'wait_time', chr(12288)))
        
        index = 0
        for i, _ in enumerate(config_list):
            if config_list[i].use_ascend_mc2:
                continue
            self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp, config_list[i].cp,
                                          config_list[i].ep,
                              round(self.comm.hccs_time_list[index][0], 2),
                              round(self.comm.hccs_x_list[index][0], 3),
                              round(self.tp_comm_overlap_time_list[index][0], 2),
                              round(self.tp_comm_total_time_list[index][0], 2), 
                              round(self.tp_comm_wait_time_list[index][0], 2),
                              chr(12288)))
            index += 1
        self.logger.debug(f"-----------")
        tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}"
        self.logger.debug(tplt.format('tp_w', 'tp_b', 'overlap_w', 'overlap_b', chr(12288)))
        self.logger.debug(tplt.format(round(self.comm.hccs_w, 3), round(self.comm.hccs_b, 3),
                          round(self.tp_hccs_overlap_w, 3), 
                          round(self.tp_hccs_overlap_b, 3),
                          chr(12288)))
        self.logger.debug(f"\n\n\n")
        return

    def performance(self, search_cfg):
        tp = search_cfg.tensor_model_parallel_size
        cp = search_cfg.context_parallel_size
        s = search_cfg.seq_length / 1000
        tp_overlap_time = 0
        tp_time = 0
        if tp > 1:
            tp_time = self.comm.hccs_w * (s / (tp * cp)) + self.comm.hccs_b
            tp_overlap_time = self.tp_hccs_overlap_w * \
                s / (tp * cp) + self.tp_hccs_overlap_b
            tp_time = tp_time - tp_overlap_time
        if tp_time < 0:
            tp_time = 0
        return tp_time
