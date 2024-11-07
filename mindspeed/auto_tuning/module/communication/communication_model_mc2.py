from mindspeed.auto_tuning.module.communication.communication_model import CommunicationModel
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import NumberConstant


class Mc2Model(CommunicationModel):
    def __init__(self, hccs_dev_num):
        super(Mc2Model, self).__init__(hccs_dev_num)

    def get_communication_info_from_profile(self, mc2_profile_time_info, hcom_info_tage_id, index):
        mc2_res = hcom_info_tage_id[index][1]
        mat_res = hcom_info_tage_id[index - 1][1]
        mc2_profile_time_info.matmul_compute_time = mat_res.matmul_total_time[0]
        mc2_profile_time_info.total_comm_time = mc2_res.mc2_total_time[0]

    def get_comm_info_list(self, mc2_profile_time_info, config):
        tp = config.tp
        cp = config.cp
        s = config.seq_length / NumberConstant.CONVERSION_TIME
        hccs_x = (s / (tp * cp))
        hccs_time = mc2_profile_time_info.total_comm_time - mc2_profile_time_info.matmul_compute_time
        self.comm.append_hccs([hccs_x], hccs_time)

    def modeling(self):
        sum_x = 0
        sum_time = 0
        for index, x in enumerate(self.comm.hccs_x_list):
            sum_x += x[0]
            sum_time += self.comm.hccs_time_list[index][0]
        self.comm.hccs_w = sum_time / sum_x

    def print_modeling(self, config_list):
        mc2lt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}"
        self.logger.debug(f"******************   mc2(ms)   ***********************")
        self.logger.debug(mc2lt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'mc2_time', 'mc2_x', chr(12288)))
        index = 0
        for cfg in config_list:
            if cfg.use_ascend_mc2:
                self.logger.debug(mc2lt.format(index, cfg.tp, cfg.dp, cfg.pp,
                                   cfg.cp,
                                   cfg.ep,
                                   round(self.comm.hccs_time_list[index][0], 2), round(
                        self.comm.hccs_x_list[index][0], 3), chr(12288)))
                index += 1
        self.logger.debug(f"-----------")
        mc2lt = "{0:<9}\t{1:<9}"
        self.logger.debug(mc2lt.format('tp_w', 'tp_b', chr(12288)))
        self.logger.debug(mc2lt.format(round(self.comm.hccs_w, 3), round(self.comm.hccs_b, 3), chr(12288)))
        self.logger.debug(f"\n\n\n")

    def performance(self, search_cfg):
        tp = search_cfg.tensor_model_parallel_size
        cp = search_cfg.context_parallel_size
        s = search_cfg.seq_length / 1000
        mc2_time = 0
        if tp > 1:
            mc2_time = self.comm.hccs_w * (s / (tp * cp)) + self.comm.hccs_b
        return mc2_time
