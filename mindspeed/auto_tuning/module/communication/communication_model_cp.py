from mindspeed.auto_tuning.module.communication.communication_model import CommunicationModel
_GLOBAL_ATTN_FORWARD_KERNEL_NAMES = [
    "aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore"
]
_GLOBAL_ATTN_BACKWARD_KERNEL_NAMES = [
    "aclnnFlashAttentionScoreGrad_FlashAttentionScoreGrad_FlashAttentionScoreGrad"
]


class CpModel(CommunicationModel):
    def __init__(self, hccs_dev_num):
        super(CpModel, self).__init__(hccs_dev_num)
        # profile建模数据信息表
        self.cp_vector_x = []
        self.cp_vector_time = []
        self.cp_attn_x = []
        self.cp_attn_time = []
        self.cp_attn_bw_x = []
        self.cp_attn_bw_time = []

        self.cp_attn_w = 0
        self.cp_attn_b = 0
        self.cp_attn_bw_w = 0
        self.cp_attn_bw_b = 0
        self.cp_vector_w = 0
        self.cp_vector_b = 0

    def get_communication_info_from_profile(self, cp_profile_time_info, hcom_info_tage_id, model, cp):
        cp_profile_time_info.total_comm_time += hcom_info_tage_id.total_time_ms
        cp_profile_time_info.wait_comm_time += hcom_info_tage_id.wait_time_ms
        cp_profile_time_info.attn_cp_time, cp_profile_time_info.attn_cpbw_time = \
            self.get_vectortime_from_profiling(model, cp)
        cp_profile_time_info.vector_cp_time += hcom_info_tage_id.vector_time_ms

    def get_comm_info_list(self, cp_profile_time_info, config):
        tp = config.tp
        cp = config.cp
        pp = config.pp
        dp = config.dp
        s = config.seq_length / 1000

        # CP的通信量为CP-1倍每台机器上的正向KV、反向KV dKV
        # CP的正向与反向attention都被在通信了,attention的计算量不随CP变化
        # CP的正向计算VECTOR隐藏于通信中,反向计算VECTOR暴露于通信间,这里只考虑正向
        if cp > 1:
            # 这里仅考虑通信隐藏的attention，前向CP-1个，反向CP个
            self.cp_attn_x.append([s / tp / cp * (cp - 1) / cp])
            self.cp_attn_time.append([cp_profile_time_info.attn_cp_time])
            self.cp_attn_bw_x.append([s / tp / cp])
            self.cp_attn_bw_time.append([cp_profile_time_info.attn_cpbw_time])
            self.cp_vector_time.append([cp_profile_time_info.vector_cp_time])
            if cp - 2 < 0:
                self.cp_vector_x.append([0])
            else:
                self.cp_vector_x.append([cp - 2])

            comm_x = (cp - 1) * s / (tp * cp) * pp
            comm_time = cp_profile_time_info.total_comm_time
            
            K = cp * tp / self.hccs_dev_num
            comm_y = (K) * s / (tp * cp) * pp
            comm_z = (K - 1) * s / (tp * cp) * pp
            iv_list = [comm_x, comm_y, comm_z]
            self.main_domain.max_domain = cp * tp
            self.main_domain.min_domain = tp
            self.main_domain.append_time_in_domain(self.comm, iv_list, comm_time)

    def modeling(self):
        # 通信量模型
        self.comm.modeling()

        # 用于隐藏的attention与vector算子建模
        self.cp_attn_w, self.cp_attn_b = self.comm.linear_x_y(
            self.cp_attn_x, self.cp_attn_time)
        self.cp_attn_bw_w, self.cp_attn_bw_b = self.comm.linear_x_y(
            self.cp_attn_bw_x, self.cp_attn_bw_time)
        self.cp_vector_w, self.cp_vector_b = self.comm.linear_x_y(
            self.cp_vector_x, self.cp_vector_time)

    def print_modeling(self, config_list):
        self.logger.debug(f"******************   cp(ms)   ***********************")
        if self.main_domain.roce_comm_exist:
            self.logger.debug(f"roce")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'cp_time', 'cp_x',
                              chr(12288)))
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8.2f}\t{7:<8}"
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].cp > 1:
                    if self.comm.roce_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          self.comm.roce_time_list[index][0], self.comm.roce_x_list[index][0],
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"--------------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('cp_w,', 'cp_b', chr(12288)))
            self.logger.debug(tplt.format(round(self.comm.roce_w, 3), round(self.comm.roce_b, 3),
                              chr(12288)))
            self.logger.debug(f"-------------")
        if self.main_domain.hccs_comm_exist:
            self.logger.debug(f"hccs")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'cp_time', 'cp_x',
                              chr(12288)))
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8.2f}\t{7:<8}"
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].cp > 1:
                    if self.comm.hccs_x_list[index][0]:
                        self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp,
                                          config_list[i].cp, config_list[i].ep,
                                          self.comm.hccs_time_list[index][0], self.comm.hccs_x_list[index][0],
                                          chr(12288)))
                    index += 1
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('cp_HCCS_w,', 'cp_HCCS_b', chr(12288)))
            self.logger.debug(tplt.format(round(self.comm.hccs_w, 3), round(self.comm.hccs_b, 3),
                              chr(12288)))
            self.logger.debug(f"-----------")
        
        if self.main_domain.cross_comm_exist:
            self.logger.debug(f"cross")
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}"
            self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp',
                  'ep', 'cp_time', 'cp_cross_x', 'cp_cross_y', chr(12288)))
            tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8.2f}\t{7:<8.2f}\t{8:<8.2f}"
            index = 0
            for i, _ in enumerate(config_list):
                if config_list[i].cp > 1:
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
        
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<8}\t{8:<8}\t{9:<8}\t{10:<8}\t{11:<8}"
        self.logger.debug(tplt.format('No', 'tp', 'dp', 'pp', 'cp', 'ep', 'attn_x', 
              'attention', 'attn_bw_x', 'attn_bw', 'vector_x', 'vector_time', chr(12288)))
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8.2f}\t{7:<8.2f}\t{8:<8.2f}\t{9:<8.2f}\t{10:<8.2f}\t{11:<8.2f}"
        index = 0
        for i, _ in enumerate(config_list):
            if config_list[i].cp > 1:
                self.logger.debug(tplt.format(i, config_list[i].tp, config_list[i].dp, config_list[i].pp, 
                                  config_list[i].cp, config_list[i].ep,
                                  self.cp_attn_x[index][0], self.cp_attn_time[index][0],
                                  self.cp_attn_bw_x[index][0], self.cp_attn_bw_time[index][0],
                                  self.cp_vector_x[index][0], self.cp_vector_time[index][0], chr(12288)))
                index += 1
        self.logger.debug(f"-----------")
        tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}\t{4:<9}\t{5:<9}"
        self.logger.debug(tplt.format('attn_w,', 'attn_b', 'attn_bw_w', 
              'attn_bw_b', 'vector_w', 'vector_b', chr(12288)))
        self.logger.debug(tplt.format(round(self.cp_attn_w, 3), round(self.cp_attn_b, 3),
                          round(self.cp_attn_bw_w, 3), round(self.cp_attn_bw_b, 3),
                          round(self.cp_vector_w, 3), round(
                              self.cp_vector_b, 3),
                          chr(12288)))
        self.logger.debug(f"\n\n\n")
        return


    def get_vectortime_from_profiling(self, model, cp):
        attn_list = []
        attn_re_list = []
        attn_gb_list = []
        profile_info = model
        attention = 0.0
        attn_bw = 0.0
        for item in profile_info.forward.operator_info[0]:
            if item.name in _GLOBAL_ATTN_FORWARD_KERNEL_NAMES and len(attn_list) < cp - 1:
                attn_list.append(item)
                attention += float(item.duration_us)
        for item in profile_info.backward.operator_info[0]:
            if item.name in _GLOBAL_ATTN_FORWARD_KERNEL_NAMES and len(attn_re_list) < cp - 1:
                attn_re_list.append(item)
                attention += float(item.duration_us)
            if item.name in _GLOBAL_ATTN_BACKWARD_KERNEL_NAMES and len(attn_gb_list) < cp:
                attn_gb_list.append(item)
                attn_bw += float(item.duration_us)
        # attention有一个是影藏的.attn_bw都需要计算上
        attention = attention / 1000
        attn_bw = attn_bw / 1000
        return attention, attn_bw

    def performance(self, search_cfg):
        tp = search_cfg.tensor_model_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        cp = search_cfg.context_parallel_size
        s = search_cfg.seq_length / 1000
        cp_time = 0.0
        comm_x = (cp - 1) * s / (tp * cp) * pp
        K = cp * tp / self.hccs_dev_num
        comm_y = (K) * s / (tp * cp) * pp
        comm_z = (K - 1) * s / (tp * cp) * pp
        iv_list = [comm_x, comm_y, comm_z]
        self.main_domain.max_domain = cp * tp
        self.main_domain.min_domain = tp
        if cp > 1:
            comm_time = self.main_domain.cal_time_in_domain(self.comm, iv_list)

            attn_time = self.cp_attn_w * (s / tp / cp * (cp - 1) / cp) + self.cp_attn_b
            # 最后一块不做影藏,重计算天然适配
            attn_bw_time = self.cp_attn_bw_w * (s / tp / cp) + self.cp_attn_bw_b
            # attention和attn_bw需要分开考虑影藏
            cp_time1 = comm_time / 2 - attn_time * pp
            if cp_time1 < 0:
                cp_time1 = 0
            cp_time2 = comm_time / 2 - attn_bw_time * pp
            if cp_time2 < 0:
                cp_time2 = 0
            cp_time = cp_time1 + cp_time2
            if cp > 2:
                cp_vector_time = self.cp_vector_w * (cp - 2) + self.cp_vector_b
                cp_time = cp_time - cp_vector_time
                self.logger.debug('cp_time:{}, attn_time:{}, attn_bw_time:{}, '
                                  'cp_vector_time:{}'.format(cp_time, attn_time, attn_bw_time, cp_vector_time))
        if cp_time < 0:
            cp_time = 0.0
            self.logger.debug(f'CP的通信时间都为等待时间, CP通信部分建模暂时失效')
        return cp_time
