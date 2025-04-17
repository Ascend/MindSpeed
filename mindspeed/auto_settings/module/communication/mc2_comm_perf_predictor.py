from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.module.communication.comm_perf_predictor import CommPerfPredictor, SimpleParallelCfg
from mindspeed.auto_settings.module.communication.communication_profile import Mc2ProfileTimeInfo


class DebugMc2Comm:
    def __init__(self):
        self.comm_time_y = 0
        self.comm_x = 0
        self.cfg = SearchConfig()
        self.cfg_no = 0


class Mc2CommPerfPredictor(CommPerfPredictor):
    def __init__(self, hard_info):
        super(Mc2CommPerfPredictor, self).__init__(hard_info)
        self.xs = []
        self.ys = []
        self.cfgs = []
        self.w = 0
        self.b = 0

    def get_communication_info_from_profile(
        self, mc2_profile_time_info: Mc2ProfileTimeInfo, hcom_info_tage_id, index
    ):
        mc2_res = hcom_info_tage_id[index][1]
        mat_res = hcom_info_tage_id[index - 1][1]
        mc2_profile_time_info.matmul_compute_time = mat_res.matmul_total_time[0]
        mc2_profile_time_info.total_comm_time = mc2_res.mc2_total_time[0]

    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, mc2_profile_time_info: Mc2ProfileTimeInfo
    ):
        config = model_config
        tp = config.tp
        cp = config.cp
        pp = config.pp
        s = config.seq_length / 1000
        hccs_x = s / (tp * cp) * pp
        hccs_time = (
            mc2_profile_time_info.total_comm_time - mc2_profile_time_info.matmul_compute_time
        )
        cfg = SimpleParallelCfg(config_no, tp, cp, '', pp, '', '')
        self.xs.append(hccs_x)
        self.ys.append(hccs_time)
        self.cfgs.append(cfg)

        debug_info = DebugMc2Comm()
        debug_info.comm_x = hccs_x
        debug_info.comm_time_y = hccs_time
        debug_info.cfg = config
        debug_info.cfg_no = config_no
        self.debug_info_list.append(debug_info)

    def fit(self):
        sum_x = 0
        sum_time = 0
        for index, x in enumerate(self.xs):
            sum_x += x
            sum_time += self.ys[index]

        self.w = sum_time / sum_x

    def debug(self, config_list):
        print("===============================================================================")
        mc2lt = "{0:<9}\t{1:<9}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}"
        self.logger.debug(f"******************   MC2 modeling   ***********************")
        self.logger.debug(mc2lt.format("MC2_x", "MC2_time", "No", "tp", "dp", "pp", "cp", "ep", chr(12288)))

        for debug_info in self.debug_info_list:
            if debug_info.cfg.use_ascend_mc2:
                self.logger.debug(
                    mc2lt.format(
                        round(debug_info.comm_x, 2), round(debug_info.comm_time_y, 2),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp, debug_info.cfg.pp,
                        debug_info.cfg.cp, debug_info.cfg.ep, chr(12288),
                    )
                )
        self.logger.debug(f"------model parameters of MC2-------------------")
        mc2lt = "{0:<9}\t{1:<9}"
        self.logger.debug(mc2lt.format("mc2_w", "mc2_b", chr(12288)))
        self.logger.debug(mc2lt.format(round(self.w, 3), round(self.b, 3), chr(12288)))
        print("===============================================================================")

    def predict(self, search_cfg: SearchConfig):
        tp = search_cfg.tensor_model_parallel_size
        cp = search_cfg.context_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        s = search_cfg.seq_length / 1000
        mc2_time = 0
        if tp > 1:
            mc2_time = self.w * (s / (tp * cp) * pp) + self.b
        return mc2_time
