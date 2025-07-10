# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.communication.linear_models import HCCSDomainModel
from mindspeed.auto_tuning.module.communication.comm_perf_predictor import CommPerfPredictor, SimpleParallelCfg
from mindspeed.auto_tuning.module.communication.communication_profile import TpProfileTimeInfo
from mindspeed.auto_tuning.utils.utils import NumberConstant


class DebugTpComm:
    def __init__(self):
        self.comm_time_y = 0
        self.total_time = 0
        self.wait_time = 0
        self.overlap_time_y = 0
        self.comm_x = 0
        self.cfg = SearchConfig()
        self.cfg_no = 0


class TpCommPerfPredictor(CommPerfPredictor):
    def __init__(self, hard_info):
        super(TpCommPerfPredictor, self).__init__(hard_info)
        self.tp_total_model = HCCSDomainModel()
        self.tp_overlap_model = HCCSDomainModel()

    def get_communication_info_from_profile(self, tp_profile_time_info, hcom_info_tag_id):
        tp_profile_time_info.total_comm_time += hcom_info_tag_id.total_time_ms
        tp_profile_time_info.wait_comm_time += hcom_info_tag_id.wait_time_ms
        tp_profile_time_info.overlap_comm_time += hcom_info_tag_id.overlap_time_ms

    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, tp_profile_time_info: TpProfileTimeInfo
    ):
        config = model_config
        tp = config.tp
        cp = config.cp
        pp = config.pp
        s = config.seq_length / NumberConstant.CONVERSION_TIME
        total_time = tp_profile_time_info.total_comm_time
        wait_time = tp_profile_time_info.wait_comm_time
        overlap_time = tp_profile_time_info.overlap_comm_time
        traffic = s * (tp - 1) / (tp * cp) * pp
        bandwidth_910b = (tp - 1)
        min_domain = tp
        bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
        comm_x = traffic / bandwidth
        cfg = SimpleParallelCfg(config_no, tp, cp, '', '', pp, '')
        comm_time_y = total_time - wait_time
        overlap_time_y = overlap_time

        self.tp_total_model.add_sample(*(comm_x, comm_time_y, cfg))
        self.tp_overlap_model.add_sample(*(comm_x, overlap_time_y, cfg))

        debug_info = DebugTpComm()
        debug_info.comm_x = comm_x
        debug_info.comm_time_y = comm_time_y
        debug_info.total_time = total_time
        debug_info.wait_time = wait_time
        debug_info.overlap_time_y = overlap_time_y
        debug_info.cfg = config
        debug_info.cfg_no = config_no
        self.debug_info_list.append(debug_info)

    def fit(self):
        self.tp_total_model.fit()
        self.tp_overlap_model.fit()

    def debug(self):
        self.logger.debug(f"******************profile info list***********************")
        tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<8}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}\t{10:<1}\t{11:<1}"
        self.logger.debug(f"******************   tp(ms)   ***********************")
        self.logger.debug(tplt.format('x', 'tp_time', 'overlap', 'total_time', 'wait_time',
                                      'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
        for debug_info in self.debug_info_list:
            if debug_info.cfg.use_ascend_mc2:
                continue
            self.logger.debug(tplt.format(
                              round(debug_info.comm_x, 2),
                              round(debug_info.comm_time_y, 3),
                              round(debug_info.overlap_time_y, 2),
                              round(debug_info.total_time, 2),
                              round(debug_info.wait_time, 2),
                              debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                              debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                              chr(12288)))
        self.logger.debug(f"-----------")
        tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}"
        self.logger.debug(tplt.format('tp_w', 'tp_b', 'overlap_w', 'overlap_b', chr(12288)))
        self.logger.debug(tplt.format(round(self.tp_total_model.w, 3), round(self.tp_total_model.b, 3),
                          round(self.tp_overlap_model.w, 3), round(self.tp_overlap_model.b, 3),
                          chr(12288)))
        self.logger.debug(f"\n\n\n")
        return

    def predict(self, search_cfg: SearchConfig):
        tp = search_cfg.tensor_model_parallel_size
        cp = search_cfg.context_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        s = search_cfg.seq_length / NumberConstant.CONVERSION_TIME
        tp_time = 0
        if tp > 1:
            traffic = s * (tp - 1) / (tp * cp) * pp
            bandwidth_910b = (tp - 1)
            min_domain = tp
            bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
            tp_x = traffic / bandwidth
            tp_time = self.tp_total_model.predict(*(tp_x,))
            tp_overlap_time = self.tp_overlap_model.predict(*(tp_x,))
            tp_time = tp_time - tp_overlap_time
        if tp_time < 0:
            tp_time = 0
        return tp_time
