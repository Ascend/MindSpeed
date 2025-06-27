from typing import List
from collections import namedtuple

from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.communication.comm_perf_linear_model_factory import (
    CommPerfLinearModelFactory,
)
from mindspeed.auto_tuning.module.communication.comm_perf_predictor import CommPerfPredictor, SimpleParallelCfg
from mindspeed.auto_tuning.module.communication.communication_profile import EpProfileTimeInfo
from mindspeed.auto_tuning.utils.utils import NumberConstant


class DebugEpComm:
    def __init__(self):
        self.comm_x = 0
        self.hccs_x = 0
        self.roce_x = 0
        self.total_time = 0
        self.cfg = SearchConfig()
        self.model_type = None
        self.cfg_no = 0


class EpCommPerfPredictor(CommPerfPredictor):
    def __init__(self, hard_info):
        super(EpCommPerfPredictor, self).__init__(hard_info)
        self.is_moe = False

    def get_communication_info_from_profile(self, ep_profile_time_info, hcom_info_tag_id):
        ep_profile_time_info.total_comm_time += hcom_info_tag_id.total_time_ms
        ep_profile_time_info.wait_comm_time += hcom_info_tag_id.wait_time_ms
        ep_profile_time_info.min_time += hcom_info_tag_id.min_comm_time_ms

    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, ep_profile_time_info: EpProfileTimeInfo
    ):
        ep = model_config.ep
        if model_config.moe_tp_extend_ep:
            ep = ep * model_config.tp
        if not ep or ep <= 1:
            return

        if model_config.num_experts:
            self.is_moe = True

        tp = model_config.tp
        cp = model_config.cp
        pp = model_config.pp
        s = model_config.seq_length / NumberConstant.CONVERSION_TIME

        traffic = s * (ep - 1) * pp / ep / tp / cp
        bandwidth_910b = (ep - 1)
        min_domain = tp
        bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
        comm_x = traffic / bandwidth
        K = ep * tp / self.max_hccs_rank_num
        hccs_x = s * K * pp / ep / tp / cp
        roce_x = s * (K - 1) / K * pp / ep / tp / cp
        comm_time = ep_profile_time_info.min_time
        cfg = SimpleParallelCfg(config_no, tp, cp, '', ep, pp, '')
        iv_list = [comm_x, hccs_x, roce_x, comm_time, cfg]

        max_domain = ep * tp
        if model_config.moe_tp_extend_ep:
            max_domain = ep
            min_domain = 0
        ep_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "ep",
            min_rank_num=min_domain,
            max_rank_num=max_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )
        ep_time_model.add_sample(*iv_list)

        debug_info = DebugEpComm()
        debug_info.comm_x = comm_x
        debug_info.hccs_x = hccs_x
        debug_info.roce_x = roce_x
        debug_info.total_time = comm_time

        debug_info.cfg = model_config
        debug_info.cfg_no = config_no
        debug_info.model_type = str(type(ep_time_model))
        self.debug_info_list.append(debug_info)

    def fit(self):
        if self.is_moe:
            for model in CommPerfLinearModelFactory.get_models_by_module_name("ep"):
                if model:
                    model.fit()
            # Prerequisites. After the WB of the HCCS is learned,
            # Only in this way can roce data be split through cross.
            # corss_time = hccs_time + roce_time
            # Obtain the wb of the ROCE.
            # model if insteance roce
            # Check whether ROCE is missing but cross exists.

    def debug(self, config_list: List[SearchConfig]):
        if not self.is_moe:
            return
        self.logger.debug(f"******************   EP modeling  ***********************")
        if "hccs" in CommPerfLinearModelFactory._instance_table["ep"].keys():
            self.logger.debug(f"HCCS")
            tplt = "{0:<8}\t{1:<8}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}"
            self.logger.debug(tplt.format('x', 'total_time',
                                          'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
            for debug_info in self.debug_info_list:
                if "HCCS" in debug_info.model_type:
                    self.logger.debug(tplt.format(
                        round(debug_info.comm_x, 2),
                        round(debug_info.total_time, 3),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                        debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                        chr(12288)))
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('w', 'b', chr(12288)))
            attn_ag_model = CommPerfLinearModelFactory._instance_table["ep"]["hccs"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table["ep"]["hccs"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")

        if "roce" in CommPerfLinearModelFactory._instance_table["ep"].keys():
            self.logger.debug(f"ROCE")
            tplt = "{0:<8}\t{1:<8}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}"
            self.logger.debug(tplt.format('x', 'total_time',
                                          'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
            for debug_info in self.debug_info_list:
                if "ROCE" in debug_info.model_type:
                    self.logger.debug(tplt.format(
                        round(debug_info.comm_x, 2),
                        round(debug_info.total_time, 3),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                        debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                        chr(12288)))
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}"
            self.logger.debug(tplt.format('w', 'b', chr(12288)))
            attn_ag_model = CommPerfLinearModelFactory._instance_table["ep"]["roce"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table["ep"]["roce"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")

        self.logger.debug(f"Cross")
        tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}"
        self.logger.debug(tplt.format('hccs_x', 'roce_x', 'total_time',
                                      'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
        for debug_info in self.debug_info_list:
            if "Cross" in debug_info.model_type:
                self.logger.debug(tplt.format(
                    round(debug_info.hccs_x, 2),
                    round(debug_info.roce_x, 2),
                    round(debug_info.total_time, 3),
                    debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                    debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                    chr(12288)))
        self.logger.debug(f"-----------")
        self.logger.debug(f"\n\n\n")

    def predict(self, search_cfg: SearchConfig):
        ep = search_cfg.ep
        if search_cfg.moe_tp_extend_ep:
            ep = ep * search_cfg.tp
        ep_time = 0.0
        if not ep or ep <= 1:
            return ep_time

        tp = search_cfg.tp
        pp = search_cfg.pp
        cp = search_cfg.cp
        s = search_cfg.seq_length / NumberConstant.CONVERSION_TIME
        traffic = s * (ep - 1) * pp / ep / tp / cp
        bandwidth = (ep - 1)
        min_domain = tp
        bandwidth_910b = (ep - 1)
        bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
        comm_x = traffic / bandwidth
        K = ep * tp / self.max_hccs_rank_num
        comm_y = s * K * pp / ep / tp / cp
        comm_z = s * (K - 1) / K * pp / ep / tp / cp
        iv_list = [comm_x, comm_y, comm_z]
        max_domain = ep * tp

        if search_cfg.moe_tp_extend_ep:
            max_domain = ep
            min_domain = 0

        ep_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "ep",
            min_rank_num=min_domain,
            max_rank_num=max_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )

        ep_time = ep_time_model.predict(*iv_list)
        return ep_time
