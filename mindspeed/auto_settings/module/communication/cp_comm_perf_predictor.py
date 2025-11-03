from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.module.communication.comm_perf_linear_model_factory import (
    CommPerfLinearModelFactory,
)
from mindspeed.auto_settings.module.communication.comm_perf_predictor import CommPerfPredictor, SimpleParallelCfg
from mindspeed.auto_settings.module.communication.communication_profile import CpProfileTimeInfo

_GLOBAL_ATTN_FORWARD_KERNEL_NAMES = [
    "aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore"
]
_GLOBAL_ATTN_BACKWARD_KERNEL_NAMES = [
    "aclnnFlashAttentionScoreGrad_FlashAttentionScoreGrad_FlashAttentionScoreGrad"
]


class DebugCpComm:
    def __init__(self):
        self.comm_x = 0
        self.hccs_x = 0
        self.roce_x = 0
        self.vector_time = 0
        self.attn_fw_time = 0
        self.attn_bw_time = 0
        self.total_time = 0
        self.cfg = SearchConfig()
        self.model_type = None
        self.cfg_no = 0


class CpCommPerfPredictor(CommPerfPredictor):
    def __init__(self, hard_info):
        super(CpCommPerfPredictor, self).__init__(hard_info)
        self.is_cp_modeling = False

    def get_communication_info_from_profile(
        self, cp_profile_time_info, hcom_info_tage_id, model, cp
    ):
        cp_profile_time_info.total_comm_time += hcom_info_tage_id.total_time_ms
        cp_profile_time_info.wait_comm_time += hcom_info_tage_id.wait_time_ms
        (
            cp_profile_time_info.attn_cp_time,
            cp_profile_time_info.attn_cpbw_time,
        ) = self.get_vectortime_from_profiling(model, cp)
        cp_profile_time_info.overlap_comm_time += hcom_info_tage_id.overlap_time_ms
        cp_profile_time_info.vector_cp_time += hcom_info_tage_id.vector_time_ms

    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, cp_profile_time_info: CpProfileTimeInfo
    ):
        if not self.is_cp_modeling:
            return
        config = model_config
        tp = config.tp
        cp = config.cp
        pp = config.pp
        s = config.seq_length / 1000

        # CP's communication volume is CP-1 times the forward KV, backward KV, and dKV per machine.
        if cp <= 1:
            return

        cp_total_comm_factor = cp * tp / self.max_hccs_rank_num
        hccs_x = cp_total_comm_factor * s / (tp * cp) * pp
        roce_x = (cp_total_comm_factor - 1) * s / (tp * cp) * pp
        cfg = SimpleParallelCfg(config_no, tp, cp, '', '', pp, '')

        max_domain = model_config.cp * model_config.tp
        min_domain = model_config.tp
        total_comm_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "cp_time",
            max_rank_num=max_domain,
            min_rank_num=min_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )

        hccs_x = cp_total_comm_factor * s / (tp * cp) * pp
        roce_x = (cp_total_comm_factor - 1) * s / (tp * cp) * pp
        cfg = SimpleParallelCfg(config_no, tp, cp, '', '', pp, '')

        # Here we consider only the attention of communication hiding, with forward CP-1 and backward CP.
        traffic = s * (cp - 1) / (tp * cp) * pp
        bandwidth_910b = (cp - 1)
        bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
        cp_x = traffic / bandwidth
        total_time = cp_profile_time_info.total_comm_time
        total_time_mdl_args = [cp_x, hccs_x, roce_x, total_time, cfg]
        total_comm_time_model.add_sample(*total_time_mdl_args)

        overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "cp_overlap",
            max_rank_num=max_domain,
            min_rank_num=min_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )
        vector_time = cp_profile_time_info.overlap_comm_time
        total_time_mdl_args = [cp_x, hccs_x, roce_x, vector_time, cfg]
        overlap_time_model.add_sample(*total_time_mdl_args)

        vector_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "cp_vector",
            max_rank_num=max_domain,
            min_rank_num=min_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )

        cp_vector_x = 0 if cp < 2 else cp - 2
        cp_vector_y = cp_profile_time_info.vector_cp_time
        vector_args = [cp_vector_x, hccs_x, roce_x, cp_vector_y, cfg]
        vector_overlap_time_model.add_sample(*vector_args)

        attn_fwd_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "cp_attn_fwd",
            max_rank_num=max_domain,
            min_rank_num=min_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )
        attn_fwd_x = s / tp / cp * (cp - 1) / cp
        attn_fw_time = cp_profile_time_info.attn_cp_time
        attn_fwd_args = [attn_fwd_x, hccs_x, roce_x, attn_fw_time, cfg]
        attn_fwd_overlap_time_model.add_sample(*attn_fwd_args)

        attn_bwd_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "cp_attn_bwd",
            max_rank_num=max_domain,
            min_rank_num=min_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )
        cp_attn_bwd_x = s / tp / cp
        attn_bw_time = cp_profile_time_info.attn_cpbw_time
        att_bwd_mdl_args = [cp_attn_bwd_x, hccs_x, roce_x, attn_bw_time, cfg]
        attn_bwd_overlap_time_model.add_sample(*att_bwd_mdl_args)

        debug_info = DebugCpComm()
        debug_info.comm_x = cp_x
        debug_info.hccs_x = hccs_x
        debug_info.roce_x = roce_x
        debug_info.total_time = total_time
        debug_info.vector_time = vector_time
        debug_info.attn_fw_time = attn_fw_time
        debug_info.attn_bw_time = attn_bw_time

        debug_info.cfg = model_config
        debug_info.cfg_no = config_no
        debug_info.model_type = str(type(total_comm_time_model))
        self.debug_info_list.append(debug_info)

    def fit(self):
        if not self.is_cp_modeling:
            return
        if self.is_cp_modeling:
            for module_name in ["cp_time", "cp_overlap", "cp_attn_fwd", "cp_attn_bwd", "cp_vector"]:
                for model in CommPerfLinearModelFactory.get_models_by_module_name(module_name):
                    if model:
                        model.fit()

    def debug(self, config_list):
        if not self.is_cp_modeling:
            return
        self.logger.debug(f"******************   CP modeling  ***********************")
        if "hccs" in CommPerfLinearModelFactory._instance_table["cp_time"].keys():
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
            attn_ag_model = CommPerfLinearModelFactory._instance_table["cp_time"]["hccs"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table["cp_time"]["hccs"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")

        if "roce" in CommPerfLinearModelFactory._instance_table["cp_time"].keys():
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
            attn_ag_model = CommPerfLinearModelFactory._instance_table["cp_time"]["roce"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table["cp_time"]["roce"]
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

        vector_overlap_time_model = None
        attn_fwd_overlap_time_model = None
        attn_bwd_overlap_time_model = None
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}"
        for i, config in enumerate(config_list):
            if config.cp <= 1:
                continue
            max_domain = config.cp * config.tp
            min_domain = config.tp

            vector_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_vector",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            attn_fwd_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_attn_fwd",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            attn_bwd_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_attn_bwd",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

        vector_overlap_time_model.debug(model_name="CP_vector_overlap_time")
        attn_fwd_overlap_time_model.debug(model_name="CP_attn_fwd_overlap_time")
        attn_bwd_overlap_time_model.debug(model_name="CP_attn_bwd_overlap_time")
        self.logger.debug(f"\n\n\n")

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
        # Attention, one of them is shadowed. attn_bw needs to be calculated.
        attention = attention / 1000
        attn_bw = attn_bw / 1000
        return attention, attn_bw

    def predict(self, search_cfg: SearchConfig):
        if not self.is_cp_modeling:
            return 0
        tp = search_cfg.tensor_model_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        cp = search_cfg.context_parallel_size
        s = search_cfg.seq_length / 1000
        cp_time = 0.0

        if cp > 1:
            traffic = s * (cp - 1) / (tp * cp) * pp
            min_domain = tp
            bandwidth_910b = (cp - 1)
            bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
            comm_x = traffic / bandwidth
            K = cp * tp / self.max_hccs_rank_num
            comm_y = (K) * s / (tp * cp) * pp
            comm_z = (K - 1) * s / (tp * cp) * pp
            iv_list = [comm_x, comm_y, comm_z]
            max_domain = cp * tp
            total_comm_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_time",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_overlap",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            vector_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_vector",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            attn_fwd_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_attn_fwd",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            attn_bwd_overlap_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "cp_attn_bwd",
                max_rank_num=max_domain,
                min_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            comm_time = total_comm_time_model.predict(*iv_list)
            overlap_time = overlap_time_model.predict(*iv_list)
            if comm_time - overlap_time > 0:
                cp_time = comm_time - overlap_time
                return cp_time


            attn_fwd_x = s / tp / cp * (cp - 1) / cp
            attn_time = attn_fwd_overlap_time_model.predict(*(attn_fwd_x,))
            # Attention and attn_bw need to be considered separately.
            attn_bwd_x = s / tp / cp
            attn_bw_time = attn_bwd_overlap_time_model.predict(*(attn_bwd_x,))

            cp_time1 = comm_time / 2 - attn_time * pp
            if cp_time1 < 0:
                cp_time1 = 0
            cp_time2 = comm_time / 2 - attn_bw_time * pp
            if cp_time2 < 0:
                cp_time2 = 0
            cp_time = cp_time1 + cp_time2
            if cp > 2:
                cp_vector_overlap_x = cp - 2

                cp_vector_time = vector_overlap_time_model.predict(*(cp_vector_overlap_x,))

                cp_time = cp_time - cp_vector_time
                self.logger.debug(
                    "cp_time:{}, attn_time:{}, attn_bw_time:{}, "
                    "cp_vector_time:{}".format(cp_time, attn_time, attn_bw_time, cp_vector_time)
                )
        if cp_time < 0:
            cp_time = 0.0
            self.logger.debug(f"The communication time of the CP is the waiting time.")
        return cp_time
