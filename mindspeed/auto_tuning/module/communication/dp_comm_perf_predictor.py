from typing import List

from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.communication.comm_perf_linear_model_factory import (
    CommPerfLinearModelFactory,
)
from mindspeed.auto_tuning.module.communication.comm_perf_predictor import CommPerfPredictor, SimpleParallelCfg
from mindspeed.auto_tuning.module.communication.communication_profile import DpProfileTimeInfo


class DebugDpComm:
    def __init__(self):
        self.comm_x = 0
        self.hccs_x = 0
        self.roce_x = 0
        self.total_time = 0
        self.ag_time = None
        self.rs_time = None
        self.cfg = SearchConfig()
        self.model_type = None
        self.cfg_no = 0


class DpCommPerfPredictor(CommPerfPredictor):
    """Data parallel communication time predictor.

    """

    def __init__(self, hard_info):
        super(DpCommPerfPredictor, self).__init__(hard_info)
        self.dp_model_list = ["dp_attn_ag", "dp_attn_rs"]
        self.mlp_debug_info_list = []

    def get_communication_info_from_profile(
        self, dp_profile_time_info: DpProfileTimeInfo, hcom_info_tag_id
    ):
        dp_profile_time_info.total_comm_time += hcom_info_tag_id.total_time_ms
        dp_profile_time_info.total_mlpzero_time += hcom_info_tag_id.mlp_zero_time_ms
        dp_profile_time_info.total_otherzero_time += (
            hcom_info_tag_id.total_time_ms - hcom_info_tag_id.mlp_zero_time_ms
        )
        dp_profile_time_info.mlp_ag_time += hcom_info_tag_id.mlp_ag_time_ms
        dp_profile_time_info.mlp_rs_time += hcom_info_tag_id.mlp_rs_time_ms
        dp_profile_time_info.attn_ag_time += hcom_info_tag_id.other_ag_time_ms
        dp_profile_time_info.attn_rs_time += hcom_info_tag_id.other_rs_time_ms

    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, dp_profile_time_info: DpProfileTimeInfo
    ):
        cp = model_config.cp
        dp = model_config.dp

        if dp * cp <= 1:
            return

        tp = model_config.tp
        pp = model_config.pp
        ep = model_config.ep
        experts = model_config.num_experts if model_config.num_experts else 1
        if model_config.num_experts:
            self.dp_model_list = ["dp_attn_ag", "dp_attn_rs", "dp_mlp_ag", "dp_mlp_rs"]
            # In the case that MOE communication is supported, if EP==1, the MLP and ATT of the DP are difficult to distinguish. Therefore, the MLP and ATT of the DP are not used.
            if ep == 1:
                return
        cfg = SimpleParallelCfg(config_no, tp, cp, dp, ep, pp, '')

        if dp * cp > 1:
            min_domain = cp * tp
            # attention layer
            attn_ag_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[0],
                min_rank_num=min_domain,
                max_rank_num=dp * cp * tp,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )
            attn_rs_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[1],
                min_rank_num=min_domain,
                max_rank_num=dp * cp * tp,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )
            attn_ag_time = dp_profile_time_info.attn_ag_time
            traffic = (dp * cp - 1) / (tp * pp)
            bandwidth_910b = (dp * cp - 1)
            bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
            attn_comm_x = traffic / bandwidth
            attn_factor = dp * cp * tp / self.max_hccs_rank_num
            attn_hccs_x1 = attn_factor / (tp * pp)
            attn_roce_x2 = (attn_factor - 1) / (tp * pp)

            attn_ag_args = [attn_comm_x, attn_hccs_x1, attn_roce_x2, attn_ag_time, cfg]
            attn_ag_model.add_sample(*attn_ag_args)

            attn_rs_time = dp_profile_time_info.attn_rs_time
            attn_rs_args = [attn_comm_x, attn_hccs_x1, attn_roce_x2, attn_rs_time, cfg]
            attn_rs_model.add_sample(*attn_rs_args)

            debug_info = DebugDpComm()
            debug_info.comm_x = attn_comm_x
            debug_info.hccs_x = attn_hccs_x1
            debug_info.roce_x = attn_roce_x2
            debug_info.total_time = dp_profile_time_info.total_comm_time
            debug_info.ag_time = attn_ag_time
            debug_info.rs_time = attn_rs_time

            debug_info.cfg = model_config
            debug_info.cfg_no = config_no
            debug_info.model_type = str(type(attn_ag_model))
            self.debug_info_list.append(debug_info)
        if experts and experts > 1 and dp * cp > ep:
            min_domain = tp * ep
            mlp_ag_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[2],
                min_rank_num=min_domain,
                max_rank_num=dp * cp * tp,
                max_hccs_dev_num=1,
            )
            mlp_rs_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[3],
                min_rank_num=min_domain,
                max_rank_num=dp * cp * tp,
                max_hccs_dev_num=1,
            )
            # MLP layer
            traffic = experts * (dp * cp / ep - 1) / tp / pp
            bandwidth_910b = (dp * cp / ep - 1)
            bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
            mlp_comm_x = traffic / bandwidth
            mlp_factor = dp * cp * tp / ep / self.max_hccs_rank_num
            mlp_hccs_x1 = experts * mlp_factor / (tp * pp)
            mlp_roce_x2 = experts * (mlp_factor - 1) / (tp * pp)
            mlp_rs_time = dp_profile_time_info.mlp_rs_time
            mlp_rs_args = [mlp_comm_x, mlp_hccs_x1, mlp_roce_x2, mlp_rs_time, cfg]
            mlp_rs_model.add_sample(*mlp_rs_args)

            mlp_ag_time = dp_profile_time_info.mlp_ag_time
            mlp_ag_args = [mlp_comm_x, mlp_hccs_x1, mlp_roce_x2, mlp_ag_time, cfg]
            mlp_ag_model.add_sample(*mlp_ag_args)

            debug_info = DebugDpComm()
            debug_info.comm_x = mlp_comm_x
            debug_info.hccs_x = mlp_hccs_x1
            debug_info.roce_x = mlp_roce_x2
            debug_info.total_time = dp_profile_time_info.total_comm_time
            debug_info.ag_time = mlp_ag_time
            debug_info.rs_time = mlp_rs_time

            debug_info.cfg = model_config
            debug_info.cfg_no = config_no
            debug_info.model_type = str(type(mlp_ag_model))
            self.mlp_debug_info_list.append(debug_info)

    def fit(self):
        for module_name in self.dp_model_list:
            for model in CommPerfLinearModelFactory.get_models_by_module_name(module_name):
                if model:
                    model.fit()
        # The MLP communication of the DP is currently only in the ROCE domain.

    def debug(self, config_list: List[SearchConfig]):
        self.logger.debug(f"******************   dp(ms)   ***********************")
        self.logger.debug(f"attention time :")
        if "hccs" in CommPerfLinearModelFactory._instance_table[self.dp_model_list[0]].keys():
            self.logger.debug(f"HCCS")
            tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}"
            self.logger.debug(tplt.format('x', 'ag_time', 'rs_time', 'total_time',
                                          'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
            for debug_info in self.debug_info_list:
                if "HCCS" in debug_info.model_type:
                    self.logger.debug(tplt.format(
                        round(debug_info.comm_x, 2),
                        round(debug_info.ag_time, 2),
                        round(debug_info.rs_time, 2),
                        round(debug_info.total_time, 3),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                        debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                        chr(12288)))
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}"
            self.logger.debug(tplt.format('rs_w', 'rs_b', 'ag_w', 'ag_b', chr(12288)))
            attn_ag_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[0]]["hccs"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[1]]["hccs"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")

        if "roce" in CommPerfLinearModelFactory._instance_table[self.dp_model_list[0]].keys():
            self.logger.debug(f"ROCE")
            tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}"
            self.logger.debug(tplt.format('x', 'ag_time', 'rs_time', 'total_time',
                                          'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
            for debug_info in self.debug_info_list:
                if "ROCE" in debug_info.model_type:
                    self.logger.debug(tplt.format(
                        round(debug_info.comm_x, 2),
                        round(debug_info.ag_time, 2),
                        round(debug_info.rs_time, 2),
                        round(debug_info.total_time, 3),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                        debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                        chr(12288)))
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}"
            self.logger.debug(tplt.format('rs_w', 'rs_b', 'ag_w', 'ag_b', chr(12288)))
            attn_ag_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[0]]["roce"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[1]]["roce"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")

        self.logger.debug(f"Cross")
        tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<8}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}\t{10:<1}"
        self.logger.debug(tplt.format('hccs_x', 'roce_x', 'ag_time', 'rs_time', 'total_time',
                                      'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
        for debug_info in self.debug_info_list:
            if "Cross" in debug_info.model_type:
                self.logger.debug(tplt.format(
                    round(debug_info.hccs_x, 2),
                    round(debug_info.roce_x, 2),
                    round(debug_info.ag_time, 2),
                    round(debug_info.rs_time, 2),
                    round(debug_info.total_time, 3),
                    debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                    debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                    chr(12288)))
        self.logger.debug(f"-----------")
        self.logger.debug(f"\n\n\n")

        self.logger.debug(f"mlp time :")
        if (len(self.dp_model_list) > 2 and
                "hccs" in CommPerfLinearModelFactory._instance_table[self.dp_model_list[2]].keys()):
            self.logger.debug(f"HCCS")
            tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}"
            self.logger.debug(tplt.format('x', 'ag_time', 'rs_time', 'total_time',
                                          'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
            for debug_info in self.mlp_debug_info_list:
                if "HCCS" in debug_info.model_type:
                    self.logger.debug(tplt.format(
                        round(debug_info.comm_x, 2),
                        round(debug_info.ag_time, 2),
                        round(debug_info.rs_time, 2),
                        round(debug_info.total_time, 3),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                        debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                        chr(12288)))
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}"
            self.logger.debug(tplt.format('rs_w', 'rs_b', 'ag_w', 'ag_b', chr(12288)))
            attn_ag_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[2]]["hccs"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[3]]["hccs"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")

        if (len(self.dp_model_list) > 2 and
                "roce" in CommPerfLinearModelFactory._instance_table[self.dp_model_list[2]].keys()):
            self.logger.debug(f"ROCE")
            tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<1}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}"
            self.logger.debug(tplt.format('x', 'ag_time', 'rs_time', 'total_time',
                                          'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
            for debug_info in self.mlp_debug_info_list:
                if "ROCE" in debug_info.model_type:
                    self.logger.debug(tplt.format(
                        round(debug_info.comm_x, 2),
                        round(debug_info.ag_time, 2),
                        round(debug_info.rs_time, 2),
                        round(debug_info.total_time, 3),
                        debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                        debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                        chr(12288)))
            self.logger.debug(f"-----------")
            tplt = "{0:<9}\t{1:<9}\t{2:<9}\t{3:<9}"
            self.logger.debug(tplt.format('rs_w', 'rs_b', 'ag_w', 'ag_b', chr(12288)))
            attn_ag_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[2]]["roce"]
            attn_rs_model = CommPerfLinearModelFactory._instance_table[self.dp_model_list[3]]["roce"]
            self.logger.debug(tplt.format(round(attn_ag_model.w, 3), round(attn_ag_model.b, 3),
                                          round(attn_rs_model.w, 3), round(attn_rs_model.b, 3),
                                          chr(12288)))
            self.logger.debug(f"----------------------")
        self.logger.debug(f"Cross")
        tplt = "{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<8}\t{5:<1}\t{6:<1}\t{7:<1}\t{8:<1}\t{9:<1}\t{10:<1}"
        self.logger.debug(tplt.format('hccs_x', 'roce_x', 'ag_time', 'rs_time', 'total_time',
                                      'No', 'tp', 'dp', 'pp', 'cp', 'ep', chr(12288)))
        for debug_info in self.mlp_debug_info_list:
            if "Cross" in debug_info.model_type:
                self.logger.debug(tplt.format(
                    round(debug_info.hccs_x, 2),
                    round(debug_info.roce_x, 2),
                    round(debug_info.ag_time, 2),
                    round(debug_info.rs_time, 2),
                    round(debug_info.total_time, 3),
                    debug_info.cfg_no, debug_info.cfg.tp, debug_info.cfg.dp,
                    debug_info.cfg.pp, debug_info.cfg.cp, debug_info.cfg.ep,
                    chr(12288)))
        self.logger.debug(f"-----------")
        self.logger.debug(f"\n\n\n")

    def predict(self, search_cfg):
        """Predict DP communication time according given model parallel config by calling it's
        fitted linear models.

        :param search_cfg: model parallel config
        :return: DP communication time.
        """
        dp = search_cfg.data_parallel_size
        cp = search_cfg.context_parallel_size
        if dp * cp <= 1:
            return 0.0

        tp = search_cfg.tensor_model_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        ep = search_cfg.expert_model_parallel_size if search_cfg.expert_model_parallel_size else 1
        experts = search_cfg.num_experts if search_cfg.num_experts else 1

        attn_time = 0.0
        attn_rs_time = 0.0
        attn_ag_time = 0.0
        mlp_time = 0.0
        mlp_rs_time = 0.0
        mlp_ag_time = 0.0

        if dp * cp > 1:
            min_domain = cp * tp
            attn_ag_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[0],
                min_rank_num=min_domain,
                max_rank_num=dp * cp * tp,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )
            attn_rs_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[1],
                min_rank_num=min_domain,
                max_rank_num=dp * cp * tp,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )

            traffic = (dp * cp - 1) / tp / pp  # HCCS/roce_x
            bandwidth_910b = (dp * cp - 1)
            bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
            attn_x = traffic / bandwidth
            attn_factor = dp * cp * tp / self.max_hccs_rank_num
            attn_cross_x1 = attn_factor / (tp * pp)
            attn_cross_x2 = (attn_factor - 1) / (tp * pp)
            attn_rs_time = attn_rs_model.predict(*(attn_x, attn_cross_x1, attn_cross_x2))
            attn_ag_time = attn_ag_model.predict(*(attn_x, attn_cross_x1, attn_cross_x2))
            attn_time = attn_ag_time + attn_rs_time

        if experts and experts > 1 and dp * cp > ep:
            # The MLP communication of the DP is currently only in the ROCE domain.
            min_domain = tp * ep
            max_domain = dp * cp * tp
            mlp_ag_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[2],
                min_rank_num=min_domain,
                max_rank_num=max_domain,
                max_hccs_dev_num=1,
            )
            mlp_rs_model = CommPerfLinearModelFactory.get_or_create_model(
                self.dp_model_list[3],
                min_rank_num=min_domain,
                max_rank_num=max_domain,
                max_hccs_dev_num=1,
            )

            traffic = experts * (dp * cp / ep - 1) / tp / pp
            bandwidth_910b = (dp * cp / ep - 1)
            bandwidth = self.hard_info.calbandwidth(bandwidth_910b, min_domain)
            mlp_x = traffic / bandwidth
            mlp_factor = dp * cp * tp / ep / self.max_hccs_rank_num
            mlp_cross_x1 = experts * mlp_factor / (tp * pp)
            mlp_cross_x2 = experts * (mlp_factor - 1) / (tp * pp)
            mlp_rs_time = mlp_rs_model.predict(*(mlp_x, mlp_cross_x1, mlp_cross_x2))
            mlp_ag_time = mlp_ag_model.predict(*(mlp_x, mlp_cross_x1, mlp_cross_x2))
            mlp_time = mlp_ag_time + mlp_rs_time

        overlap_time = 0.0
        zero_enabled = search_cfg.use_distributed_optimizer
        if zero_enabled:
            if pp > 1:
                overlap_time += (pp - 1) / pp * (attn_rs_time + mlp_rs_time)
            if pp > 2:
                overlap_time += (pp - 2) / pp * (attn_ag_time + mlp_ag_time)

        dp_time = attn_time + mlp_time - overlap_time
        if dp_time < 0:
            dp_time = 0.0
        # DP here is the total gbs time effect
        return dp_time
