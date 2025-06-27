from typing import List

from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.communication.comm_perf_linear_model_factory import (
    CommPerfLinearModelFactory,
)
from mindspeed.auto_tuning.module.communication.comm_perf_predictor import CommPerfPredictor, SimpleParallelCfg
from mindspeed.auto_tuning.module.communication.communication_profile import PpProfileTimeInfo


class DebugPpComm:
    def __init__(self):
        self.comm_x = 0
        self.total_time = 0
        self.cfg = SearchConfig()
        self.model_type = None
        self.cfg_no = 0


class PpCommPerfPredictor(CommPerfPredictor):
    def __init__(self, hard_info):
        super(PpCommPerfPredictor, self).__init__(hard_info)

    def get_communication_info_from_profile(self, pp_profile_time_info, hcom_info_tag_id, pp):
        pp_profile_time_info.each_pp_time = hcom_info_tag_id.min_pp_time

    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, pp_profile_time_info: PpProfileTimeInfo
    ):
        tp = model_config.tp
        cp = model_config.cp
        pp = model_config.pp
        dp = model_config.dp
        layers_per_vpp = model_config.layers_per_vpp if model_config.layers_per_vpp else 1
        comm_x = 1 / (layers_per_vpp * tp * cp)
        # pp does not need to consider cross modeling.
        comm_time = pp_profile_time_info.each_pp_time
        cfg = SimpleParallelCfg(config_no, tp, cp, dp, '', pp, layers_per_vpp)

        if pp > 1:
            max_domain = pp * dp * cp * tp
            min_domain = pp * dp * cp * tp
            pp_time_model = CommPerfLinearModelFactory.get_or_create_model(
                "pp",
                min_rank_num=max_domain,
                max_rank_num=min_domain,
                max_hccs_dev_num=self.max_hccs_rank_num,
            )
            # PPtime indicates the duration of each PP communication.
            pp_time_model.add_sample(*(comm_x, comm_time, cfg))

            debug_info = DebugPpComm()
            debug_info.comm_x = comm_x
            debug_info.total_time = comm_time

            debug_info.cfg = model_config
            debug_info.cfg_no = config_no
            debug_info.model_type = str(type(pp_time_model))
            self.debug_info_list.append(debug_info)

    def fit(self):
        for model in CommPerfLinearModelFactory.get_models_by_module_name("pp"):
            if model:
                model.fit()

    def debug(self, config_list: List[SearchConfig]):
        self.logger.debug(f"******************   PP modeling   ***********************")
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<1}"
        header = tplt.format("ConfigNo", "tp", "dp", "pp", "vp", "cp", "ep", chr(12288))
        self.logger.debug(header)
        for i, model_config in enumerate(config_list):
            if model_config.pp > 1:
                pp = model_config.pp
                dp = model_config.dp
                cp = model_config.cp
                tp = model_config.tp
                cur_row = tplt.format(
                    i,
                    tp,
                    dp,
                    pp,
                    str(model_config.layers_per_vpp),
                    cp,
                    model_config.ep,
                    chr(12288),
                )
                self.logger.debug(cur_row)

        for model in CommPerfLinearModelFactory.get_models_by_module_name("pp"):
            if model:
                model.debug(f"pp_{model.protocol_name}")

    def predict(self, search_cfg: SearchConfig):
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        vp = (
            search_cfg.num_layers // (pp * search_cfg.num_layers_per_virtual_pipeline_stage)
            if search_cfg.num_layers_per_virtual_pipeline_stage
            else 1
        )
        cp = search_cfg.context_parallel_size

        pp_time = 0.0  # TODO
        comm_x = 1 / (vp * tp * cp)
        # pp does not need to consider cross modeling.
        max_domain = pp * dp * cp * tp
        min_domain = pp * dp * cp * tp
        pp_time_model = CommPerfLinearModelFactory.get_or_create_model(
            "pp",
            min_rank_num=max_domain,
            max_rank_num=min_domain,
            max_hccs_dev_num=self.max_hccs_rank_num,
        )

        if pp > 1:
            each_pp_time = pp_time_model.predict(*(comm_x,))
            each_pp_time = each_pp_time * 2
            pp_time = each_pp_time * (pp * vp - 1) * 2
        return pp_time
