"""
算子预估
"""
import math

from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.communication.comm_perf_predictor_manager import CommPerfPredictorManager
from mindspeed.auto_settings.utils.logger import get_logger


class Communication(object):

    def __init__(self):
        self.logger = get_logger("Communication")
        self.predictor_mgr = CommPerfPredictorManager(get_system_config(), get_model_config())

    def train_models(self, profiling_results):
        self.predictor_mgr.communication_modeling(profiling_results)

    def get_communication_time(self, config):
        """
        通信执行耗时
        """
        search_cfg = config
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        vp = search_cfg.num_layers // (pp * search_cfg.num_layers_per_virtual_pipeline_stage) \
            if search_cfg.num_layers_per_virtual_pipeline_stage else 1
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size if search_cfg.expert_model_parallel_size else 1
        search_micro_batch_size = search_cfg.micro_batch_size
        num_layers = self.predictor_mgr.model_cfg.num_layers

        global_batch_size = self.predictor_mgr.model_cfg.global_batch_size
        cp_time = self.predictor_mgr.cp_predictor.predict(search_cfg) / pp
        dp_time = self.predictor_mgr.dp_predictor.predict(search_cfg)
        pp_time = self.predictor_mgr.pp_predictor.predict(search_cfg)
        ep_time = self.predictor_mgr.ep_predictor.predict(search_cfg) / pp
        model_micro_batch_size = 1

        self.predictor_mgr.communication_roce_predict_fix()
        # Time for each micro-batch in each layer.
        mc2_time = self.predictor_mgr.mc2_predictor.predict(search_cfg)
        tp_time = self.predictor_mgr.tp_predictor.predict(search_cfg)

        self.logger.debug(f"mc2_time:{mc2_time} tp_time:{tp_time}")
        use_mc2 = mc2_time < tp_time
        tp_time = min(mc2_time, tp_time) / pp

        micro_batch_num = global_batch_size / (dp * search_micro_batch_size)
        # total layer number，total global_batch_size
        layer_num = math.ceil(micro_batch_num * (num_layers / pp))
        search_model_mbs_ratio = search_micro_batch_size / model_micro_batch_size
        communication_time = (tp_time + cp_time + ep_time) * search_model_mbs_ratio * layer_num
        fw_communication_time = (tp_time + cp_time + ep_time) / 3

        # TODO pp的实际是还需要考虑mbs的影响
        print('communication:', communication_time, 'pp:', pp_time, 'mbs:', search_model_mbs_ratio, 'dp:', dp_time)
        return {
            "use_mc2": use_mc2,
            "fw_communication_time": fw_communication_time,
            "communication_time": communication_time,
            "pp_time": pp_time,
            "dp_time": dp_time,
            "tp_time": tp_time,
            "cp_time": cp_time,
            "ep_time": ep_time
        }
