"""
算子预估
"""
import math

from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.module.operator.operator import Operator
from mindspeed.auto_settings.module.communication.communication import Communication
from mindspeed.auto_settings.utils.utils import get_num_warmup_micro_batches


class TimeCost(object):

    def __init__(self):
        self.logger = get_logger("TimeCost")
        self.operator = Operator()
        self.communication = Communication()

    def train_models(self, profile_results):
        self.operator.train_models(profile_results)
        self.communication.train_models(profile_results)

    def get_communication_time(self, search_cfg):
        """
        获取通信相关耗时
        """
        return self.communication.get_communication_time(search_cfg)

    def get_operator_time(self, search_cfg):
        """
        获取算子耗时信息
        """
        return self.operator.get_operator_info(search_cfg)

    def get_time_cost(self, search_cfg, memory_info):
        """
        考虑vpp，返回最终的耗时信息
        """
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        vp = search_cfg.num_layers // (pp * search_cfg.num_layers_per_virtual_pipeline_stage) \
            if search_cfg.num_layers_per_virtual_pipeline_stage else 1
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size if search_cfg.expert_model_parallel_size else 1

        num_layers = get_model_config().num_layers
        global_batch_size = get_model_config().global_batch_size
        model_micro_batch_size = 1
        search_micro_batch_size = search_cfg.micro_batch_size
        micro_batch_num = global_batch_size / (dp * search_micro_batch_size)
        layer_num = math.ceil(micro_batch_num * (num_layers / pp))
        search_model_mbs_ratio = search_micro_batch_size / model_micro_batch_size
        bubble_ratio = (pp - 1) / (micro_batch_num * vp + pp - 1)

        operator_info = self.get_operator_time(search_cfg)
        operator_time = operator_info["operator_time"]
        operator_fw_time = operator_info["operator_fw_time"]

        communication_info = self.get_communication_time(search_cfg)
        use_mc2 = communication_info["use_mc2"]
        fw_communication_time = communication_info["fw_communication_time"]
        communication_time = communication_info["communication_time"]
        pp_time = communication_info["pp_time"]
        dp_time = communication_info["dp_time"]
        tp_time = communication_info["tp_time"]
        cp_time = communication_info["cp_time"]
        ep_time = communication_info["ep_time"]

        fw_performance = operator_fw_time + fw_communication_time
        total_operator_time = operator_time * layer_num
        total_time = total_operator_time + communication_time

        self.logger.debug('global_batch_size : {}, num_layers : {}, search_micro_batch_size : {}, operator_time : {}, '
                          'layer_num : {}'.format(global_batch_size, num_layers, search_micro_batch_size,
                                                  operator_time, layer_num))
        total_time = total_time / (1 - bubble_ratio)
        bubble_time = total_time * bubble_ratio
        total_time = total_time + pp_time * search_model_mbs_ratio + dp_time

        need_recompute = memory_info["need_recompute"]
        model_cfg = get_model_config()
        layer_calculate = memory_info["layer_calculate"]
        warmup_micro_batchs, total_num_micro_batches = get_num_warmup_micro_batches(search_cfg, model_cfg)
        num_layers = model_cfg.num_layers // search_cfg.pp

        self.logger.debug(f"******************   total_time(ms)  ***********************")
        tplt = "{0:<2}\t{1:<2}\t{2:<2}\t{3:<2}\t{4:<2}\t{5:<2}\t{6:<8}\t{7:<10}\t{8:<8}\t{9:<8}\t{10:<8}\t{11:<8}"
        self.logger.debug(tplt.format('tp', 'dp', 'pp', 'vp', 'cp', 'ep', 'operator_time',
                                      'comm_time', 'bubble_time', 'total_time', 'fw_time', chr(12288)))
        tplt = "{0:<2}\t{1:<2}\t{2:<2}\t{3:<2}\t{4:<2}\t{5:<2}\t{6:8.2f}\t{7:8.2f}\t{8:8.2f}\t{9:8.2f}\t{10:8.2f}"
        total_communication_time = communication_time + pp_time * search_model_mbs_ratio + dp_time
        self.logger.debug(tplt.format(tp, dp, pp, vp, cp, ep, total_operator_time,
                                      total_communication_time, bubble_time, total_time, operator_fw_time, chr(12288)))
        tplt = "{0:<4}\t{1:<4}\t{2:<4}\t{3:<4}\t{4:<4}\t{5:<4}"
        self.logger.debug(f"*******   each layer mbs communication time(ms)  ********")
        self.logger.debug(tplt.format('tp_time', 'dp_time', 'pp_time',
                                      'bubble', 'cp_time', 'ep_time', chr(12288)))
        tplt = "{0:4.2f}\t{1:4.2f}\t{2:4.2f}\t{3:4.2f}\t{4:4.2f}\t{5:4.2f}"
        self.logger.debug(tplt.format(tp_time, dp_time, pp_time,
                                      bubble_time, cp_time, ep_time, chr(12288)))
        self.logger.debug(f"end-to-end, each*(global_batch_size / (dp *pp))* num_layers")
        tplt = "{0:<4}\t{1:<4}\t{2:<4}\t{3:<4}\t{4:<4}\t{5:<4}"
        self.logger.debug(tplt.format('tp_time', 'dp_time', 'pp_time',
                                      'bubble', 'cp_time', 'ep_time', chr(12288)))
        tplt = "{0:4.0f}\t{1:4.2f}\t{2:4.2f}\t{3:4.2f}\t{4:4.2f}\t{5:4.2f}"
        self.logger.debug(tplt.format(tp_time * layer_num * search_model_mbs_ratio, dp_time,
                                      pp_time, bubble_time, cp_time * layer_num * search_model_mbs_ratio,
                                      ep_time * layer_num * search_model_mbs_ratio, chr(12288)))
        self.logger.debug(f"before recompute, perf = {total_time}")
        self.logger.debug(f"success enter recompute_solver and tp = {search_cfg.tensor_model_parallel_size} "
                          f"pp = {search_cfg.pipeline_model_parallel_size} "
                          f"layers_per_vpp={search_cfg.num_layers_per_virtual_pipeline_stage} "
                          f"dp = {search_cfg.data_parallel_size} cp = {search_cfg.context_parallel_size} "
                          f"ep = {search_cfg.expert_model_parallel_size} zero = {search_cfg.use_distributed_optimizer}")
        if not need_recompute:
            total_time = total_time - total_num_micro_batches * num_layers * fw_performance
            return {
                "use_mc2": use_mc2,
                "total_time": total_time,
                "num_layers": 0
            }
        if search_cfg.layers_per_vpp:
            time_cost = total_num_micro_batches * layer_calculate * fw_performance
        else:
            time_cost = total_num_micro_batches * layer_calculate * fw_performance
        total_time = total_time - time_cost
        num_layers = num_layers - layer_calculate
        return {
            "use_mc2": use_mc2,
            "total_time": total_time,
            "num_layers": num_layers
        }
