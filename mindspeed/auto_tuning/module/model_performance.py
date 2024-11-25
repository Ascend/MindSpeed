import math
from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.module.communication.communication import Communication
from mindspeed.auto_tuning.module.operator.operator import OperatorPerformance
from mindspeed.auto_tuning.module.operator.operator_re_profile import search_operator
from mindspeed.auto_tuning.utils.logger import get_logger


class ModelPerformance(object):
    """
    Model Performance modeling
    """

    def __init__(self, hardware=None, model_cfg: ModelConfig = None, working_dir: str = None):
        self.communication = Communication(hardware, model_cfg)
        self.operator = OperatorPerformance(model_cfg, working_dir=working_dir)
        self.hardware = hardware
        self.logger = get_logger("ModelPerformance")

    def get_profiling_info(self, profiling_results):
        self.communication.communication_modeling(profiling_results)
        profiling_wo_mc2 = []
        for item in profiling_results:
            if item[0].use_ascend_mc2:
                pass
            else:
                profiling_wo_mc2.append(item)
        self.operator.model_operator_timer(profiling_wo_mc2)

    def performance(self, search_cfg, working_dir, profile_count, re_profile_flag=False):
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        vp = search_cfg.num_layers // (pp * search_cfg.num_layers_per_virtual_pipeline_stage) \
            if search_cfg.num_layers_per_virtual_pipeline_stage else 1
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size if search_cfg.expert_model_parallel_size else 1
        num_layers = self.communication.model_cfg.num_layers
        global_batch_size = self.communication.model_cfg.global_batch_size
        model_micro_batch_size = self.communication.model_cfg.micro_batch_size
        search_micro_batch_size = search_cfg.micro_batch_size
        zero = search_cfg.use_distributed_optimizer
        operator_time, unsampled_profiling = self.operator_performance(
            search_cfg, working_dir, profile_count, re_profile_flag
        )
        comm_gap = 8

        # Time for each micro-batch in each layer.
        mc2_time = self.communication.mc2_model.performance(search_cfg)
        tp_time = self.communication.tp_model.performance(search_cfg)

        self.logger.debug(f"mc2_time:{mc2_time} tp_time:{tp_time}")
        use_mc2 = mc2_time < tp_time
        tp_time = min(mc2_time, tp_time)

        cp_time = self.communication.cp_model.performance(search_cfg)
        dp_time = self.communication.dp_model.performance(search_cfg)
        pp_time = self.communication.pp_model.performance(search_cfg)
        ep_time = self.communication.ep_model.performance(search_cfg)

        micro_batch_num = global_batch_size / (dp * search_micro_batch_size)
        # total layer number，total global_batch_size
        layer_num = math.ceil(micro_batch_num * (num_layers / pp))
        search_model_mbs_ratio = search_micro_batch_size / model_micro_batch_size
        communication_time = (tp_time + cp_time + ep_time) * search_model_mbs_ratio * layer_num
        total_operator_time = operator_time * layer_num
        total_time = total_operator_time + communication_time

        total_communication_time = communication_time + pp_time * search_model_mbs_ratio + dp_time
        self.logger.debug('global_batch_size : {}, num_layers : {}, search_micro_batch_size : {}, operator_time : {}, '
                          'layer_num : {}'.format(global_batch_size, num_layers, search_micro_batch_size,
                                                  operator_time, layer_num))
        bubble_ratio = (pp - 1) / (micro_batch_num * vp + pp - 1)
        total_time = total_time / (1 - bubble_ratio)
        bubble_time = total_time * bubble_ratio
        total_time = total_time + pp_time * search_model_mbs_ratio + dp_time

        self.logger.debug(f"******************   total_time(ms)  ***********************")
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:<8}\t{7:<10}\t{8:<8}\t{9:<8}"
        self.logger.debug(tplt.format('tp', 'dp', 'pp', 'vp', 'cp', 'ep', 'operator_time',
                          'comm_time', 'bubble_time', 'total_time', chr(12288)))
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<1}\t{6:8.2f}\t{7:8.2f}\t{8:8.2f}\t{9:8.2f}"
        self.logger.debug(tplt.format(tp, dp, pp, vp, cp, ep, total_operator_time,
                          total_communication_time, bubble_time, total_time, chr(12288)))
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
        return total_time, unsampled_profiling, use_mc2

    def operator_performance(self, search_cfg, working_dir, profile_count,
                             re_profile_flag=False):
        tp = search_cfg.tensor_model_parallel_size
        cp = search_cfg.context_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        ep = search_cfg.expert_model_parallel_size
        dp = search_cfg.data_parallel_size
        mbs = search_cfg.micro_batch_size
        num_experts = search_cfg.num_experts if search_cfg.num_experts else 1
        communication = self.communication
        model_config = communication.model_cfg
        unsampled_profiling_info = []
        operators, cp_exist_list, cp_diff_list, ep_exist_list, ep_diff_list, operator_not_found_list = \
            self.operator.cal_operator_timer(search_cfg)

        scal_flag = True if model_config.global_world_size > Hardware().num_devices else False
        self.logger.debug("Total number of operators have been found is {0}".format((len(operators)
                                                                                     + len(cp_exist_list)
                                                                                     + len(cp_diff_list)
                                                                                     + len(ep_exist_list)
                                                                                     + len(ep_diff_list))))
        if (re_profile_flag and profile_count[0] < 6 and
                len(operator_not_found_list) / (len(operators) + len(cp_exist_list) + len(cp_diff_list) +
                                                len(ep_exist_list) + len(ep_diff_list)) > 1):
            unsampled_profiling_info = search_operator(working_dir, search_cfg, communication, profile_count, scal_flag)
            operators, cp_exist_list, cp_diff_list, ep_exist_list, ep_diff_list, operator_not_found_list = \
                self.operator.cal_operator_timer(search_cfg)
        operator_time = 0.0
        for operator in operators:
            operator_time += operator.duration

        cp_exist_time = 0.0
        cp_diff_time = 0.0
        if cp > 1:
            for operator in cp_exist_list:
                cp_exist_time = cp_exist_time + operator.duration
            operator_time += cp_exist_time
            if cp > 2:
                for operator in cp_diff_list:
                    cp_diff_time = cp_diff_time + operator.duration
                operator_time += cp_diff_time * (cp - 2)

        ep_each_exist_time, ep_each_diff_time = 0.0, 0.0
        num_experts = self.communication.model_cfg.num_experts
        if num_experts and num_experts > 0:
            for operator in ep_exist_list:
                ep_each_exist_time = ep_each_exist_time + operator.duration
            ep_each_exist_time = ep_each_exist_time / 2
            for operator in ep_diff_list:
                ep_each_diff_time = ep_each_diff_time + operator.duration
            ep_each_diff_time = ep_each_diff_time / 2
            if num_experts:
                operator_time = operator_time + (num_experts / ep - 1) * ep_each_exist_time

        # Convert to the total operator time for one micro_batch on a single node.
        operator_time = (operator_time * 0.001)
        return operator_time, unsampled_profiling_info
