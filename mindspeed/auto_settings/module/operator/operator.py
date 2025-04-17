"""
算子预估
"""
from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.operator.operator_performance import OperatorPerformance
from mindspeed.auto_settings.module.operator.operator_re_profile import search_operator
from mindspeed.auto_settings.utils.logger import get_logger


class Operator(object):

    def __init__(self):
        self.logger = get_logger("Operator")
        model_config = get_model_config()
        work_dir = get_system_config().work_dir
        self.operator = OperatorPerformance(model_config, working_dir=work_dir)
        self.re_profile_flag = True

    def train_models(self, profiling_results):
        profiling_wo_mc2 = []
        for item in profiling_results:
            if item[0].use_ascend_mc2:
                pass
            else:
                profiling_wo_mc2.append(item)
        self.operator.model_operator_timer(profiling_wo_mc2)

    def get_operator_info(self, config):
        """
        算子执行耗时
        """
        search_cfg = config
        system_config = get_system_config()
        model_cfg = get_model_config()
        search_world_size = system_config.search_world_size
        work_dir = system_config.work_dir
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size
        operators, cp_exist_list, cp_diff_list, ep_exist_list, ep_diff_list, operator_not_found_list, operator_layer = \
            self.operator.cal_operator_timer(search_cfg)

        scal_flag = True if search_world_size > system_config.world_size else False
        self.logger.debug("Total number of operators have been found is {0}".format((len(operators)
                                                                                     + len(cp_exist_list)
                                                                                     + len(cp_diff_list)
                                                                                     + len(ep_exist_list)
                                                                                     + len(ep_diff_list)
                                                                                     - len(operator_not_found_list))))
        self.logger.debug("Total number of operators use regression is {0}".format(len(operator_not_found_list)))
        profile_count = [0]
        if (self.re_profile_flag and profile_count[0] < 6 and
                len(operator_not_found_list) / (len(operators) + len(cp_exist_list) + len(cp_diff_list) +
                                                len(ep_exist_list) + len(ep_diff_list)) > 1):
            unsampled_profiling_info = search_operator(work_dir, search_cfg, profile_count, scal_flag)
            operators, cp_exist_list, cp_diff_list, ep_exist_list, ep_diff_list, operator_not_found_list, operator_layer = \
                self.operator.cal_operator_timer(search_cfg)
        operator_time = 0.0
        operator_fw_time = 0.0
        operator_bw_time = 0.0
        for operator in operators:
            operator_time += operator.duration
        operator_fw_time += operator_layer.base_operator.fw
        operator_bw_time += operator_layer.base_operator.bw

        cp_exist_time = 0.0
        cp_diff_time = 0.0
        if cp > 1:
            for operator in cp_exist_list:
                cp_exist_time = cp_exist_time + operator.duration
            operator_time += cp_exist_time
            operator_fw_time += operator_layer.cp_exist.fw
            operator_bw_time += operator_layer.cp_exist.bw
            if cp > 2:
                for operator in cp_diff_list:
                    cp_diff_time = cp_diff_time + operator.duration
                operator_time += cp_diff_time * (cp - 2)
                operator_fw_time += operator_layer.cp_diff.fw * (cp - 2)
                operator_bw_time += operator_layer.cp_diff.bw * (cp - 2)

        ep_each_exist_time, ep_each_diff_time = 0.0, 0.0
        num_experts = model_cfg.num_experts
        if num_experts and num_experts > 0:
            for operator in ep_exist_list:
                ep_each_exist_time = ep_each_exist_time + operator.duration
            ep_each_exist_time = ep_each_exist_time / 2
            operator_layer.ep_exist.fw = operator_layer.ep_exist.fw / 2
            operator_layer.ep_exist.bw = operator_layer.ep_exist.bw / 2
            for operator in ep_diff_list:
                ep_each_diff_time = ep_each_diff_time + operator.duration
            ep_each_diff_time = ep_each_diff_time / 2
            operator_layer.ep_diff.fw = operator_layer.ep_diff.fw / 2
            operator_layer.ep_diff.bw = operator_layer.ep_diff.bw / 2
            if num_experts:
                operator_time = operator_time + (num_experts / ep - 1) * ep_each_exist_time
                operator_fw_time += operator_fw_time + (num_experts / ep - 1) * operator_layer.ep_exist.fw
                operator_bw_time += operator_bw_time + (num_experts / ep - 1) * operator_layer.ep_exist.bw

        # Convert to the total operator time for one micro_batch on a single node.
        operator_time = (operator_time * 0.001)
        operator_fw_time = (operator_fw_time * 0.001)
        operator_bw_time = (operator_bw_time * 0.001)

        return {
            "operator_time": operator_time,
            "operator_fw_time": operator_fw_time,
        }
