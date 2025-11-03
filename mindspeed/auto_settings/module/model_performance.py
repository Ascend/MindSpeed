import math
from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.module.communication.comm_perf_predictor_manager import CommPerfPredictorManager
from mindspeed.auto_settings.module.operator.operator import OperatorPerformance
from mindspeed.auto_settings.module.operator.operator_re_profile import search_operator
from mindspeed.auto_settings.utils.logger import get_logger, change_stream_handler


class ModelPerformance(object):
    """
    Model Performance modeling
    """

    def __init__(self, hardware=None, working_dir: str = None, predictor_mgr=None, operator=None):
        self.predictor_mgr = CommPerfPredictorManager = predictor_mgr
        self.operator: OperatorPerformance = operator
        self.hardware = hardware
        self.logger = get_logger("ModelPerformance")

    def get_profiling_info(self, profiling_results):
        """
        :param profiling_results: List[List[SearchConfig, ProfilingRes]]
        :return:
        """
        self.predictor_mgr.communication_modeling(profiling_results)
        profiling_wo_mc2 = []
        for item in profiling_results:
            if item[0].use_ascend_mc2:
                pass
            else:
                profiling_wo_mc2.append(item)
        self.operator.model_operator_timer(profiling_wo_mc2)

    def performance(self, search_cfg, working_dir, profile_count, re_profile_flag=False, output=None, lock=None):
        self.operator.refresh_db_connection()
        tp = search_cfg.tensor_model_parallel_size
        dp = search_cfg.data_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        vp = search_cfg.num_layers // (pp * search_cfg.num_layers_per_virtual_pipeline_stage) \
            if search_cfg.num_layers_per_virtual_pipeline_stage else 1
        cp = search_cfg.context_parallel_size
        ep = search_cfg.expert_model_parallel_size if search_cfg.expert_model_parallel_size else 1
        num_layers = self.predictor_mgr.model_config.num_layers
        global_batch_size = self.predictor_mgr.model_config.global_batch_size
        model_micro_batch_size = 1
        search_micro_batch_size = search_cfg.micro_batch_size
        zero = search_cfg.use_distributed_optimizer
        change_stream_handler(self.logger, output)
        operator_time, unsampled_profiling, operator_fw_time, operator_bw_time = self.operator_performance(
            search_cfg, working_dir, profile_count, re_profile_flag, lock
        )
        comm_gap = 8

        self.predictor_mgr.communication_roce_predict_fix()

        # Time for each micro-batch in each layer.

        tp_time = self.predictor_mgr.tp_predictor.predict(search_cfg)

        use_mc2 = False

        tp_time = tp_time / pp
        cp_time = self.predictor_mgr.cp_predictor.predict(search_cfg) / pp
        dp_time = self.predictor_mgr.dp_predictor.predict(search_cfg)
        pp_time = self.predictor_mgr.pp_predictor.predict(search_cfg)
        ep_time = self.predictor_mgr.ep_predictor.predict(search_cfg) / pp

        micro_batch_num = global_batch_size / (dp * search_micro_batch_size)
        # total layer number，total global_batch_size
        layer_num = math.ceil(micro_batch_num * (num_layers / pp))
        search_model_mbs_ratio = search_micro_batch_size / model_micro_batch_size
        communication_time = (tp_time + cp_time + ep_time) * search_model_mbs_ratio * layer_num
        fw_communication_time = (tp_time + cp_time + ep_time) / 3
        fw_performance = operator_fw_time + fw_communication_time
        total_operator_time = operator_time * layer_num
        total_time = total_operator_time + communication_time
        self.logger.debug(f"comunication: {communication_time}, pp:, {pp_time}, mbs:, {search_model_mbs_ratio}, dp:, {dp_time}")
        total_communication_time = communication_time + pp_time * search_model_mbs_ratio + dp_time

        self.logger.debug(f"global_batch_size : {global_batch_size}, num_layers : {num_layers}, \
                          search_micro_batch_size : {search_micro_batch_size}, operator_time : {operator_time}, \
                          layer_num : {layer_num}")
        bubble_ratio = (pp - 1) / (micro_batch_num * vp + pp - 1)
        total_time = total_time / (1 - bubble_ratio)
        bubble_time = total_time * bubble_ratio
        total_time = total_time + pp_time * search_model_mbs_ratio + dp_time

        self.logger.debug(f"******************   total_time(ms)  ***********************")
        tplt = "{0:<2}\t{1:<2}\t{2:<2}\t{3:<2}\t{4:<2}\t{5:<2}\t{6:<8}\t{7:<10}\t{8:<8}\t{9:<8}\t{10:<8}\t{11:<8}"
        self.logger.debug(tplt.format('tp', 'dp', 'pp', 'vp', 'cp', 'ep', 'operator_time',
                          'comm_time', 'bubble_time', 'total_time', 'fw_time', chr(12288)))
        tplt = "{0:<2}\t{1:<2}\t{2:<2}\t{3:<2}\t{4:<2}\t{5:<2}\t{6:8.2f}\t{7:8.2f}\t{8:8.2f}\t{9:8.2f}\t{10:8.2f}"
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
                          pp_time * search_model_mbs_ratio, bubble_time,
                          cp_time * layer_num * search_model_mbs_ratio,
                          ep_time * layer_num * search_model_mbs_ratio, chr(12288)))
        return total_time, unsampled_profiling, use_mc2, fw_performance

    def operator_performance(self, search_cfg, working_dir, profile_count,
                             re_profile_flag=False, lock=None):
        tp = search_cfg.tensor_model_parallel_size
        cp = search_cfg.context_parallel_size
        pp = search_cfg.pipeline_model_parallel_size
        ep = search_cfg.expert_model_parallel_size
        dp = search_cfg.data_parallel_size
        mbs = search_cfg.micro_batch_size
        num_experts = search_cfg.num_experts
        communication = self.predictor_mgr
        model_config = communication.model_config
        unsampled_profiling_info = []
        operators, cp_exist_list, cp_diff_list, ep_exist_list, ep_diff_list, operator_not_found_list, operator_layer = \
            self.operator.cal_operator_timer(search_cfg)

        scal_flag = True if get_system_config().search_world_size > get_system_config().world_size else False
        self.logger.debug(f"{len(operators)}, {len(cp_exist_list)}, {len(cp_diff_list) },\
                          {len(ep_exist_list)}, {len(ep_diff_list) }, {len(operator_not_found_list)}")
        self.logger.debug("Total number of operators have been found is {0}".format((len(operators)
                                                                                     + len(cp_exist_list)
                                                                                     + len(cp_diff_list)
                                                                                     + len(ep_exist_list)
                                                                                     + len(ep_diff_list)
                                                                                     - len(operator_not_found_list))))
        self.logger.debug("Total number of operators use regression is {0}".format(len(operator_not_found_list)))
        # lock不为None时意味着开启并行模式，同时仅允许一个re-profiling进程执行,需要加锁处理
        if lock is not None:
            #判断是否re-profiling
            if (re_profile_flag and (profile_count[0] < 6) and
                len(operator_not_found_list) / (len(operators) + len(cp_exist_list) + len(cp_diff_list) +
                                                len(ep_exist_list) + len(ep_diff_list)) > 1):
                lock.acquire()
                try:
                    #再判断一遍，因为上一个re-profiling进程更新算子库可能影响结果
                    if (re_profile_flag and profile_count[0] < 6 and
                        len(operator_not_found_list) / (len(operators) + len(cp_exist_list) + len(cp_diff_list) +
                                                len(ep_exist_list) + len(ep_diff_list)) > 1):
                        self.logger.debug("re_profiling")
                    else:
                        self.logger.debug("previously re_profile, now no re_profile")
                        pass

                finally:
                    lock.release()
        else:
            if (re_profile_flag and profile_count[0] < 6 and
                len(operator_not_found_list) / (len(operators) + len(cp_exist_list) + len(cp_diff_list) +
                                                len(ep_exist_list) + len(ep_diff_list)) > 1):
                unsampled_profiling_info = search_operator(working_dir, search_cfg, communication, profile_count, scal_flag)
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
        num_experts = self.predictor_mgr.model_config.num_experts
        if num_experts and num_experts > 0:
            for operator in ep_exist_list:
                ep_each_exist_time = ep_each_exist_time + operator.duration
            ep_each_exist_time = ep_each_exist_time / (num_experts / 2 - num_experts / 4)
            operator_layer.ep_exist.fw = operator_layer.ep_exist.fw / (num_experts / 2 - num_experts / 4)
            operator_layer.ep_exist.bw = operator_layer.ep_exist.bw / (num_experts / 2 - num_experts / 4)
            if num_experts:
                operator_time = operator_time + (num_experts / ep - 1) * ep_each_exist_time
                operator_fw_time += operator_fw_time + (num_experts / ep - 1) * operator_layer.ep_exist.fw
                operator_bw_time += operator_bw_time + (num_experts / ep - 1) * operator_layer.ep_exist.bw

        # Convert to the total operator time for one micro_batch on a single node.
        operator_time = (operator_time * 0.001)
        operator_fw_time = (operator_fw_time * 0.001)
        operator_bw_time = (operator_bw_time * 0.001)
        return operator_time, unsampled_profiling_info, operator_fw_time, operator_bw_time
