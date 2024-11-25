import json
import time

from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.operator.operator_profile_get import OriginalProfileDataList
from mindspeed.auto_tuning.module.operator.operator_note_cal import OperatorNoteList
from mindspeed.auto_tuning.module.operator.operator_base_block import BaseBlock
from mindspeed.auto_tuning.module.operator.operator_change_block_cp import CpBlock
from mindspeed.auto_tuning.module.operator.operator_change_block_ep import EpBlock
from mindspeed.auto_tuning.module.operator.operator_elemental import DictCalShape
from mindspeed.auto_tuning.module.operator.operator_database import DataBase, Operator, OperatorHistory
from mindspeed.auto_tuning.module.operator.operator_shape_analysis import separate_ep, separate_cp_tp
from mindspeed.auto_tuning.module.operator.operator_shape_cal import (model_operator_with_tp,
                                                                      model_operator_with_shape,
                                                                      cal_new_shape_tce,
                                                                      cal_operator_flops,
                                                                      cal_operator_duration_with_shape)


class OperatorPerformance(object):
    """
    Operator Performance modeling
        1. Test Run
        2. Profiling Parser
        3. Modeling [taking the results from the test run and placing them into all modules within
        modeling for mathematical modeling estimation, then dynamically adjusting the test run configuration and
        performing mathematical modeling estimation again [loop]]
        4. Return recommended configuration
    """

    def __init__(self, model_config: ModelConfig, working_dir: str):
        self.db = DataBase(working_dir=working_dir)
        self.origin_profile_data_list = OriginalProfileDataList()
        self.model_config = model_config
        self._logger = get_logger('operator')

        self.base_block = BaseBlock()
        self.cp_block = CpBlock()
        self.ep_block = EpBlock()

        self.dict_model = dict()

    def model_operator_timer(self, profiling_results):
        """
        Model shape and duration based on the profiling result. Currently, all operator only takes one micro_batch,
        no matter whether pp is enabled.
        """
        self.dict_model = dict()
        # 1. get original data
        self.origin_profile_data_list.get_origin_profile_data(profiling_results)
        # 2. get base_block
        self.base_block.get_block(self.origin_profile_data_list.data_list)
        # 3. get change block
        self.cp_block.get_block(self.origin_profile_data_list, self.base_block)
        if self.origin_profile_data_list.data_list[0].config_info.num_experts:
            self.ep_block.get_block(self.origin_profile_data_list, self.base_block)

        st_time = time.time()
        # 第 3 轮, Note数据表重新排序，按照新生成的index_name分类
        operator_note_list = OperatorNoteList()
        operator_note_list.get_operator_note(self)

        self.get_history_db(operator_note_list.operator_note_list)
        self._logger.info(f'-----------------------------------')
        # 第 4 轮，基于operator_note_model建shape计算operator_model_dao
        self.get_operator_model(operator_note_list.operator_note_dict)

        self._logger.info("get operator_base_dao successful")
        self._logger.info("total number of operator_note_dict: {}, dict_model {}, base_block {}, cp_block {}, "
                          "ep_block {}".format(len(operator_note_list.operator_note_dict), len(self.dict_model),
                                               len(self.base_block.fw) + len(self.base_block.bw),
                                               len(self.cp_block.fw) + len(self.cp_block.bw) + len(self.cp_block.re),
                                               len(self.ep_block.fw) + len(self.ep_block.bw) + len(self.ep_block.re)))
        self._logger.info(f'total time: {time.time() - st_time}')
        self._logger.info(f'---------------------------【Add operator to db】---------------------------')

    def get_history_db(self, operator_note_list):
        self._logger.info("******************   duration_sum(ms)  ***********************")
        tplt = "{0:<1}\t{1:<1}\t{2:<1}\t{3:<1}\t{4:<1}\t{5:<8}\t{6:<8}\t{7:<8}"
        self._logger.info(tplt.format('tp', 'dp', 'pp', 'cp', 'ep', 'duration_sum', 'operator_num', chr(12288)))
        self._logger.info(f'--------------------------------------------------------------------------')
        for (index, operator_note) in enumerate(operator_note_list):
            operator_history_list = []
            duration_sum = 0
            operator_list = operator_note.fw + operator_note.bw
            for operator in operator_list:
                duration_sum += float(operator.duration)
                operator_history = OperatorHistory(types=operator.type,
                                                   accelerator_core=operator.accelerator_core,
                                                   input_shape=operator.input_shape,
                                                   output_shape=operator.output_shape,
                                                   duration=operator.duration,
                                                   device=Hardware().device_type,
                                                   jit=operator.jit,
                                                   cann="8.0.RC2.alpha002",
                                                   driver="24.1.rc2.b030",
                                                   dtype=self.model_config.dtype.value[0])
                operator_history_list.append(operator_history.convert_to_dict())
            # 历史数据
            self.db.operator_history_dao.insert_history(operator_history_list)
            self._logger.info(tplt.format(
                self.origin_profile_data_list.data_list[index].config_info.tp,
                self.origin_profile_data_list.data_list[index].config_info.dp,
                self.origin_profile_data_list.data_list[index].config_info.pp,
                self.origin_profile_data_list.data_list[index].config_info.cp,
                self.origin_profile_data_list.data_list[index].config_info.ep,
                int(duration_sum), len(operator_note.fw), len(operator_note.bw), chr(12288)))

    def get_operator_model(self, operator_note_dict):
        operator_list = self.base_block.fw + self.base_block.bw
        self.get_operator_model_dao(operator_list, operator_note_dict)
        self.base_block.exist_cal_list = self.get_dict_base_shape(operator_list, operator_note_dict)

        operator_list = self.cp_block.fw + self.cp_block.bw + self.cp_block.re
        self.get_operator_model_dao(operator_list, operator_note_dict)
        self.cp_block.exist_cal_list = self.get_dict_base_shape(operator_list, operator_note_dict)

        operator_list = self.cp_block.diff_list.fw + self.cp_block.diff_list.bw + self.cp_block.diff_list.re
        self.get_operator_model_dao(operator_list, operator_note_dict)
        self.cp_block.diff_cal_list = self.get_dict_base_shape(operator_list, operator_note_dict)

        operator_list = self.ep_block.fw + self.ep_block.bw + self.ep_block.re
        self.get_operator_model_dao(operator_list, operator_note_dict)
        self.ep_block.exist_cal_list = self.get_dict_base_shape(operator_list, operator_note_dict)

        operator_list = self.ep_block.diff_list.fw + self.ep_block.diff_list.bw + self.ep_block.diff_list.re
        self.get_operator_model_dao(operator_list, operator_note_dict)
        self.ep_block.diff_cal_list = self.get_dict_base_shape(operator_list, operator_note_dict)


    def get_dict_base_shape(self, operator_list, operator_note_dict):
        re_list = []
        for operator in operator_list:
            index_name = operator.index_name
            # cp 1  tp 1 2 4 8  -> shape_tp
            # cp 2  tp 1 2 4 8  -> shape_tp
            # shape_cp
            # model the shape, according to the change between profiling result with different tp value, calculate the
            # change formula for each position in the operator's shape
            results = operator_note_dict[index_name]
            # take ep first
            result = separate_ep(results)
            input_shape_cal, output_shape_cal = separate_cp_tp(result)
            dict_shape = DictCalShape()
            dict_shape.name = operator.name
            dict_shape.index_name = index_name
            dict_shape.accelerator_core = operator.accelerator_core
            dict_shape.types = operator.type
            dict_shape.input_cal = json.dumps(input_shape_cal)
            dict_shape.output_cal = json.dumps(output_shape_cal)
            re_list.append(dict_shape)
        return re_list

    def get_operator_model_dao(self, operator_list, operator_note_dict):
        for operator in operator_list:
            index_name = operator.index_name
            # cp 1  tp 1 2 4 8  -> shape_tp
            # cp 2  tp 1 2 4 8  -> shape_tp
            # shape_cp
            # model the shape, according to the change between profiling result with different tp value, calculate the
            # change formula for each position in the operator's shape
            results = operator_note_dict[index_name]
            # input_shape_cal, has the same format as the shape array, with positive numbers representing unchanged
            # positions, and negative numbers representing varying positions. Assuming the number is num, the variation
            # rule is -num/tp.

            # duration is modeled based on the same position operators and TPs. For operators with shape changes,
            # it is initially observed that as TP increases [2, 4, 8], the duration decreases approximately by a
            # factor of 2.
            # tp_model_w is the number calculated when the duration decreases. Theoretically, it is the duration of the
            # operator when tp=1. Therefore, when tp = 2, duration(2) = tp_model_w/2; tp_model_b is the redundancy
            # coefficient.
            tp_model_w, tp_model_b = model_operator_with_tp(results)

            # duration is modeled based on the Flops calculated from the shape. For all operators,
            # F(duration) = shape_model_w * Flops + shape_model_b.
            history_results = self.db.operator_history_dao.get_by_types_and_accelerator_core(
                operator.accelerator_core, operator.type)
            shape_model_w, shape_model_b = model_operator_with_shape(history_results)
            dict_shape = {
                'index_name': index_name,
                'accelerator_core': operator.accelerator_core,
                'model_w': float(tp_model_w),
                'model_b': float(tp_model_b),
                'shape_model_w': shape_model_w,
                'shape_model_b': shape_model_b,
            }
            accelerator_core_exist = False
            if dict_shape["index_name"] in self.dict_model.keys():
                for dict_temp in self.dict_model[dict_shape["index_name"]]:
                    if dict_temp['accelerator_core'] == dict_shape['accelerator_core']:
                        accelerator_core_exist = True
                        break
                if not accelerator_core_exist:
                    self.dict_model[dict_shape["index_name"]].append(dict_shape)
            else:
                self.dict_model[dict_shape["index_name"]] = [dict_shape]

    def getmodel_by_accelerator_core_and_index_name(self, accelerator_core, index_name):
        for dict_shape in self.dict_model.get(index_name):
            if dict_shape['accelerator_core'] == accelerator_core:
                return dict_shape
        self._logger.info("can not find the accelerator_core!")
        return self.dict_model.get(index_name)[0]

    def cal_operator_timer_bymodel(self, operator_list, search_cfg: SearchConfig, ratio=0.3,
                                   re_profiling_flag=False):
        operator_list_re = []

        operator_total_num = len(operator_list)
        operator_not_found = []
        for operator_base in operator_list:
            # Calculate input_shape and output_shape based on tp, cp, and ep.
            input_shape = cal_new_shape_tce(operator_base.input_cal, search_cfg)
            output_shape = cal_new_shape_tce(operator_base.output_cal, search_cfg)
            # 1. search duration through operator_history based on input_shape and types
            operators = self.db.operator_history_dao.get_by_types_and_input_shape(operator_base.types, input_shape)
            if len(operators) > 0:
                operator_list_re.append(Operator(name=operator_base.index_name, types=operator_base.types,
                                                 accelerator_core=operator_base.accelerator_core,
                                                 input_shape=input_shape,
                                                 output_shape=output_shape,
                                                 duration=operators[0].duration))

            # 2. Predict the results based on the tp --- duration modeling results.
            else:
                operator_not_found.append([OperatorHistory(types=operator_base.types,
                                                           accelerator_core=operator_base.accelerator_core,
                                                           input_shape=input_shape,
                                                           output_shape=output_shape,
                                                           duration=0,
                                                           device=Hardware().device_type,
                                                           jit=int(self.model_config.jit_compile),
                                                           cann="8.0.RC2.alpha002",
                                                           driver="24.1.rc2.b030",
                                                           dtype=self.model_config.dtype.value[0]),
                                           operator_base.index_name])

        operator_not_found_total_num = len(operator_not_found)
        if operator_not_found_total_num / operator_total_num > ratio and re_profiling_flag:
            return operator_list_re, operator_not_found

        else:
            # If the proportion of missing operators is relatively low, by default, supplement the operators using
            # linear interpolation.
            if re_profiling_flag:
                self._logger.info(
                    f'The total operator not found proportion is {operator_not_found_total_num / operator_total_num},'
                    f' there is no need for re profiling.')
            for operator_cal_base in operator_not_found:
                operator_base, operator_index_name = operator_cal_base
                operator_model = self.getmodel_by_accelerator_core_and_index_name(
                    operator_base.accelerator_core, operator_index_name
                )
                flops = cal_operator_flops(operator_base.input_shape, operator_base.output_shape,
                                           operator_base.types)

                duration = cal_operator_duration_with_shape(operator_model["shape_model_w"],
                                                            operator_model["shape_model_b"],
                                                            flops)
                operator_list_re.append(Operator(name=operator_index_name, types=operator_base.types,
                                                 accelerator_core=operator_base.accelerator_core,
                                                 input_shape=operator_base.input_shape,
                                                 output_shape=operator_base.output_shape,
                                                 duration=duration))
        return operator_list_re, operator_not_found

    def cal_operator_timer(self, search_cfg: SearchConfig) -> tuple:
        """
            External interface, returns the duration based on changes in tp.
        """
        # Obtain all operators of a model layer.
        operator_not_found = []
        if len(self.base_block.fw) == 0:
            return [], [], [], 1, 1, 1
        operator_base_list = self.base_block.exist_cal_list
        operator_list, operator_not_found_list = self.cal_operator_timer_bymodel(operator_base_list,
                                                                                 search_cfg)
        operator_not_found.extend(operator_not_found_list)
        cp_operator_exist_list = self.cp_block.exist_cal_list
        cp_operator_diff_list = self.cp_block.diff_cal_list
        ep_operator_exist_list = self.ep_block.exist_cal_list
        ep_operator_diff_list = self.ep_block.diff_cal_list
        cp_exist_list, cp_exist_not_found_list = [], []
        if len(cp_operator_exist_list) > 0:
            cp_exist_list, cp_exist_not_found_list = self.cal_operator_timer_bymodel(
                cp_operator_exist_list,
                search_cfg)
            if search_cfg.cp > 1:
                operator_not_found.extend(cp_exist_not_found_list)
        cp_diff_list, cp_diff_not_found_list = [], []
        if len(cp_operator_diff_list) > 0:
            cp_diff_list, cp_diff_not_found_list = self.cal_operator_timer_bymodel(cp_operator_diff_list,
                                                                                  search_cfg)
            if search_cfg.cp > 1:
                operator_not_found.extend(cp_diff_not_found_list)
        ep_exist_list, ep_exist_not_found_list = [], []
        if len(ep_operator_exist_list) > 0:
            ep_exist_list, ep_exist_not_found_list = self.cal_operator_timer_bymodel(
                ep_operator_exist_list, search_cfg
            )
            if search_cfg.ep and search_cfg.ep > 1:
                operator_not_found.extend(ep_exist_not_found_list)
        ep_diff_list, ep_diff_not_found_list = [], []
        if len(ep_operator_diff_list) > 0:
            ep_diff_list, ep_diff_not_found_list = self.cal_operator_timer_bymodel(ep_operator_exist_list,
                                                                                   search_cfg)
            if search_cfg.ep and search_cfg.ep > 1:
                operator_not_found.extend(ep_diff_not_found_list)
        self.db.insert_not_found_list(operator_not_found)
        return operator_list, cp_exist_list, cp_diff_list, ep_exist_list, ep_diff_list, operator_not_found
