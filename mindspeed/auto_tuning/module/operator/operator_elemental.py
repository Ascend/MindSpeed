# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import numpy as np
from mindspeed.auto_tuning.utils.logger import get_logger


class ProfileList(object):
    def __init__(self):
        self.fw = []
        self.bw = []
        self._logger = get_logger('ProfileList')

    # MatMulV3与MatMulV2做模糊匹配
    def matchv2_v3_list_type(self, operator1, operator2):
        if operator1['list_type'] == operator2['list_type']:
            return True
        if "MatMulV" in operator1['list_type'] and "MatMulV" in operator2['list_type']:
            return True
        return False

    # MatMulV3与MatMulV2做模糊匹配
    def matchv2_v3_operator(self, operator1, operator2):
        if operator1.type == operator2.type:
            return True
        if "MatMulV" in operator1.type and "MatMulV" in operator2.type:
            return True
        return False

    # 最长公共连续子序列的解法
    def reset_index_name(self, list1, list2):
        len_list1, len_list2 = len(list1), len(list2)
        list_lift = []
        list_right = []
        first_mat = 0
        for i in range(len_list1):
            if list1[i].index_name == '':
                list_lift.append({'list_type': list1[i].type, 'list_index': i})
        for i in range(len_list2):
            list_right.append({'list_type': list2[i].type, 'list_index': i})
        max_len = -1
        while max_len != 0:
            max_len, mat_index = self.find_max_length(list_lift, list_right)
            op_list = [list_lift, list_right, list1, list2]
            list1, list_lift, list_right = self.swap_max_length(op_list, max_len, mat_index)
            if first_mat == 0:
                first_mat = mat_index[0] + 1
        return list1, first_mat

    # 找到最长公共连续子序列
    def find_max_length(self, list_lift, list_right):
        len_list1, len_list2 = len(list_lift), len(list_right)
        mark_matrix = np.zeros((len_list1, len_list2), dtype=int)
        max_len = 0
        mat_index = [0, 0]
        for i in range(len_list1):
            for j in range(len_list2):
                if self.matchv2_v3_list_type(list_lift[i], list_right[j]):
                    if i == 0 or j == 0:
                        mark_matrix[i, j] = 1
                    else:
                        mark_matrix[i, j] = mark_matrix[i - 1][j - 1] + 1
                if (mark_matrix[i, j] > max_len):
                    max_len = mark_matrix[i, j]
                    mat_index = [i, j]
                # 当算子匹配长度都为1的单算子时，且命中了多个算子，吸附最近一个匹配的方向
                if max_len == 1 and mark_matrix[i, j] == 1:
                    if len_list1 - i < mat_index[0]:
                        mat_index = [i, j]
        return max_len, mat_index

    # 根据最长公共连续子序列赋值
    def swap_max_length(self, op_list, max_len, mat_index):
        list_lift, list_right, list1, list2 = op_list
        for i in range(max_len):
            # MatMulV3与MatMulV2做模糊匹配
            if self.matchv2_v3_operator(list1[list_lift[mat_index[0] - i]['list_index']],
            list2[list_right[mat_index[1] - i]['list_index']]):
                list1[list_lift[mat_index[0] - i]['list_index']].index_name = list2[
                    list_right[mat_index[1] - i]['list_index']].index_name
            else:
                list1[list_lift[mat_index[0] - i]['list_index']].index_name = list1[
                    list_lift[mat_index[0] - i]['list_index']].index_name

        for i in range(max_len):
            list_lift.pop(mat_index[0] - i)
            list_right.pop(mat_index[1] - i)
        return list1, list_lift, list_right

    def reset_index_name_single(self, list1, list2, i, j, last_mat):
        len_list1, len_list2 = len(list1), len(list2)
        dp_flag = False
        mat_flag = False
        disperses_list = []
        first_mat = 0
        continue_num = 0
        while i < len_list1:
            if j < len_list2 and list1[i].index_name == '':
                if (list1[i].type == list2[j].type or ('MatMulV' in list1[i].type and 'MatMulV' in list2[j].type)):
                    mat_flag = True
                    if dp_flag:
                        disperses_list.append(i)
                        continue_num += 1
                        if continue_num > 5 or i >= len_list1 - 1:
                            dp_flag = False
                            continue_num = 0
                            list1 = self.attract_list(disperses_list, list1, i)
                            disperses_list = []
                    list1[i].index_name = list2[j].index_name
                    last_mat = (i, j)
                    j += 1
                else:
                    if mat_flag and first_mat == 0:
                        first_mat = i
                        disperses_list.append(i)
                    continue_num = 0
                    dp_flag = True
            elif dp_flag and len(disperses_list) > 0:
                while i < len_list1 and list1[i].index_name == '':
                    i += 1
                i = i - 1
                dp_flag = False
                continue_num = 0
                list1 = self.attract_list(disperses_list, list1, i)
                disperses_list = []
            i += 1
        return list1, i, j, last_mat, first_mat

    def attract_list(self, disperses_list, list1, i):
        index = 0
        len_dp = len(disperses_list)
        while i - index >= 0 and len_dp - index - 1 >= 0 and \
                (list1[i - index].type == list1[disperses_list[len_dp - index - 1]].type or \
                 ('MatMulV' in list1[i - index].type and 'MatMulV' in list1[disperses_list[len_dp - index - 1]].type)):
            temp = list1[disperses_list[len_dp - index - 1]].index_name
            list1[disperses_list[len_dp - index - 1]].index_name = ''
            list1[i - index].index_name = temp
            index += 1
        return list1

    def print_list(self):
        self.print_list_fw()
        self.print_list_bw()

    def print_list_fw(self):
        self._logger.debug("fw")
        for item in self.fw:
            self._logger.debug(f"name,{item.name}, type, {item.type}, index_name, {item.index_name}")

    def print_list_bw(self):
        self._logger.debug("bw")
        for item in self.bw:
            self._logger.debug(f"name,{item.name}, type, {item.type}, index_name, {item.index_name}")


class ChangeOperatorList:
    def __init__(self):
        super(ChangeOperatorList, self).__init__()
        self.list_2 = ProfileList()
        self.list_4 = ProfileList()


class DictShape(object):
    def __init__(self):
        self.name = ""
        self.type = ""
        self.accelerator_core = ""
        self.index_name = ""

    def change_profile_into_dictshape(self, item, index):
        self.name = item.name
        self.type = item.type
        self.accelerator_core = item.accelerator_core
        if index == -1:
            self.index_name = ""
        else:
            self.index_name = str(index) + str(item.type)


class OperatorLayerTime(object):
    def __init__(self):
        self.base_operator = self.Element()
        self.cp_exist = self.Element()
        self.cp_diff = self.Element()
        self.ep_exist = self.Element()
        self.ep_diff = self.Element()

    class Element:
        def __init__(self, fw=0.0, bw=0.0):
            self.fw = fw
            self.bw = bw


class DictModelShape(DictShape):
    def __init__(self):
        super(DictModelShape, self).__init__()
        self.model_w = 0.0
        self.model_b = 0.0
        self.shape_model_w = 0.0
        self.shape_model_b = 0.0


class DictCalShape(DictShape):
    def __init__(self):
        super(DictCalShape, self).__init__()
        self.input_cal = 0.0
        self.output_cal = 0.0


class OperatorList(ProfileList):
    def __init__(self):
        super(OperatorList, self).__init__()
        self.fw = []
        self.bw = []
        self.re = []
        self._logger = get_logger('operator_list')

    def print_list(self):
        self.print_list_fw()
        self.print_list_bw()
        self.print_list_re()

    def print_list_fw(self):
        self._logger.debug("fw")
        for item in self.fw:
            self._logger.debug(f"name,{item.name}, type, {item.type}, index_name, {item.index_name}")

    def print_list_bw(self):
        self._logger.debug("bw")
        for item in self.bw:
            self._logger.debug(f"name,{item.name}, type, {item.type}, index_name, {item.index_name}")

    def print_list_re(self):
        self._logger.debug("re")
        for item in self.re:
            self._logger.debug(f"name,{item.name}, type, {item.type}, index_name, {item.index_name}")