# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import copy
from mindspeed.auto_tuning.module.operator.operator_base_block import Block
from mindspeed.auto_tuning.module.operator.operator_elemental import (DictShape, OperatorList)


class ChangeBlock(Block):
    def __init__(self):
        super(ChangeBlock, self).__init__()
        self.diff_list = OperatorList()
        self.diff_cal_list = []

    def get_profile_info(self, change_num, change_profile_list, fw, bw):
        if change_num == 2:
            if len(change_profile_list.list_2.fw) == 0:
                change_profile_list.list_2.fw = copy.deepcopy(fw)
                change_profile_list.list_2.bw = copy.deepcopy(bw)
            else:
                change_profile_list.list_2.fw = self.longest_common_subsequence(change_profile_list.list_2.fw, fw)
                change_profile_list.list_2.bw = self.longest_common_subsequence(change_profile_list.list_2.bw, bw)
        if change_num == 4:
            if len(change_profile_list.list_4.fw) == 0:
                change_profile_list.list_4.fw = copy.deepcopy(fw)
                change_profile_list.list_4.bw = copy.deepcopy(bw)
            else:
                change_profile_list.list_4.fw = self.longest_common_subsequence(change_profile_list.list_4.fw, fw)
                change_profile_list.list_4.bw = self.longest_common_subsequence(change_profile_list.list_4.bw, bw)
        if len(change_profile_list.list_2.fw) * len(change_profile_list.list_4.fw) > 0:
            change_profile_list.list_2.fw = self.longest_common_subsequence(change_profile_list.list_2.fw,
                                                                            change_profile_list.list_4.fw)
            change_profile_list.list_2.bw = self.longest_common_subsequence(change_profile_list.list_2.bw,
                                                                            change_profile_list.list_4.bw)
        return

    def get_change_operator(self, change_profile_list, change_operator_list):
        self.change_profilelist_into_dictshapelist(change_profile_list.list_2, change_operator_list.list_2)
        self.change_profilelist_into_dictshapelist(change_profile_list.list_4, change_operator_list.list_4)

    def get_exist_block(self, change_operator_list, base_block, index_id):
        self.fw = self.comp_with_get_diff_list(change_operator_list.list_2.fw, base_block.fw, index_id)
        self.bw = self.comp_with_get_diff_list(change_operator_list.list_2.bw, base_block.bw, index_id + 500)
        # 重计算
        if len(self.bw) > len(self.fw):
            self.re, self.bw = self.get_re_block(self.bw, self.fw)
        return

    # 计算重计算列表 1是反向 2是正向
    def get_re_block(self, list1, list2):
        m, n = len(list1), len(list2)
        list_re = []
        list_bw = []
        i, j = 0, 0
        while i < m:
            if j < n and self.matchv2_v3_operator(list1[i], list2[j]):
                list_re.append(list1[i])
                i += 1
                j += 1
            else:
                list_bw.append(list1[i])
                i += 1
        return list_re, list_bw

    # 把列表1与列表2对齐
    def comp_with_get_diff_list(self, list1, list2, index_id):
        # 先对齐
        list1, _ = self.reset_index_name(list1, list2)
        diff_info = []
        diff_index = index_id
        for item in list1:
            if item.index_name == '':
                dict_shape = DictShape()
                if diff_index != -1:
                    item.index_name = str(diff_index) + item.type
                    diff_index += 1
                else:
                    item.index_name = ''
                dict_shape.name = item.name
                dict_shape.type = item.type
                dict_shape.accelerator_core = item.accelerator_core
                dict_shape.index_name = item.index_name
                diff_info.append(dict_shape)
        return diff_info