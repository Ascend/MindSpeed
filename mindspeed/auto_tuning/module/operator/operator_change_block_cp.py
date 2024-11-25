from mindspeed.auto_tuning.module.operator.operator_change_block import ChangeBlock
from mindspeed.auto_tuning.module.operator.operator_elemental import (DictShape, ChangeList,
                                                                      ChangeOperatorList)


class CpBlock(ChangeBlock):
    def __init__(self):
        super(CpBlock, self).__init__()

    def get_block(self, origin_profile_data_list, base_block):
        change_operator_list = self.get_profile(origin_profile_data_list)
        index_id = 1000
        self.get_exist_block(change_operator_list, base_block, index_id)
        self.get_diff_block(change_operator_list, -1)

    def get_profile(self, origin_profile_data_list):
        change_profile_list = ChangeList()
        change_operator_list = ChangeOperatorList()
        for origin_profile_data in origin_profile_data_list.data_list:
            fw = origin_profile_data.profile_list.fw
            bw = origin_profile_data.profile_list.bw
            cp = origin_profile_data.config_info.cp
            self.get_profile_info(cp, change_profile_list, fw, bw)
        self.get_change_operator(change_profile_list, change_operator_list)
        return change_operator_list

    def get_change_operator(self, change_profile_list, change_operator_list):
        self.change_profilelist_into_dictshapelist(change_profile_list.list_2, change_operator_list.list_2)
        self.change_profilelist_into_dictshapelist(change_profile_list.list_4, change_operator_list.list_4)

    def get_exist_block(self, change_operator_list, base_block, index_id):
        self.fw = self.comp_with_get_diff_list(change_operator_list.list_2.fw, base_block.fw, index_id)
        self.bw = self.comp_with_get_diff_list(change_operator_list.list_2.bw, base_block.bw, index_id + 500)
        # recompute
        if len(self.bw) > len(self.fw):
            self.re, self.bw = self.get_re_block(self.bw, self.fw)

    def get_diff_block(self, change_operator_list, index_id):
        self.diff_list.fw = self.comp_with_get_diff_list(change_operator_list.list_4.fw, change_operator_list.list_2.fw,
                                                         -1)
        self.diff_list.bw = self.comp_with_get_diff_list(change_operator_list.list_4.bw, change_operator_list.list_2.bw,
                                                         index_id)
        self.diff_list.fw = self.get_operator_longest_common_subsequence(self.fw, self.diff_list.fw)
        self.diff_list.re = self.get_operator_longest_common_subsequence(self.re, self.diff_list.bw)
        self.diff_list.bw = self.get_operator_longest_common_subsequence(self.bw, self.diff_list.bw)

    def get_re_block(self, list1, list2):
        m, n = len(list1), len(list2)
        list_re = []
        list_bw = []
        i, j = 0, 0
        while i < m:
            if j < n and list1[i].type == list2[j].type:
                list_re.append(list1[i])
                i += 1
                j += 1
            else:
                list_bw.append(list1[i])
                i += 1
        return list_re, list_bw

    def comp_with_get_diff_list(self, list1, list2, index_id):
        # Align first.
        list1, first_mat = self.reset_index_name(list1, list2)
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

    def reset_index_diff_cp(self, list1, list2, diff_flag, cp_num):
        m, n = len(list1), len(list2)
        if n < 2 or m < 2:
            return list1
        i, j = diff_flag - 1, n
        index = 0
        last_mat = (diff_flag - 1, n)
        temp = -1, -1
        while j >= n - 2 and last_mat[0] + n < m and last_mat != temp:
            cp_num -= 1
            if cp_num <= 0:
                break
            # Ensure that the entire process has been gone through once.
            # Ensure that there is enough remaining space for one round of re-matching.
            j = 0
            i = last_mat[0] + 1
            index = 0
            temp = last_mat
            # Restart a round of matching.
            list1, list2, i, j, last_mat = self.restart_mat(list1, list2, i, j, last_mat)
        return list1

    @staticmethod
    def restart_mat(list1, list2, i, j, last_mat):
        m, n = len(list1), len(list2)
        flag = 0
        max_miss = 3
        while i < m and j < n:
            if j < n and list1[i].index_name == '' and list1[i].type == list2[j].type:
                list1[i].index_name = list2[j].index_name
                last_mat = (i, j)
                i += 1
                j += 1
            else:
                if i + 1 < m and list1[i + 1].index_name == '' and list1[i + 1].type == list2[j].type:
                    i += 1
                elif j + 1 < n and list1[i].index_name == '' and list1[i].type == list2[j + 1].type:
                    j += 1
                else:
                    i += 1
                    j += 1
                max_miss = max_miss - 1
            if max_miss <= 0:
                return list1, list2, i, j, (0, 0)
        return list1, list2, i, j, last_mat
