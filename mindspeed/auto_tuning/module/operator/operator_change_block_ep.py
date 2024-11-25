import copy
from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.module.operator.operator_change_block import ChangeBlock
from mindspeed.auto_tuning.module.operator.operator_elemental import (DictShape, ChangeList,
                                                                      ChangeOperatorList)


class EpBlock(ChangeBlock):
    def __init__(self):
        super(EpBlock, self).__init__()
        self._logger = get_logger('ep_block')

    def get_block(self, origin_profile_data_list, base_block):
        change_operator_list = self.get_profile(origin_profile_data_list)
        index_id = 2000
        self.get_exist_block(change_operator_list, base_block, index_id)
        self.get_diff_block(change_operator_list, index_id)
        self.diff_list.bw.pop()

    def get_profile(self, origin_profile_data_list):
        change_profile_list = ChangeList()
        change_operator_list = ChangeOperatorList()
        for origin_profile_data in origin_profile_data_list.data_list:
            fw = origin_profile_data.profile_list.fw
            bw = origin_profile_data.profile_list.bw
            ep = origin_profile_data.config_info.ep
            num_experts = origin_profile_data.config_info.num_experts
            self.get_profile_info(num_experts / ep, change_profile_list, fw, bw)
        self.get_change_operator(change_profile_list, change_operator_list)
        return change_operator_list

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

    # Compare the longest subsequence of 1 and 2, return the values of 1.

    def get_exist_block(self, change_operator_list, base_block, index_id):
        self.fw = self.comp_with_get_diff_list(change_operator_list.list_2.fw, base_block.fw, index_id)
        self.bw = self.comp_with_get_diff_list(change_operator_list.list_2.bw, base_block.bw, index_id + 500)
        # recompute
        if len(self.bw) > len(self.fw):
            self.re, self.bw = self.get_re_block(self.bw, self.fw)
        return

    def get_diff_block(self, change_operator_list, index_id):
        if not change_operator_list.list_2.fw:
            self._logger.warning("warning:缺少了并行配置为 EP=2 的数据，从而无法得到EPdiff")
            return
        self.diff_list.fw = self.comp_with_get_diff_list(change_operator_list.list_4.fw, change_operator_list.list_2.fw,
                                                         -1)
        self.diff_list.bw = self.comp_with_get_diff_list(change_operator_list.list_4.bw, change_operator_list.list_2.bw,
                                                         -1)
        # recompute
        if len(self.diff_list.bw) > len(self.diff_list.fw):
            self.diff_list.re, self.diff_list.bw = self.get_re_block(self.diff_list.bw, self.diff_list.fw)
            self.diff_list.re = self.comp_with_get_diff_list(self.diff_list.re, self.re, -1)
        self.diff_list.fw = self.comp_with_get_diff_list(self.diff_list.fw, self.fw, -1)
        self.diff_list.bw = self.comp_with_get_diff_list(self.diff_list.bw, self.bw, -1)
        return

    # calculate the recompute list, 1 for forward, 2 for backward
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

    # Align list1 with list2
    def comp_with_get_diff_list(self, list1, list2, index_id):
        # Align first
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

    def get_exist_base_ep(self):
        self.fw = self.get_diff_list_without_index(self.fw, self.diff_list.fw)
        self.re = self.get_diff_list_without_index(self.re, self.diff_list.re)
        self.bw = self.get_diff_list_without_index(self.bw, self.diff_list.bw)

        # Subtract the subsequence of 2 from 1.

    def get_diff_list_without_index(self, list1, list2):
        list_comm = self.get_operator_longest_common_subsequence(list1, list2)
        m, n = len(list1), len(list_comm)
        flag = 0
        max_miss = 3
        diff_list = []
        i, j = 0, 0
        while i < m and j < n:
            if list1[i].type == list_comm[j].type:
                i += 1
                j += 1
            else:
                diff_list.append(list1[i])
                i += 1
        if i < m:
            diff_list.append(list1[i])
            i += 1
        return diff_list

    def reset_index_diff_ep(self, list1, list2, ep_diff_num):
        m, n = len(list1), len(list2)
        i, j = 0, 0
        index = 0
        last_mat, this_mat = (0, 0), (-1, 0)
        while 1:
            # Restart a round
            if this_mat[0] + n > m or this_mat == last_mat or ep_diff_num <= 0:
                break
            last_mat = this_mat
            list1, i, j, this_mat = self.reset_index_name_single_ep(list1, list2, i, j, last_mat)
            ep_diff_num -= 1
            if j < n - 1 and index < 3:
                # skip one base operator
                index += 1
                i = this_mat[0] + 1
                j += 1
            else:
                j = 0
                i = this_mat[0] + 1
        return list1

    def reset_index_name_single_ep(self, list1, list2, i, j, start_mat):
        m, n = len(list1), len(list2)
        dp_flag = True
        disperses_list = []
        continue_num = 0
        last_mat = start_mat
        while i < m:
            if j < n and list1[i].index_name == '':
                if list1[i].type == list2[j].type:
                    if j == 0 and start_mat[0] > 0 and i - start_mat[0] > 3:
                        break
                    if dp_flag:
                        disperses_list.append(i)
                        continue_num += 1
                        if continue_num > 5 or j + 1 == n:
                            dp_flag = False
                            continue_num = 0
                            list1 = self.attract_list(disperses_list, list1, i)
                            disperses_list = []
                    list1[i].index_name = list2[j].index_name
                    last_mat = (i, j)
                    j += 1
                else:
                    continue_num = 0
                    dp_flag = True
            i += 1
        return list1, i, j, last_mat
