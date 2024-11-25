import copy
from mindspeed.auto_tuning.module.operator.operator_base_block import Block
from mindspeed.auto_tuning.module.operator.operator_elemental import (OperatorList, ChangeList,
                                                                      ChangeOperatorList)


class ChangeBlock(Block):
    def __init__(self):
        super(ChangeBlock, self).__init__()
        self.diff_list = OperatorList()
        self.diff_cal_list = []

    @staticmethod
    def get_operator_longest_common_subsequence(list1, list2):
        m, n = len(list1), len(list2)
        dp = [[] for _ in range(m + 1)]
        for index in range(m + 1):
            dp[index] = [[] for _ in range(n + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if list1[i - 1].type == list2[j - 1].type:
                    dp[i][j] = dp[i - 1][j - 1].copy()
                    dp[i][j].append(list1[i - 1])
                else:
                    if len(dp[i - 1][j]) > len(dp[i][j - 1]):
                        dp[i][j] = dp[i - 1][j].copy()
                    else:
                        dp[i][j] = dp[i][j - 1].copy()
        return dp[m][n]

    def get_profile(self, origin_profile_data_list):
        change_profile_list = ChangeList()
        change_operator_list = ChangeOperatorList()
        for origin_profile_data in origin_profile_data_list:
            fw = origin_profile_data.operator_list.fw
            bw = origin_profile_data.operator_list.bw
            cp = origin_profile_data.config_info.cp
            dp = origin_profile_data.config_info.dp
            pp = origin_profile_data.config_info.pp
            ep = origin_profile_data.config_info.ep
            num_experts = origin_profile_data.config_info.num_experts
            self.get_profile_info(cp, change_profile_list, fw, bw)
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

    def get_exist_block(self, change_operator_list, base_block, index_id):
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

    def comp_with_get_diff_list(self, list1, list2, index_id):
        return

        #

    def reset_index_name(self, list1, list2):
        m, n = len(list1), len(list2)
        i, j = 0, 0
        index = 0
        last_mat = (0, 0)
        first_mat = 0
        while 1:
            list1, i, j, last_mat, first_mat = self.reset_index_name_single(list1, list2, i, j, last_mat)
            if j < n - 1 and index < 3:
                # Skip a base operator
                index += 1
                i = last_mat[0] + 1
                j += 1
            else:
                break
        if first_mat == 0:
            first_mat = last_mat[0] + 1
        return list1, first_mat

    def reset_index_name_single(self, list1, list2, i, j, last_mat):
        m, n = len(list1), len(list2)
        dp_flag = False
        mat_flag = False
        disperses_list = []
        first_mat = 0
        continue_num = 0
        while i < m:
            if j < n and list1[i].index_name == '':
                if list1[i].type == list2[j].type:
                    mat_flag = True
                    if dp_flag:
                        disperses_list.append(i)
                        continue_num += 1
                        if continue_num > 5 or i >= m - 1:
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
                while i < m and list1[i].index_name == '':
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
        while (i - index >= 0 and len_dp - index - 1 >= 0 and
               list1[i - index].type == list1[disperses_list[len_dp - index - 1]].type):
            temp = list1[disperses_list[len_dp - index - 1]].index_name
            list1[disperses_list[len_dp - index - 1]].index_name = ''
            list1[i - index].index_name = temp
            index += 1
        return list1
