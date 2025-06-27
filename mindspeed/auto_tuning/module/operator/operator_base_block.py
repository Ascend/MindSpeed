import copy
from mindspeed.auto_tuning.module.operator.operator_elemental import OperatorList, ProfileList, DictShape


class Block(OperatorList):
    def __init__(self):
        super(Block, self).__init__()
        self.exist_cal_list = []

    @staticmethod
    def longest_common_subsequence(list1, list2):
        m, n = len(list1), len(list2)
        dp = [[] for _ in range(m + 1)]
        for index in range(m + 1):
            dp[index] = [[] for _ in range(n + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if list1[i - 1].type == list2[j - 1].type or \
     ('MatMulV' in list1[i - 1].type and 'MatMulV' in list2[j - 1].type):
                    dp[i][j] = dp[i - 1][j - 1].copy()
                    dp[i][j].append(list1[i - 1])
                else:
                    if len(dp[i - 1][j]) > len(dp[i][j - 1]):
                        dp[i][j] = dp[i - 1][j].copy()
                    else:
                        dp[i][j] = dp[i][j - 1].copy()
        return dp[m][n]

    @staticmethod
    def change_profilelist_into_dictshapelist_withindex(change_profile_list, change_operator_list):
        for (index, item) in enumerate(change_profile_list.fw):
            dict_shape_fw = DictShape()
            dict_shape_fw.change_profile_into_dictshape(item, index)
            change_operator_list.fw.append(dict_shape_fw)
        for (index, item) in enumerate(change_profile_list.bw):
            dict_shape_bw = DictShape()
            dict_shape_bw.change_profile_into_dictshape(item, index)
            change_operator_list.bw.append(dict_shape_bw)

    @staticmethod
    def change_profilelist_into_dictshapelist(change_profile_list, change_operator_list):
        for (index, item) in enumerate(change_profile_list.fw):
            dict_shape_fw = DictShape()
            dict_shape_fw.change_profile_into_dictshape(item, -1)
            change_operator_list.fw.append(dict_shape_fw)
        for (index, item) in enumerate(change_profile_list.bw):
            dict_shape_bw = DictShape()
            dict_shape_bw.change_profile_into_dictshape(item, -1)
            change_operator_list.bw.append(dict_shape_bw)


class BaseBlock(Block):
    def __init__(self):
        super(BaseBlock, self).__init__()

    def get_block(self, data_list):
        profile_list = self.get_profile(data_list)
        self.change_profilelist_into_dictshapelist_withindex(profile_list, self)

    def get_profile(self, data_list):
        profile_list = ProfileList()
        for origin_profile_data in data_list:
            fw = origin_profile_data.profile_list.fw
            bw = origin_profile_data.profile_list.bw
            if len(profile_list.fw) == 0:
                profile_list.fw = copy.deepcopy(fw)
                profile_list.bw = copy.deepcopy(bw)
            else:
                profile_list.fw = self.longest_common_subsequence(profile_list.fw, fw)
                profile_list.bw = self.longest_common_subsequence(profile_list.bw, bw)
        return profile_list