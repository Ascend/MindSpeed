from mindspeed.auto_tuning.utils.logger import get_logger


class ProfileList(object):
    def __init__(self):
        self.fw = []
        self.bw = []
        self._logger = get_logger('ProfileList')

        # test 1

    def reset_index_name(self, list1, list2):
        M, N = len(list1), len(list2)
        i, j = 0, 0
        index = 0
        last_mat = (0, 0)
        first_mat = 0
        while 1:
            list1, i, j, last_mat, first_mat = self.reset_index_name_single(list1, list2, i, j, last_mat)
            if j < N - 1 and index < 3:
                # skip a base operator
                index += 1
                i = last_mat[0] + 1
                j += 1
            else:
                break
        if first_mat == 0:
            first_mat = last_mat[0] + 1
        return list1, first_mat

    def reset_index_name_single(self, list1, list2, i, j, last_mat):
        M, N = len(list1), len(list2)
        dp_flag = False
        mat_flag = False
        disperses_list = []
        first_mat = 0
        continue_num = 0
        while i < M:
            if j < N and list1[i].index_name == '':
                if list1[i].type == list2[j].type:
                    mat_flag = True
                    if dp_flag:
                        disperses_list.append(i)
                        continue_num += 1
                        if continue_num > 5 or i >= M - 1:
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
                while i < M and list1[i].index_name == '':
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
        while i - index >= 0 and len_dp - index - 1 >= 0 and list1[i - index].type == list1[
                disperses_list[len_dp - index - 1]].type:
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
            self._logger.debug("name", item.name, "type", item.type, "index_name", item.index_name)

    def print_list_bw(self):
        self._logger.debug("bw")
        for item in self.bw:
            self._logger.debug("name", item.name, "type", item.type, "index_name", item.index_name)


class ChangeList:
    def __init__(self):
        super(ChangeList, self).__init__()
        self.list_2 = ProfileList()
        self.list_4 = ProfileList()


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
            self._logger.debug("name", item.name, "type", item.type, "index_name", item.index_name)

    def print_list_bw(self):
        self._logger.debug("bw")
        for item in self.bw:
            self._logger.debug("name", item.name, "type", item.type, "index_name", item.index_name)

    def print_list_re(self):
        self._logger.debug("re")
        for item in self.re:
            self._logger.debug("name", item.name, "type", item.type, "index_name", item.index_name)


class OperatorDetailList(OperatorList):
    def __init__(self):
        super(OperatorDetailList, self).__init__()
        self.fw = []
        self.bw = []
        self.re = []
        self.all = []
