import abc
from mindspeed.auto_tuning.module.operator.operator_shape_cal import linear_regression
from mindspeed.auto_tuning.utils.logger import get_logger


class CommunicationList():
    def __init__(self):
        self.roce_x_list = []
        self.roce_time_list = []
        self.hccs_x_list = []
        self.hccs_time_list = []
        self.cross_x_list = []
        self.cross_y_list = []
        self.cross_time_list = []

        self.roce_w = 0
        self.roce_b = 0
        self.hccs_w = 0
        self.hccs_b = 0
        self.cross_list = (0, 0)

    def append_roce(self, iv_list, time):
        self.roce_x_list.append([iv_list[0]])
        self.roce_time_list.append([time])
        self.hccs_x_list.append([None])
        self.hccs_time_list.append([None])
        self.cross_x_list.append([None])
        self.cross_y_list.append([None])
        self.cross_time_list.append([None])

    def append_hccs(self, iv_list, time):
        self.roce_x_list.append([None])
        self.roce_time_list.append([None])
        self.hccs_x_list.append([iv_list[0]])
        self.hccs_time_list.append([time])
        self.cross_x_list.append([None])
        self.cross_y_list.append([None])
        self.cross_time_list.append([None])

    def append_cross(self, iv_list, time):
        self.roce_x_list.append([None])
        self.roce_time_list.append([None])
        self.hccs_x_list.append([None])
        self.hccs_time_list.append([None])
        self.cross_x_list.append([iv_list[1]])
        self.cross_y_list.append([iv_list[2]])
        self.cross_time_list.append([time])

    def cal_roce(self, iv_list):
        return self.roce_w * iv_list[0] + self.roce_b
    
    def cal_hccs(self, iv_list):
        return self.hccs_w * iv_list[0] + self.hccs_b
    
    def cal_cross(self, iv_list):
        return self.hccs_w * iv_list[1] + self.hccs_b + self.roce_w * iv_list[2] + self.roce_b

    def modeling(self):
        lists = (
            self.hccs_x_list,
            self.hccs_time_list,
            self.roce_x_list,
            self.roce_time_list,
            self.cross_x_list,
            self.cross_y_list,
            self.cross_time_list
        )
        (hccs_x_cal, hccs_time_cal), (roce_x_cal, roce_time_cal), (cross_x_cal,
                                                                   cross_time_cal) = self.get_hccs_roce_list(lists)
        if roce_x_cal:
            self.roce_w, self.roce_b = self.linear_x_y(roce_x_cal, roce_time_cal)
        if hccs_x_cal:
            self.hccs_w, self.hccs_b = self.linear_x_y(hccs_x_cal, hccs_time_cal)

    def get_hccs_roce_list(self, lists):
        hccs_x_list = []
        hccs_y_list = []
        roce_x_list = []
        roce_y_list = []
        cross_x_list = []
        cross_y_list = []
        for i, x_index in enumerate(lists[0]):
            if lists[0][i] != [None]:
                hccs_x_list.append(lists[0][i])
                hccs_y_list.append(lists[1][i])
            elif lists[2][i] != [None]:
                roce_x_list.append(lists[2][i])
                roce_y_list.append(lists[3][i])
            else:
                cross_x_list.append([lists[4][i][0] / lists[5][i][0]])
                cross_y_list.append([lists[6][i][0] / lists[5][i][0]])
        hccs_lists = (hccs_x_list, hccs_y_list)
        roce_lists = (roce_x_list, roce_y_list)
        cross_lists = (cross_x_list, cross_y_list)
        re_hccs_lists = self.add_origin_whith_single_point(hccs_lists)
        re_roce_lists = self.add_origin_whith_single_point(roce_lists)
        re_cross_lists = self.add_origin_whith_single_point(cross_lists)

        return re_hccs_lists, re_roce_lists, re_cross_lists

    @classmethod
    def add_origin_whith_single_point(cls, lists):
        last = None
        for item in lists[0]:
            if last:
                if item != last:
                    last = None
                    break
            else:
                last = item
        listres = lists
        if last:
            listres = [[], []]
            listres[0].append(lists[0][0])
            listres[1].append(lists[1][0])
        if len(listres[0]) == 1:
            listres[0].append([0])
            listres[1].append([0])
        return listres

    @classmethod
    def linear_x_y(cls, list1, list2):
        w, b = 0, 0
        if len(list1) > 0:
            w, b = linear_regression(list1, list2) if list1 else (0, 0)
        return w, b


class CommunicationModel:
    def __init__(self, hccs_dev_num):
        self.comm = CommunicationList()
        self.main_domain = Domain(hccs_dev_num)
        self.hccs_dev_num = hccs_dev_num
        self.logger = get_logger("Communication")

    @abc.abstractmethod
    def get_communication_info_from_profile(self, hcom_info_tage_id):
        pass

    @abc.abstractmethod
    def get_comm_info_list(self, profile_info):
        pass

    @abc.abstractmethod
    def modeling(self):
        pass

    @abc.abstractmethod
    def print_modeling(self):
        pass


class Domain:
    def __init__(self, hccs_dev_num):
        self.max_domain = 0
        self.min_domain = 0
        self.roce_comm_exist = False
        self.hccs_comm_exist = False
        self.cross_comm_exist = False
        self.hccs_dev_num = hccs_dev_num

    def is_hccs_domain(self):
        return self.max_domain <= self.hccs_dev_num
    
    def is_cross_domain(self):
        return self.min_domain < self.hccs_dev_num < self.max_domain
    
    def is_roce_domain(self):
        return not (self.is_hccs_domain() or self.is_hccs_domain())
    
    def append_method_for_domain(self):
        if self.is_hccs_domain():
            self.hccs_comm_exist = True
            return "append_hccs"
        if self.is_cross_domain():
            self.cross_comm_exist = True
            return "append_cross"
        self.roce_comm_exist = True
        return "append_roce"
    
    def append_time_in_domain(self, communication_list, iv_list, time):
        method_for_domain = self.append_method_for_domain()
        append_domain = getattr(communication_list, method_for_domain)
        append_domain(iv_list, time)

    def cal_method_for_domain(self):
        if self.is_hccs_domain():
            return "cal_hccs"
        if self.is_cross_domain():
            return "cal_cross"
        return "cal_roce"
    
    def cal_time_in_domain(self, communication_list, iv_list):
        method_for_domain = self.cal_method_for_domain()
        cal_domain = getattr(communication_list, method_for_domain)
        return cal_domain(iv_list)
