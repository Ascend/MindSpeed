# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import abc

import numpy as np
from sklearn.linear_model import LinearRegression

from mindspeed.auto_tuning.utils.logger import get_logger


class LinearModel(metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.linear_model = LinearRegression()
        self.xs = []
        self.ys = []
        self.cfgs = []
        self.w = None
        self.b = None
        self.trained = False

    @abc.abstractmethod
    def predict(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def add_sample(self, *args):
        """args [x_in_domain, x_hccs, x_roce, y]
            x_in_domain就是不跨域的自变量
            x_hccs是跨域拆分后的hccs的自变量，
            x_roce 是跨域拆分后的roce的自变量
            y 因变量

        :param args:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def debug(self, model_name=""):
        raise NotImplementedError


class CommProtocolModel(LinearModel):
    def __init__(self, protocol_name=None):
        super().__init__()
        self.protocol_name = protocol_name
        self.logger = get_logger(protocol_name)

    def _handle_abnormal_samples(self):
        """特殊样本情况处理：
        1. 如果只有1个样本，添加[0, 0]
        2. 如果所有样本，x相同时去重添加原点，例如： [10, 10.1], [10, 9], [10, 9.9]---> [10, (10.1+9 + 9.9）/3]
         [10, 10.1], [10.1] [10.1], [10, 10.2]--> 不用去重
        """
        xs_set = set(np.array(self.xs).flatten())
        if len(xs_set) == 1:
            x_cal_list = [[0], self.xs[0]]
            y_cal_list = [0, sum(self.ys) / len(self.ys)]
            return x_cal_list, y_cal_list
        return self.xs, self.ys

    def add_sample(self, *args):
        x, *_, y, cfg = args
        self.xs.append([x])
        self.ys.append(y)
        self.cfgs.append(cfg)

    def debug(self, model_name=""):
        print("===============================================================================")
        tplt = "{0:<5}\t\t{1:<5}\t{2:<3}\t{3:<3}\t{4:<3}\t{5:<3}\t{6:<3}\t{7:<3}\t{8:<3}"
        self.logger.debug(f"-------samples of model {model_name} of {self.__class__.__name__}---")
        header = tplt.format("x", "y", "cfg_no", "tp", "cp", "dp", "ep", "pp", "vp", chr(12288))
        self.logger.debug(header)
        tplt = "{0:<5.2f}\t\t{1:<5.2f}\t{2:<3}\t{3:<3}\t{4:<3}\t{5:<3}\t{6:<3}\t{7:<3}\t{8:<3}"
        for sample_idx in range(len(self.xs)):
            cfg = self.cfgs[sample_idx]
            cur_row = (round(self.xs[sample_idx][0], 3), round(self.ys[sample_idx], 2), cfg.config_no,
                       cfg.tp, cfg.cp, cfg.dp, cfg.ep, cfg.pp, cfg.vp)
            self.logger.debug(tplt.format(*cur_row))

        self.logger.debug(f"------------------model parameters of model {model_name}------------")
        self.logger.debug(
            f"W: {getattr(self.linear_model, 'coef_', None)},  "
            f"b: {getattr(self.linear_model, 'intercept_', None)}"
        )
        print("===============================================================================")

    def fit(self):
        x_cal_list, y_cal_list = self._handle_abnormal_samples()
        x = np.array(x_cal_list).reshape(-1, 1)
        y = np.array(y_cal_list)
        if len(y) == 0:
            self.logger.warning(f"Empty samples for model: {self.protocol_name}")
            return

        self.linear_model.fit(x, y)

        self.w = self.linear_model.coef_[0]
        self.b = self.linear_model.intercept_

        self.trained = True

    def predict(self, *args):
        x = args[0]
        if not self.trained:
            raise AssertionError(f"{self.protocol_name} model should be trained before prediction")
        return self.linear_model.predict([[x]])[0]


class ROCEDomainModel(CommProtocolModel):
    def __init__(self, protocol_name="ROCE"):
        super().__init__(protocol_name)


class HCCSDomainModel(CommProtocolModel):
    def __init__(self, protocol_name="HCCS"):
        super().__init__(protocol_name)


class CrossDomainModel(LinearModel):
    def __init__(
        self, hccs_model: HCCSDomainModel, roce_model: ROCEDomainModel, protocol_name="Cross",
    ):
        super().__init__()
        self.protocol_name = protocol_name
        self.logger = get_logger(protocol_name)
        self.hccs_model = hccs_model
        self.roce_model = roce_model
        self.trained = True

    def add_sample(self, *args):
        _, hccs_x, roce_x, y, cfg = args
        self.xs.append([hccs_x, roce_x])
        self.ys.append(y)
        self.cfgs.append(cfg)

    def fit(self):
        # After ep_extend_tp is enabled, data in the roce domain needs to be split through the cross domain.
        if not self.roce_model.xs and self.hccs_model.xs:
            for index, corss_time in enumerate(self.ys):
                hccs_time = self.xs[index][0] * self.hccs_model.w + self.hccs_model.b
                roce_time = corss_time - hccs_time
                self.roce_model.xs.append([self.xs[index][1]])
                self.roce_model.ys.append(roce_time)
                self.roce_model.cfgs.append(self.cfgs[index])
            self.roce_model.fit()

    def predict(self, *args):
        self.trained = self.roce_model.trained and self.hccs_model.trained
        _, x1, x2, *_ = args
        if not self.trained:
            raise AssertionError(f"{self.protocol_name} model should be trained before prediction")
        y_roce_model = 0
        if self.roce_model:
            y_roce_model = self.roce_model.predict(x2)
        y_hccs_model = self.hccs_model.predict(x1)
        return y_hccs_model + y_roce_model

    def debug(self, model_name=""):
        tplt = "{0:<10}\t\t{1:<10}\t\t{2:<10}"
        self.logger.debug(f"  Samples of model {model_name} of CrossDomainModel")
        header = tplt.format(*("x1", "x2", "y", chr(12288)))
        self.logger.debug(header)
        for sample_idx in range(len(self.xs)):
            x1, x2 = self.xs[sample_idx]
            y = self.ys[sample_idx]
            cur_row = (round(x1, 3), round(x2, 3), round(y, 2))
            self.logger.debug(tplt.format(*cur_row))
