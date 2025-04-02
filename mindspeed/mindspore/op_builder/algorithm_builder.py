# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from abc import abstractmethod

from mindspore.utils import _reuse_data_ptr
from mindspeed.op_builder.builder import MindSpeedOpBuilder



class AlgorithmOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "algorithm"
    reuse_data_ptr = _reuse_data_ptr

    def __init__(self):
        super(AlgorithmOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        """子类必须实现该方法"""
        pass

    def load(self, verbose=True):
        return self  # 新增的 load 方法
