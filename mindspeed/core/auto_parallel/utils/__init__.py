# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
import sys
import logging
from .data_type import TaskType
from .model_config import get_model_config
from .system_config import get_system_config


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(log_level=logging.INFO):
    logger_ob = logging.getLogger('flex-parallel')
    if not any(isinstance(h, FlushStreamHandler) for h in logger_ob.handlers):
        logger_ob.setLevel(log_level)
        logger_ob.propagate = False
        # 日志输出到控制台
        log_format = logging.Formatter('%(levelname)s:%(message)s')
        console_handler = FlushStreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        logger_ob.addHandler(console_handler)
    return logger_ob


logger = get_logger(logging.INFO)