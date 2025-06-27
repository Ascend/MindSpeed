# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional
from argparse import Namespace
from dataclasses import dataclass
import logging
import os

from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_executor import MindSpeedExecutor
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_runner import MindSpeedRunner
from mindspeed.auto_tuning.utils.logger import init_logger, get_logger
from mindspeed.auto_tuning.utils.singleton import Singleton


@dataclass
class MindSpeedSettingsPKL:
    FILENAME = "auto_tuning_settings.pkl"

    model_cfg: ModelConfig = None  # type: ignore
    devices_per_node: int = None  # type: ignore
    nnodes: int = None  # type: ignore
    node_rank: int = None  # type: ignore
    device_type: str = None  # type: ignore
    wait_timeout: int = None  # type: ignore
    memory_cap: float = None  # type: ignore
    driver_version: str = None  # type: ignore
    cann_version: str = None  # type: ignore


class MindSpeedSettings(metaclass=Singleton):
    DISABLE_CP = False

    def __init__(self):
        self.work_dir: str = None  # type: ignore
        self.search_world_size: int = None  # type: ignore
        self.log_level: int = None  # type: ignore
        self.waas_ip_addr: Optional[str] = None
        self.waas_ip_port: Optional[int] = None
        self.executor: MindSpeedExecutor = None  # type: ignore

        self.model_cfg: ModelConfig = None  # type: ignore
        self.devices_per_node: int = None  # type: ignore
        self.nnodes: int = None  # type: ignore
        self.node_rank: int = None  # type: ignore
        self.device_type: str = None  # type: ignore
        self.wait_timeout: int = None  # type: ignore
        self.memory_cap: float = None  # type: ignore
        self.driver_version: str = None  # type: ignore
        self.cann_version: str = None  # type: ignore

        self._logger = get_logger("settings")

    @property
    def waas_enabled(self) -> bool:
        return self.waas_ip_addr is not None and \
            self.waas_ip_port is not None

    @property
    def profiling_world_size(self) -> int:
        return self.nnodes * self.devices_per_node

    def init_settings(self, args: Namespace):
        self.work_dir = os.path.realpath(args.auto_tuning_work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.search_world_size = args.auto_tuning_ranks
        if self.search_world_size < 16:
            raise AssertionError("Auto-tuning searching space should be >= 16.")

        self.log_level = logging.INFO
        if args.auto_tuning_log_level == "warning":
            self.log_level = logging.WARNING
        elif args.auto_tuning_log_level == "debug":
            self.log_level = logging.DEBUG

        init_logger(self.log_level)

        if args.auto_tuning_database_address:
            self.waas_ip_addr = args.auto_tuning_database_address

        if args.auto_tuning_database_port:
            self.waas_ip_port = args.auto_tuning_database_port

        self.executor = MindSpeedExecutor(MindSpeedRunner(args))

        self._logger.info("Auto-tuning is working @ %s.", self.work_dir)
        self._logger.info("Auto-tuning is searching for %d cards.", self.search_world_size)

    def load_settings_from_pkl(self, pkl: MindSpeedSettingsPKL):
        for k, v in vars(pkl).items():
            setattr(self, k, v)
