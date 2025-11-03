import os
import logging
from dataclasses import dataclass, field

import torch
import torch.cuda as cuda
import torch.distributed as dist

from mindspeed.auto_settings.utils.logger import init_logger, get_logger
from mindspeed.auto_settings.utils.mem_utils import mem_b_to_mb
from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.mindspeed_adaptor.mindspeed_executor import MindSpeedExecutor
from mindspeed.auto_settings.mindspeed_adaptor.mindspeed_runner import MindSpeedRunner


@dataclass
class SystemConfig:
    DISABLE_CP = False
    nnodes: int

    nproc_per_node: int

    node_rank: int

    master_addr: str

    master_port: int

    target_nnodes: int

    work_dir: str

    log_level: int

    waas_enabled: bool

    # 支持搜索的并行维度个数
    search_dimensions: int

    # 实际用于搜索的小集群规模
    world_size: int = field(init=0)

    # 需要搜索的大集群规模
    target_world_size: int = field(init=0)
    auto_settings_ranks: int

    max_available_memory: float = field(init=0.)

    use_operator_model: bool = field(init=False)

    model_config: ModelConfig = None  # type: ignore

    device_type: str = None
    wait_timeout: int = None
    memory_cap: float = None
    driver_version: str = ""
    cann_version: str = ""
    search_world_size: int = field(init=0)
    executor: MindSpeedExecutor = None  # type: ignore
    gloo_group: dist.ProcessGroup = None  # type: ignore

    dist_train: bool = None  # type: ignore
    mm_model: str = None # type: ignore
    mm_data: str = None # type: ignore
    model_name: list = field(default_factory=list) # type: ignore

    _logger = get_logger("settings")

    def __post_init__(self):
        self.world_size = self.nnodes * self.nproc_per_node
        self.target_world_size = self.target_nnodes * self.nproc_per_node
        self.search_world_size = self.auto_settings_ranks
        self.max_available_memory = torch.npu.get_device_properties(0).total_memory / (1024 ** 3)
        self.memory_cap = mem_b_to_mb(torch.npu.get_device_properties(0).total_memory)
        self.use_operator_model = False

    @property
    def cache_path(self):
        work_dir = self.work_dir
        if not self.work_dir.endswith(os.sep):
            work_dir += os.sep

        try:
            os.makedirs(work_dir, exist_ok=True)
        except BaseException:
            work_dir = os.getcwd()

        return work_dir

    def load_settings(self, data_class):
        for k, v in vars(data_class).items():
            setattr(self, k, v)


_SYSTEM_CONFIG: SystemConfig = None


def set_system_config(args):
    global _SYSTEM_CONFIG
    if _SYSTEM_CONFIG is not None:
        raise AssertionError('SYSTEM_CONFIG has been initialized')

    log_level = logging.INFO
    if args.auto_settings_log_level == "warning":
        log_level = logging.WARNING
    elif args.auto_settings_log_level == "debug":
        log_level = logging.DEBUG

    init_logger(log_level)
    sys_config = SystemConfig(
        nnodes=int(args.nnodes),
        nproc_per_node=int(args.nproc_per_node),
        node_rank=int(args.node_rank),
        master_addr=args.master_addr,
        master_port=int(args.master_port),
        target_nnodes=int(args.target_nnodes),
        work_dir=args.auto_settings_work_dir,
        log_level=log_level,
        search_dimensions=8,
        waas_enabled=False,
        auto_settings_ranks=args.auto_settings_ranks

    )
    
    if hasattr(args, "mm_data") and args.mm_data:
        sys_config.mm_data = args.mm_data
    
    if hasattr(args, "mm_model") and args.mm_model:
        sys_config.mm_model = args.mm_model
        if args.dist_train:
            sys_config.dist_train = True
            sys_config.model_name = ["gpt", "vit"]
            sys_config.mm_model = args.mm_model
        sys_config.executor = MindSpeedExecutor(runner=MindSpeedRunner(args))
    _SYSTEM_CONFIG = sys_config


def get_system_config():
    global _SYSTEM_CONFIG
    if _SYSTEM_CONFIG is None:
        raise AssertionError('SYSTEM_CONFIG is not initialized')
    return _SYSTEM_CONFIG
