import torch
from mindspeed.auto_settings.auto_settings import AutoSettings


def ms_init(self, args):
    self._init_configs(args)
    self._init_global_group()
    self._init_hardware(args)


def _init_global_group(self):
    torch.distributed.init_process_group(backend="mccl")
