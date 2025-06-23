import torch
from mindspeed.auto_settings.utils.mem_utils import mem_b_to_mb


def __post_init__(self):
    self.world_size = self.nnodes * self.nproc_per_node
    self.target_world_size = self.target_nnodes * self.nproc_per_node
    self.search_world_size = self.target_world_size
    free_memory, total_memory = torch.cuda.mem_get_info(device=0)
    self.max_available_memory = total_memory / (1024 ** 3)
    self.memory_cap = mem_b_to_mb(total_memory)
    self.use_operator_model = False