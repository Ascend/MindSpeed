import os
from functools import wraps
from logging import getLogger

import torch
import torch_npu

LOG = getLogger(__name__)


def extend_seed_all(seed=1234):
    """ set npu deterministic mode
    'HCCL_DETERMINISTIC' is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic, set it to 'False' to disable ops level deterministic.
    'CLOSE_MATMUL_K_SHIFT' is a switch of matmul K-axis shift, set it to '1' to close matmul K-axis shift, set it to '0' to enable matmul K-axis shift.
    'PYTHONHASHSEED' refers to python hash seed, use a string of non-negative integer to specify the seed.
    """
    os.environ['HCCL_DETERMINISTIC'] = 'True'
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def npu_deterministic_wrapper(fn):
    @wraps(fn)
    def wrapper(seed, *args, **kwargs):
        fn(seed, *args, **kwargs)
        extend_seed_all(seed)
        LOG.info("Deterministic computing is applied for npu.")
    return wrapper
