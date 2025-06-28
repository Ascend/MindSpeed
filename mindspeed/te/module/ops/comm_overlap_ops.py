from abc import ABC, abstractmethod


class CommOverlapConfig:

    @classmethod
    def get_tp_size(cls):
        if cls.tp_size is None:
            from megatron.core.parallel_state import get_tensor_model_parallel_world_size
            cls.tp_size = get_tensor_model_parallel_world_size()
        return cls.tp_size

    @classmethod
    def init_tp_size(cls, tp_size):
        cls.tp_size = tp_size

    @classmethod
    def get_tp_group(cls):
        if cls.tp_group is None:
            from megatron.core.parallel_state import get_tensor_model_parallel_group
            cls.tp_group = get_tensor_model_parallel_group()
        return cls.tp_group

    @classmethod
    def init_tp_group(cls, tp_group):
        cls.tp_group = tp_group



    tp_size = None
    tp_group = None
    save_allgather_input = True
    parallel_num = 2


class CommOverlapOps(ABC):

    @staticmethod
    @abstractmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None):
        ...

    @staticmethod
    @abstractmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None):
        ...

    @staticmethod
    @abstractmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None):
        ...


COMM_OVERLAP_CONFIG = CommOverlapConfig()
