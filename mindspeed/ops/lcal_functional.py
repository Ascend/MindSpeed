import torch

from megatron.core import mpu

from mindspeed.op_builder import LcalOpBuilder


class CoCConfig:
    def __init__(self, rank_id, rank_size, comm_domain):
        self.rank_id = rank_id
        self.rank_size = rank_size
        self.comm_domain = comm_domain


class TP2DConfig:
    def __init__(self, ag_dim, rs_dim, inner_dim_is_ag):
        self.ag_dim = ag_dim
        self.rs_dim = rs_dim
        self.inner_dim_is_ag = inner_dim_is_ag


class CoCOperations:
    mindspeed_ops = LcalOpBuilder().load()

    def __init__(self):
        self.comm_config = None

    def set_comm_config(self, config):
        self.comm_config = config

    def matmul_all_reduce(self, input1, input2, output, bias=None):
        device = input1.device.index
        tp_size = mpu.get_tensor_model_parallel_world_size()
        comm_domain = str(device // tp_size)
        rank = device % tp_size
        CoCOperations.mindspeed_ops.matmul_all_reduce(input1, input2, bias, output, rank, tp_size, comm_domain)
        return output

    def all_gather_matmul(self, input1, input2, output, bias=None):
        device = input1.device.index
        tp_size = mpu.get_tensor_model_parallel_world_size()
        comm_domain = str(device // tp_size)
        rank = device % tp_size
        CoCOperations.mindspeed_ops.all_gather_matmul(input1, input2, bias, output, rank, tp_size, comm_domain)
        return output

    def all_gather_matmul_v2(self, input1, input2, output, comm_output, bias=None):
        device = input1.device.index
        if self.comm_config is None:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            comm_domain = str(device // tp_size)
            rank = device % tp_size
        else:
            tp_size = self.comm_config.rank_size
            comm_domain = self.comm_config.comm_domain
            rank = self.comm_config.rank_id
        CoCOperations.mindspeed_ops.all_gather_matmul_v2(input1, input2, bias, output, comm_output, rank, tp_size, comm_domain)
        return output, comm_output

    def matmul_reduce_scatter(self, input1, input2, output, bias=None):
        device = input1.device.index
        tp_size = mpu.get_tensor_model_parallel_world_size()
        comm_domain = str(device // tp_size)
        rank = device % tp_size
        CoCOperations.mindspeed_ops.matmul_reduce_scatter(input1, input2, bias, output, rank, tp_size, comm_domain)
        return output

    def pure_matmul(self, input1, input2, output, bias=None):
        device = input1.device.index
        tp_size = mpu.get_tensor_model_parallel_world_size()
        comm_domain = str(device // tp_size)
        rank = device % tp_size
        CoCOperations.mindspeed_ops.pure_matmul(input1, input2, bias, output, rank, tp_size, comm_domain)
        return output
    
    def all_gather_matmul_reduce_scatter(self, input1, input2, output, tp2d_config, bias=None):
        from megatron.core.parallel_state import get_tensor_model_parallel_rank
        tp_size = tp2d_config.ag_dim * tp2d_config.rs_dim
        rank = get_tensor_model_parallel_rank()
        comm_domain = "1"
        CoCOperations.mindspeed_ops.all_gather_matmul_reduce_scatter(input1, input2, bias, output, rank, tp_size, comm_domain, tp2d_config.ag_dim, tp2d_config.rs_dim, tp2d_config.inner_dim_is_ag)
        return output

coc_ops = CoCOperations()
