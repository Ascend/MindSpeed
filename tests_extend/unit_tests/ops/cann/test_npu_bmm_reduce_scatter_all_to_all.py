import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from unit_tests.common import DistributedTest
from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUBMMReduceScatterAlltoAll(DistributedTest):
    world_size = 4
    ep_size = 2
    tp_size = 2

    def supported_op_exec(self):
        bmm_out = torch.bmm(self.x_npu, self.weight_npu)
        bmm_out = bmm_out.reshape(self.reshape_1_shape)

        if self.y_shard_type == 0:
            bmm_out = bmm_out.permute(2, 0, 1, 3)
        else:
            bmm_out = bmm_out.permute(2, 0, 1, 3, 4)
        bmm_out = bmm_out.reshape(self.reshape_2_shape)
        bmm_out = bmm_out.contiguous()
        dist._reduce_scatter_base(self.reduce_scatter_out, bmm_out, op=ReduceOp.SUM, group=self.group_tp)

        self.reduce_scatter_out = self.reduce_scatter_out.reshape(self.reshape_3_shape)
        self.reduce_scatter_out = self.reduce_scatter_out.permute(1, 0, 2, 3).contiguous()
        dist.all_to_all_single(self.all_to_all_out, self.reduce_scatter_out, group=self.group_ep)
        output_golden = self.all_to_all_out.reshape(self.reshape_4_shape)
        return output_golden

    def custom_op_exec(self):
        return npu_bmm_reducescatter_alltoall(self.x_npu,
                                            self.weight_npu,
                                            bias=self.bias_npu,
                                            group_ep=self.ep_hcomm_info,
                                            group_ep_worldsize=self.ep_size,
                                            group_tp=self.tp_hcomm_info,
                                            group_tp_worldsize=self.tp_size,
                                            shard_type=self.y_shard_type)

    def get_hcomm_info(self, n, i):
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(i)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(i)
        return hcomm_info
    
    def setup_ep_tp(self, rank, tp_size, ep_size, backend_type):
        # 初始化EP域
        print("device %d initialize ep group" % rank, flush=True)
        for i in range(tp_size):
            ep_ranks = [x + ep_size * i for x in range(ep_size)]
            ep_group = dist.new_group(backend=backend_type, ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_tmp = ep_group
        print("device %d initialize tp group" % rank, flush=True)
        for i in range(ep_size):
            tp_ranks = [x * ep_size + i for x in range(tp_size)]
            tp_group = dist.new_group(backend=backend_type, ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_tmp = tp_group
        return ep_group_tmp, tp_group_tmp
    
    def get_ep_tp_hcomm_info(self, rank, ep_size, tp_size):
        hcomm_info_dist = {'ep_hcomm_info': None, 'ep_group': None, 'tp_hcomm_info': None, 'tp_group': None}
        ep_group, tp_group = self.setup_ep_tp(rank, tp_size, ep_size, "hccl")
        if torch.__version__ > '2.0.1':
            ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            tp_hcomm_info = tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            ep_hcomm_info = ep_group.get_hccl_comm_name(rank)
            tp_hcomm_info = tp_group.get_hccl_comm_name(rank)
        hcomm_info_dist['ep_hcomm_info'] = ep_hcomm_info
        hcomm_info_dist['tp_hcomm_info'] = tp_hcomm_info
        hcomm_info_dist['ep_group'] = ep_group
        hcomm_info_dist['tp_group'] = tp_group
        return hcomm_info_dist

    
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910C', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize('y_shard_type', [1])
    @pytest.mark.parametrize('transpose_weight', [False, True])
    #@pytest.mark.skip(reason="MatmulAllRedcueAddRmsNorm is not in this cann.")
    def test_npu_bmm_reducescatter_alltoall(self, dtype, y_shard_type, transpose_weight):
        rank = int(os.environ["LOCAL_RANK"])
        hcomm_info_dist = self.get_ep_tp_hcomm_info(rank, self.ep_size, self.tp_size)
        self.group_ep = hcomm_info_dist['ep_group']
        self.group_tp = hcomm_info_dist['tp_group']
        self.ep_hcomm_info = hcomm_info_dist['ep_hcomm_info']
        self.tp_hcomm_info = hcomm_info_dist['tp_hcomm_info']
        hcomm_info = self.get_hcomm_info(self.world_size, rank)
        print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {self.ep_hcomm_info}, {self.tp_hcomm_info}')
        E, C, H, M = 2, 64, 128, 256
        self.y_shard_type = y_shard_type
        if self.y_shard_type == 0:
            x_shape = (E / self.ep_size, self.ep_size * C, M / self.tp_size)
            bias_shape = (E / self.ep_size, 1, H / self.tp_size)
        else:
            x_shape = (E / self.ep_size, self.ep_size * C, M / self.tp_size)
            bias_shape = (E / self.ep_size, 1, H)
        weight_shape = (E / self.ep_size, M / self.tp_size, H)
        if transpose_weight == True:
            weight_shape = (E / self.ep_size, H, M / self.tp_size)
        
        x_shape = tuple(int(item) for item in x_shape)
        weight_shape = tuple(int(item) for item in weight_shape)
        bias_shape = tuple(int(item) for item in bias_shape)
        x = torch.rand(x_shape)
        weight = torch.rand(weight_shape)
        bias = torch.rand(bias_shape)
        self.x_npu = x.npu().to(dtype)
        self.weight_npu = weight.npu().to(dtype)
        if transpose_weight == True:
            print(f'!!!!before transpose, weight_npu.size()={self.weight_npu.size()}')
            self.weight_npu = self.weight_npu.transpose(1, 2)
            print(f'!!!!after transpose, weight_npu.size()={self.weight_npu.size()}')
            print(f'!!!!after transpose, weight_npu.is_contiguous()={self.weight_npu.is_contiguous()}')
        self.bias_npu = bias.npu().to(dtype)
        self.bias_npu = None
        
        y = self.custom_op_exec()
        print(f'y_shape = {y.size()}')
        if y_shard_type == 0:
            assert y.size() == (E, C, int(H / self.tp_size))
        else:
            assert y.size() == (E, int(C / self.tp_size), H)

        self.reshape_1_shape = [int(E / self.ep_size), self.ep_size, self.tp_size, int(C / self.tp_size), H]
        self.reshape_2_shape = [int(self.tp_size * E / self.ep_size), int(self.ep_size * C / self.tp_size), H]
        self.reshape_3_shape = [int(E / self.ep_size), self.ep_size, int(C / self.tp_size), H]
        self.reshape_4_shape = [E, int(C / self.tp_size), H]
        self.tensor_scatter_shape = [int(E / self.ep_size), int(self.ep_size * C / self.tp_size), H]
        self.alltoall_shape = [self.ep_size, int(E / self.ep_size), int(C / self.tp_size), H]
        self.reduce_scatter_out = torch.zeros(self.tensor_scatter_shape, dtype=dtype).npu()
        self.all_to_all_out = torch.zeros(self.alltoall_shape, dtype=dtype).npu()

        y_golden = self.supported_op_exec()

        for y_i, y_golden_i in zip(y, y_golden):
            assert torch.allclose(y_i, y_golden_i, rtol=0.002, atol=0.002)
