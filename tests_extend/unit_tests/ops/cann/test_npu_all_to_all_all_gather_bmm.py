import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from unit_tests.common import DistributedTest
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUAlltoAllAllGatherBMM(DistributedTest):
    world_size = 8
    ep_size = 4
    tp_size = 2

    def activate(self, bmm_out_golden):
        act_type = self.act_type
        if act_type == "none":
            output = bmm_out_golden
        elif act_type == "gelu":
            gelu = torch.nn.GELU()
            output = gelu(bmm_out_golden)
        elif act_type == "silu":
            selu = torch.nn.SELU()
            output = selu(bmm_out_golden)
        elif act_type == "relu":
            relu = torch.nn.ReLU()
            output = relu(bmm_out_golden)
        elif act_type == "fastgelu":
            output = bmm_out_golden / (1 + torch.exp(-1.702 * bmm_out_golden))

        return output

    def supported_op_exec(self):
        dist.all_to_all_single(self.all_to_all_out, self.x_npu, group=self.group_ep)
        self.all_to_all_out = self.all_to_all_out.reshape(self.reshape_1_shape).permute(1, 0, 2, 3).contiguous()
        dist._all_gather_base(self.tensor_allgather, self.all_to_all_out, group=self.group_tp)
        all_gather_out = self.tensor_allgather.reshape(self.reshape_2_shape)
        if self.x_shard_type == 0:
            all_gather_out = all_gather_out.permute(1, 2, 0, 3)
        else:
            all_gather_out = all_gather_out.permute(1, 2, 0, 3, 4)
        gather_output_golden = all_gather_out.reshape(self.reshape_3_shape)
        bmm_out_golden = torch.bmm(gather_output_golden, self.weight_npu)
        output_golden = self.activate(bmm_out_golden)
        return output_golden, gather_output_golden, bmm_out_golden

    def custom_op_exec(self):
        return npu_alltoall_allgather_bmm(self.x_npu,
                                          self.weight_npu,
                                          bias=self.bias_npu,
                                          group_ep=self.ep_hcomm_info,
                                          group_ep_worldsize=self.ep_size,
                                          group_tp=self.tp_hcomm_info,
                                          group_tp_worldsize=self.tp_size,
                                          shard_type=self.x_shard_type,
                                          act_type=self.act_type,
                                          need_allgather_out=self.out_y2_flag,
                                          need_activation_feature=self.out_y3_flag)
    
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
    @pytest.mark.parametrize('out_y2_flag', [False, True])
    @pytest.mark.parametrize('out_y3_flag', [False])
    @pytest.mark.parametrize('act_type', ["none"])
    @pytest.mark.parametrize('transpose_weight', [False, True])
    #@pytest.mark.skip(reason="MatmulAllRedcueAddRmsNorm is not in this cann.")
    def test_npu_alltoall_allgather_bmm(self, dtype, out_y2_flag, out_y3_flag, act_type, transpose_weight):
        rank = int(os.environ["LOCAL_RANK"])
        hcomm_info_dist = self.get_ep_tp_hcomm_info(rank, self.ep_size, self.tp_size)
        self.group_ep = hcomm_info_dist['ep_group']
        self.group_tp = hcomm_info_dist['tp_group']
        self.ep_hcomm_info = hcomm_info_dist['ep_hcomm_info']
        self.tp_hcomm_info = hcomm_info_dist['tp_hcomm_info']

        self.x_shard_type = 1
        self.out_y2_flag = out_y2_flag
        self.out_y3_flag = out_y3_flag
        self.act_type = act_type

        print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {self.ep_hcomm_info}, {self.tp_hcomm_info}')

        E, C, H, M = 16, 282, 12288, 12288
        if self.x_shard_type == 0:
            x_shape = (E, C, H / self.tp_size)
        elif self.x_shard_type == 1:
            x_shape = (E, C / self.tp_size, H)
        else:
            x_shape = (E / self.ep_size, self.tp_size * self.ep_size * C, M / self.tp_size)

        weight_shape = (E / self.ep_size, H, M / self.tp_size)
        if transpose_weight == True:
            weight_shape = (E / self.ep_size, M / self.tp_size, H)

        bias_shape = (E / self.ep_size, 1, M / self.tp_size)
        
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
        
        y1, y2, y3 = self.custom_op_exec()
        print(f'y1_shape = {y1.size()}')
        assert y1.size() == (int(E / self.ep_size), self.ep_size * C, int(M / self.tp_size))
        if self.out_y2_flag == True:
            assert y2.size() == (int(E / self.ep_size), self.ep_size * C, H)
        if self.act_type != "None" and self.out_y3_flag == True:
            if self.x_shard_type == 0:
                assert y3.size() == (int(E / self.ep_size), self.ep_size * C, int(M / self.tp_size))
            elif self.x_shard_type == 1:
                assert y3.size() == (int(E / self.ep_size), self.tp_size * self.ep_size * C, int(M / self.tp_size))

        self.reshape_1_shape = [self.ep_size, int(E / self.ep_size), int(C / self.tp_size), H]
        self.reshape_2_shape = [self.tp_size, int(E / self.ep_size), self.ep_size, int(C / self.tp_size), H]
        self.reshape_3_shape = [int(E / self.ep_size), int(self.ep_size * C), H]
        self.tensor_alltoall_shape = x_shape
        self.tensor_allgather_shape = [int(self.tp_size * E / self.ep_size), self.ep_size, int(C / self.tp_size), H]
        self.all_to_all_out = torch.zeros(self.tensor_alltoall_shape, dtype=dtype).npu()
        self.tensor_allgather = torch.zeros(self.tensor_allgather_shape, dtype=dtype).npu()

        y1_golden, y2_golden, y3_golden = self.supported_op_exec()

        if self.x_npu.dtype == torch.float16:
            diff_hold = 0.001
        elif self.x_npu.dtype == torch.bfloat16:
            diff_hold = 0.005
        else:
            diff_hold = 0.002
        print(f'diff threshold is {diff_hold}')
        for y1_i, y1_golden_i in zip(y1, y1_golden):
            assert torch.allclose(y1_i, y1_golden_i, rtol=diff_hold, atol=diff_hold)
        
        if self.out_y2_flag == True:
            for y2_i, y2_golden_i in zip(y2, y2_golden):
                assert torch.allclose(y2_i, y2_golden_i, rtol=diff_hold, atol=diff_hold)
        if self.out_y3_flag == True:
            for y3_i, y3_golden_i in zip(y3, y3_golden):
                assert torch.allclose(y3_i, y3_golden_i, rtol=diff_hold, atol=diff_hold)