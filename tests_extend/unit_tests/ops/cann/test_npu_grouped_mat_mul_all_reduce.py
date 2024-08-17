import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from unit_tests.common import DistributedTest
from mindspeed.op_builder import GroupedMatMulAllReduceOpBuilder

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUGroupedMatMulAllReduce(DistributedTest):
    world_size = 2
    backend = 'hccl'

    def supported_op_exec(self, rank, x, weight, bias, group_list, split_item, outputs_num):
        torch_version = torch.__version__
        if torch_version >= '2.1':
            if bias is not None and len(bias) == 0:
                bias = None
            output_npu = torch_npu.npu_grouped_matmul(x, weight, bias=bias, scale=None,
                            offset=None, antiquant_scale=None, antiquant_offset=None,
                            group_list=group_list, split_item=split_item)
        else:
            output_npu = torch_npu.npu_grouped_matmul(x, weight, bias=bias, scale=[],
                            offset=[], antiquant_scale=[], antiquant_offset=[],
                            group_list=group_list, split_item=split_item)

        for i in range(outputs_num):
            dist.all_reduce(output_npu[i], op=ReduceOp.SUM)
            if rank == 0:
                print("[supported_op_exec] allreduce i = {}, shape = {}".format(
                        i, output_npu[i].shape))

        return output_npu

    def custom_op_exec(self, x, weight, bias, group_list, split_item, hccl_group, reduce_op, comm_turn):
        mindspeed_ops = GroupedMatMulAllReduceOpBuilder().load()
        return mindspeed_ops.npu_grouped_mat_mul_all_reduce(x, weight, bias, group_list, split_item,
                    hccl_group, reduce_op, comm_turn)

    def get_hcomm_info(self, n, i):
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(i)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(i)
        return hcomm_info

    @pytest.mark.skip(reason='skip because of gmm bug')
    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_grouped_mat_mul_all_reduce(self):
        rank = int(os.environ["LOCAL_RANK"])
        hcomm_info = self.get_hcomm_info(self.world_size, rank)
        print("current device: {}, local rank = {}, hcomm_info = {}".format(torch_npu.npu.current_device(), rank, hcomm_info))

        x1 = torch.randn(32, 256, dtype=torch.float16)
        x1_npu = x1.npu()
        w1 = torch.randn(256, 32, dtype=torch.float16)
        w1_npu = w1.npu()

        x2 = torch.randn(64, 128, dtype=torch.float16)
        x2_npu = x2.npu()
        w2 = torch.randn(128, 64, dtype=torch.float16)
        w2_npu = w2.npu()

        group_list = None
        split_item = 0
        hccl_group = hcomm_info
        reduce_op = "sum"
        comm_turn = 0

        support_y = self.supported_op_exec(rank, [x1_npu, x2_npu], [w1_npu, w2_npu], [], group_list, split_item, 2)
        y = self.custom_op_exec([x1_npu, x2_npu], [w1_npu, w2_npu], [], group_list, split_item,
                                hccl_group, reduce_op, comm_turn)

        for y_i, support_y_i in zip(y, support_y):
            assert torch.allclose(y_i, support_y_i, rtol=0.005, atol=0.005)
