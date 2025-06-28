import time
import os
import sys

import torch
import torch_npu

from tests_extend.tools.coc_tools.utils import Utils, Gen_Matrix, CommOverlayType, OpsType


class Ops():

    def __init__(self, tp_size=None, tp_groups=None, tp_groups_list=None, loop_num=1):
        self.tp_size = tp_size
        self.tp_groups = tp_groups
        self.tp_groups_list = tp_groups_list
        self.loop_num = loop_num

    def supported_op_exec1(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.mc2_ops import Mc2Ops
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        for _ in range(self.loop_num):
            output = Mc2Ops.allgather_matmul(x, weight, None)

        return output

    def supported_op_exec2(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.mc2_ops import Mc2Ops
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        for _ in range(self.loop_num):
            output_npu = Mc2Ops.matmul_reduce_scatter(x, weight, None)

        return output_npu

    def supported_op_exec3(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.cockernel_ops import CocKernelOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        output = Gen_Matrix.generate_zeros(x, weight, self.tp_size)

        for _ in range(self.loop_num):
            output = CocKernelOps.allgather_matmul(x, weight, None)

        return output

    def supported_op_exec4(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.cockernel_ops import CocKernelOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        output = Gen_Matrix.generate_zeros(x, weight, self.tp_size, "reduce_scatter")

        for _ in range(self.loop_num):
            output = CocKernelOps.matmul_reduce_scatter(x, weight, None)

        return output

    def supported_op_exec7(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.coc_ops import CocOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        output = Gen_Matrix.generate_zeros(x, weight, self.tp_size)

        for _ in range(self.loop_num):
            output = CocOps.allgather_matmul(x, weight, None)

        return output

    def supported_op_exec8(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.coc_ops import CocOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        output = Gen_Matrix.generate_zeros(x, weight, self.tp_size)

        for _ in range(self.loop_num):
            output = CocOps.matmul_reduce_scatter(x, weight, None)

        return output

    def supported_op_exec9(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.default_ops import DefaultOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        output = Gen_Matrix.generate_zeros(x, weight, self.tp_size)

        for _ in range(self.loop_num):
            output = DefaultOps.allgather_matmul(x, weight, None)

        return output

    def supported_op_exec10(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.default_ops import DefaultOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        output = Gen_Matrix.generate_zeros(x, weight, self.tp_size)

        for _ in range(self.loop_num):
            output = DefaultOps.matmul_reduce_scatter(x, weight, None)

        return output

    def supported_op_exec11(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.mc2_ops import Mc2Ops
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        for _ in range(self.loop_num):
            output_npu = Mc2Ops.matmul_all_reduce(x, weight, None)

        return output_npu

    def supported_op_exec12(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.cockernel_ops import CocKernelOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        for _ in range(self.loop_num):
            output_npu = CocKernelOps.matmul_all_reduce(x, weight, None)

        return output_npu

    def supported_op_exec13(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.coc_ops import CocOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        for _ in range(self.loop_num):
            output_npu = CocOps.matmul_all_reduce(x, weight, None)

        return output_npu

    def supported_op_exec14(self, x, weight):
        rank = int(os.environ["LOCAL_RANK"])
        tp_group_index = rank // self.tp_size

        from mindspeed.te.module.ops.default_ops import DefaultOps
        from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG
        COMM_OVERLAP_CONFIG.init_tp_size(self.tp_size)
        COMM_OVERLAP_CONFIG.init_tp_group(self.tp_groups[tp_group_index])

        for _ in range(self.loop_num):
            output_npu = DefaultOps.matmul_all_reduce(x, weight, None)

        return output_npu

    def supported_op_exec5(self, x, weight):
        for _ in range(self.loop_num):
            output = torch.matmul(x, weight)

        return output

    def supported_op_exec6(self, x, weight, group_list, group_type=0):
        from mindspeed.ops import gmm

        for _ in range(self.loop_num):
            output = gmm.npu_gmm_v2(x, weight, bias=None, group_list=group_list, group_type=group_type)

        return output

    def matrix_multiply(self, op_type, coc_type, A, B, C):
        if op_type == OpsType.allgather_matmul and coc_type == CommOverlayType.mc2:
            return self.supported_op_exec1(A, B)
        elif op_type == OpsType.matmul_reduce_scatter and coc_type == CommOverlayType.mc2:
            return self.supported_op_exec2(A, B.t())
        elif op_type == OpsType.allgather_matmul and coc_type == CommOverlayType.coc_kernel:
            return self.supported_op_exec3(A, B)
        elif op_type == OpsType.matmul_reduce_scatter and coc_type == CommOverlayType.coc_kernel:
            return self.supported_op_exec4(A, B.t())
        elif op_type == OpsType.allgather_matmul and coc_type == CommOverlayType.coc:
            return self.supported_op_exec7(A, B)
        elif op_type == OpsType.matmul_reduce_scatter and coc_type == CommOverlayType.coc:
            return self.supported_op_exec8(A, B.t())
        elif op_type == OpsType.allgather_matmul and coc_type == CommOverlayType.default:
            return self.supported_op_exec9(A, B)
        elif op_type == OpsType.matmul_reduce_scatter and coc_type == CommOverlayType.default:
            return self.supported_op_exec10(A, B.t())
        elif op_type == OpsType.matmul_all_reduce and coc_type == CommOverlayType.mc2:
            return self.supported_op_exec11(A, B.t())
        elif op_type == OpsType.matmul_all_reduce and coc_type == CommOverlayType.coc_kernel:
            return self.supported_op_exec12(A, B.t())
        elif op_type == OpsType.matmul_all_reduce and coc_type == CommOverlayType.coc:
            return self.supported_op_exec13(A, B.t())
        elif op_type == OpsType.matmul_all_reduce and coc_type == CommOverlayType.default:
            return self.supported_op_exec14(A, B.t())
        else:
            print("Invalid operator type. Choose fro ['mm', 'mc2', 'coc'].")
            return self.supported_op_exec14(A, B)



    def test_shapes(self, operators, coc_types, shapes, data_type, num_expert=8):
        results = {"shape": [], "operator": [], "coc_type": [], "time": [], "index": []}
        index = 0
        for coc_type in coc_types:
            index = index + 1
            for op in operators:
                for shape in shapes:
                    # 生成随机矩阵
                    A, B, C = Gen_Matrix.get_matrix(op, shape, data_type, num_expert)

                    tmp = self.loop_num
                    self.loop_num = 10
                    _ = self.matrix_multiply(op, coc_type, A, B, C)
                    # 计算运行时间
                    self.loop_num = 40
                    torch.npu.synchronize()

                    start_time = time.time()
                    _ = self.matrix_multiply(op, coc_type, A, B, C)
                    torch.npu.synchronize()
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) / self.loop_num

                    # 记录结果
                    results["shape"].append(f"{shape[0]}x{shape[1]}x{shape[2]}")
                    results["coc_type"].append(coc_type)
                    results["operator"].append(op)
                    results["time"].append(elapsed_time)
                    results["index"].append(index)
                    if torch.npu.current_device() == 0:
                        print(f"Operator: {op}, Turbo Type: {coc_type},Tensor: {shape}, Time: {elapsed_time:.6f} ")

        return results