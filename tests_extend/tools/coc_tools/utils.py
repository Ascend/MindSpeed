import os
import random
import enum
import torch
import torch.distributed as dist



class OpsType(enum.Enum):
    allgather_matmul = 1
    matmul_reduce_scatter = 2
    matmul_all_reduce = 3


class CommOverlayType(enum.Enum):
    default = 1
    mc2 = 2
    coc_kernel = 3
    coc = 4


class Utils:

    @staticmethod
    def initialize_distributed(world_size):
        rank = int(os.environ['LOCAL_RANK'])

        if not torch.distributed.is_initialized() and rank >= 0:
            print(f'Initializing torch.distributed with rank: {rank}, world_size: {world_size}')
            print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<{torch.cuda.device_count()}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            torch.cuda.set_device(rank % world_size)
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '6000')
            init_method += master_ip + ':' + master_port
            torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank,
                                                 init_method=init_method)
            torch.distributed.barrier()

    @staticmethod
    def get_tp_list_and_groups(tp_size, world_size):
        dp_size = world_size // tp_size
        tp_groups_list = []

        tp_groups = []

        all_list = []
        all_group = []

        for i in range(dp_size):
            tp_group_list = [i * tp_size + j for j in range(tp_size)]
            tp_groups_list.append(tp_group_list)
            tp_group = dist.new_group(tp_group_list)
            tp_groups.append(tp_group)

        all_list = [*range(world_size)]
        all_group = dist.new_group(all_list)

        return tp_groups_list, tp_groups, all_list, all_group

    @staticmethod
    def get_hcomm_info(comm_group):
        rank = torch.distributed.get_rank(comm_group)
        hcomm_info = None
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(comm_group, rank)
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )

        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)

        return hcomm_info


def check_config(config):
    if config is None:
        raise AssertionError("config is None")
    if config.data_type is None:
        raise AssertionError("config error: data_type is None")
    if config.shape_list is None or not isinstance(config.shape_list, list):
        raise AssertionError("config error: shape_list is None")
    if config.comm_overlap_type is None or not isinstance(config.comm_overlap_type, list):
        raise AssertionError("config error: comm_overlap_type is None")
    if config.ops is None or not isinstance(config.ops, list):
        raise AssertionError("config error: ops is None")

    if config.data_type != torch.bfloat16 and config.data_type != torch.float16:
        raise AssertionError("config error: supported data_type is bf16 or fp16")
    if len(config.shape_list) < 1:
        raise AssertionError("config error: shape_list is empty")
    if len(config.comm_overlap_type) < 1 or len(config.comm_overlap_type) > 4:
        raise AssertionError("config error: supported comm_overlap_type >=1 and <=4")
    if len(config.ops) < 1 or len(config.ops) > 3:
        raise AssertionError("config error: support ops >=1 and <=3")


class Gen_Matrix:

    @staticmethod
    def generate_random_list(A, Num_expert):
        # 初始化列表，生成随机整数，范围为 1 到 A 的 1/Num_expert
        my_list = [random.randint(1, A // Num_expert) for _ in range(Num_expert)]

        # 计算当前列表的和
        current_sum = sum(my_list)

        # 根据需要的差值进行调整
        difference = A - current_sum

        # 不断随机调整元素，直到列表的和等于 A
        while difference != 0:
            # 随机选择一个索引来调整
            idx = random.randint(0, Num_expert - 1)

            # 如果 difference > 0，增加该元素；如果 difference < 0，减少该元素
            adjustment = random.randint(1, abs(difference))
            if difference > 0:
                my_list[idx] += adjustment
            else:
                # 确保不会调整为负数
                if my_list[idx] > adjustment:
                    my_list[idx] -= adjustment
                else:
                    continue  # 如果减法会导致负值，重新选择索引调整

            # 重新计算差值
            current_sum = sum(my_list)
            difference = A - current_sum

        return my_list

    @staticmethod
    def get_matrix(op, shape, data_type, num_expert):
        A = torch.randn(shape[0], 1, shape[1], device='npu', dtype=data_type)
        C = None
        if op != 'gmm':
            B = torch.randn(shape[1], shape[2], device='npu', dtype=data_type)
        else:
            B = torch.randn(num_expert, shape[1], shape[2], device='npu', dtype=data_type)
            C = Gen_Matrix.generate_random_list(shape[0], num_expert)
            C = torch.tensor(C, device='npu')

        return A, B, C

    @staticmethod
    def generate_zeros(x, weight, tp_size, coc_type="all_gather"):
        dim_size = list(x.shape)[:-1] + list([weight.shape[1]])

        if coc_type == "all_gather":
            dim_size[0] = dim_size[0] * tp_size
        else:
            dim_size[0] = dim_size[0] // tp_size

        return torch.zeros(dim_size[0], dim_size[1], dtype=x.dtype, device=torch.npu.current_device())
