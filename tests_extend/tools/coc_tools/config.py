import sys
import torch


class Config:
    from tests_extend.tools.coc_tools.utils import CommOverlayType, OpsType
    data_type = torch.bfloat16
    shape_list = [
        [2256, 6144, 12288],
        [18048, 6144, 12288],
        [18048, 12288, 6144],
        [2256, 12288, 6144],
        [2000, 4096, 4000],
    ]
    comm_overlap_type = [CommOverlayType.mc2, CommOverlayType.coc_kernel, CommOverlayType.coc]
    ops = [OpsType.allgather_matmul, OpsType.matmul_reduce_scatter, OpsType.matmul_all_reduce]

    model_ops_shape = [
        (OpsType.allgather_matmul, [512, 8192, 1280]),
        (OpsType.matmul_reduce_scatter, [4096, 1024, 8192]),
        (OpsType.allgather_matmul, [512, 8192, 7168]),
        (OpsType.allgather_matmul, [512, 8192, 3584]),
        (OpsType.matmul_reduce_scatter, [4096, 3584, 8192]),
        (OpsType.allgather_matmul, [512, 8192, 1024]),
    ]
