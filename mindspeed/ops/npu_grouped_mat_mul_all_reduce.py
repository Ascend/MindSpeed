from typing import Optional, List, Dict
import torch
import torch_npu
from mindspeed.op_builder import GroupedMatMulAllReduceOpBuilder

__all__ = ["npu_grouped_mat_mul_all_reduce"]


op_builder = GroupedMatMulAllReduceOpBuilder()


def npu_grouped_mat_mul_all_reduce(x: List[torch.Tensor],
                                   weight: List[torch.Tensor],
                                   hcomm: str,
                                   *,
                                   bias: Optional[List[torch.Tensor]] = None,
                                   group_list: Optional[List[int]] = None,
                                   split_item: Optional[int] = 0,
                                   reduce_op: str = "sum",
                                   comm_turn: int = 0) -> List[torch.Tensor]:
    if torch.__version__ < '2.1.0':
        if bias is None:
            bias = []
    outputs = op_builder.load().npu_grouped_mat_mul_all_reduce(
                        x, weight, bias, group_list, split_item,
                        hcomm, reduce_op, comm_turn)
    return outputs