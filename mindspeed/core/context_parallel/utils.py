# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu
import torch.distributed as dist
import numpy as np
from einops import rearrange
from scipy.sparse.linalg import eigsh
from megatron.training import get_args
from megatron.core.parallel_state import get_context_parallel_global_ranks
from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
from mindspeed.core.parallel_state import get_context_parallel_for_hybrid_ring_global_ranks
from mindspeed.op_builder import AdaptiveCpOpBuilder


ADAPTIVE_CP_SCHEDULING_INFO = None
ADAPTIVE_CP_SEQ_ORDER = None
CACHED_GRID_MASK = None
CACHED_SEQ = None
CACHED_MASK_LIST = []
CACHED_SCHEDULING = None
COMM_THRESHOLD = 6
ADAPTIVE_CP_DEFAULT_SHAPE = 1024
ADAPTIVE_CP_MASK_LIST_SET_BY_USER = None
ADAPTIVE_CP_GRID_MASK_SET_BY_USER = None


# SBH -> TND
def sbh_to_tnd(x, n):
    s, b, h = x.shape
    d, t = h // n, int(b * s)
    return x.transpose(0, 1).view(t, h).view(t, n, d)


# TND -> SBH
def tnd_to_sbh(x, b):
    t, n, d = x.shape
    s, h = t // b, int(n * d)
    return x.view(b, s, n, d).transpose(0, 1).view(s, b, h)


def get_selection_indices_for_tnd_softmax_update(t, n, sub_seq_len):
    full_indices = list(range(t * n))
    cur_seq_start_idx = 0
    indices = []
    seq_start = 0
    for seq_len in sub_seq_len:
        for i in range(n):
            start = seq_start + seq_len * 2 * i + seq_len
            end = seq_start + seq_len * 2 * (i + 1)
            indices.extend(full_indices[start:end])
        seq_start += seq_len * n * 2
    
    return torch.tensor(indices)


def flatten_softmax(x, sub_seq_len):
    orig_shape = x.shape 
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    merged = [item.view(orig_shape[1], -1, orig_shape[-1]).transpose(0, 1) for item in splits]
    merged = torch.cat(merged, dim=0)
    return merged


def unflatten_softmax(x, sub_seq_len):
    orig_shape = x.shape 
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    merged = [item.view(-1, orig_shape[1], orig_shape[-1]).transpose(0, 1) \
              .view(-1, orig_shape[-1]) for item in splits]
    merged = torch.cat(merged, dim=0)
    return merged.view(*orig_shape)


def forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                 cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    if layout == 'TND':
        cur_softmax_max = flatten_softmax(cur_softmax_max, actual_seq_qlen)
        cur_softmax_sum = flatten_softmax(cur_softmax_sum, actual_seq_qlen)
        prev_softmax_max = flatten_softmax(prev_softmax_max, actual_seq_qlen)
        prev_softmax_sum = flatten_softmax(prev_softmax_sum, actual_seq_qlen)
    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    if layout == 'SBH':
        n = prev_out_scale.shape[1]
        h = prev_attn_out.shape[-1]
        d = h // n
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
        prev_out_scale = rearrange(prev_out_scale, 'b n s d -> s b (n d)').contiguous()
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
        cur_out_scale = rearrange(cur_out_scale, 'b n s d -> s b (n d)').contiguous()
    elif layout == 'TND':
        d = prev_attn_out.shape[-1]
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(2).repeat(1, 1, d)
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(2).repeat(1, 1, d)

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    if layout == 'TND':
        softmax_max = unflatten_softmax(softmax_max, actual_seq_qlen)
        softmax_sum = unflatten_softmax(softmax_sum, actual_seq_qlen)
    return attn_out, softmax_max, softmax_sum


class RingP2P:
    def __init__(self, ring_global_ranks, group, group_for_send_recv_overlap=None, is_backward=False) -> None:
        self.group = group
        self.group_for_send_recv_overlap = group
        if group_for_send_recv_overlap is not None:
            self.group_for_send_recv_overlap = group_for_send_recv_overlap

        global_rank = dist.get_rank()
        ring_rank = ring_global_ranks.index(global_rank)
        ring_size = len(ring_global_ranks)
        self.next = ring_global_ranks[(ring_rank + 1) % ring_size]
        self.prev = ring_global_ranks[(ring_rank + ring_size - 1) % ring_size]
        self.ring_rank = ring_rank
        if is_backward:
            self.next, self.prev = self.prev, self.next

        self.send_recv_ops = []
    
    def async_send_recv(self, send_tensor, recv_tensor):
        if self.ring_rank % 2 == 0:
            send_op = dist.isend(send_tensor, self.next, self.group)
            recv_op = dist.irecv(recv_tensor, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(send_op)
            self.send_recv_ops.append(recv_op)
        else:
            recv_op = dist.irecv(recv_tensor, self.prev, self.group)
            send_op = dist.isend(send_tensor, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(recv_op)
            self.send_recv_ops.append(send_op)
    
    def wait(self):
        if len(self.send_recv_ops) > 0:
            for op in self.send_recv_ops:
                op.wait()
            self.send_recv_ops = []
            return 1
        else:
            return 0


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    """
    Updates the attention output and softmax statistics for the ring attention mechanism,
    with added parameters for enhanced flexibility and extensibility.

    This function is designed to update the attention output and related softmax statistics
    for a given sequence length in a ring attention mechanism. It handles the merging of
    previous and current attention outputs and their corresponding softmax statistics.
    The introduction of `actual_seq_qlen` and `layout` parameters allows for greater flexibility
    in handling variable sequence lengths and different tensor layouts, respectively.

    Parameters:
    - prev_attn_out (Tensor): The attention output from the previous process.
    - prev_softmax_max (Tensor): The maximum value of the softmax distribution from the previous process.
    - prev_softmax_sum (Tensor): The sum of the softmax distribution from the previous process.
    - cur_attn_out (Tensor): The attention output from the current process.
    - cur_softmax_max (Tensor): The maximum value of the softmax distribution from the current process.
    - cur_softmax_sum (Tensor): The sum of the softmax distribution from the current process.
    - actual_seq_qlen (Tensor, optional): The actual sequence length for the query. This parameter
                                      is crucial for handling variable-length sequences and ensuring
                                      that the attention mechanism operates correctly under such conditions.
                                      If not provided, it defaults to the length of the current attention output.
    - layout (str, optional): The layout format of the input tensors. This parameter allows for the specification
                              of different tensor layouts, enhancing the function's versatility across various
                              model architectures. Default is 'SBH', where:
        - S: Sequence length
        - B: Batch size
        - H: Hidden size (number of attention heads)

    Returns:
    - updated_attn_out (Tensor): The updated attention output after merging previous and current process.
    - updated_softmax_max (Tensor): The updated maximum value of the softmax distribution.
    - updated_softmax_sum (Tensor): The updated sum of the softmax distribution.
    """
    _args = get_args()
    if hasattr(_args, 'use_fused_ring_attention_update') and _args.use_fused_ring_attention_update:
        def accumulate_list(input_list):
            """
            借助numpy库将列表转换为numpy数组进行元素累加，再转换回列表并在开头添加0
            """
            np_array = np.array(input_list)
            cumsum_result = np.cumsum(np_array)
            return torch.tensor([0] + list(cumsum_result), dtype=torch.int64).to(prev_attn_out.device)
        
        if layout == "TND":
            actual_seq_qlen = accumulate_list(actual_seq_qlen)
        return npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)

    return forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)


def tnd_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs, q_index, softmax_indices, cur_sub_out_seq_len):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs

    layout = 'TND'

    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=cur_sub_out_seq_len, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    else:
        n = attn_out.shape[1]
        t = attn_out.shape[0]
        prev_softmax_max = softmax_max.view(-1, 8)[softmax_indices].view(-1, n, 8)
        prev_softmax_sum = softmax_sum.view(-1, 8)[softmax_indices].view(-1, n, 8)

        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            torch.index_select(attn_out, 0, q_index), prev_softmax_max, prev_softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=cur_sub_out_seq_len, layout=layout
        )
        attn_out.index_copy_(0, q_index, attn_out_updated)
        softmax_max = softmax_max.view(-1, 8).index_copy(0, softmax_indices, softmax_max_updated.view(-1, 8)).view(-1, n, 8)
        softmax_sum = softmax_sum.view(-1, 8).index_copy(0, softmax_indices, softmax_sum_updated.view(-1, 8)).view(-1, n, 8)

    
    return [attn_out, softmax_max, softmax_sum, rng_states]


def causal_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    else:
        # [2s, b, h] -> [2, s, b, h]
        attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
        # [b, n, 2s, 8] -> [b, n, 2, s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                        2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                        2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout=layout
        )
        attn_out[1].copy_(attn_out_updated)
        softmax_max[:, :, 1, :, :].copy_(softmax_max_updated)
        softmax_sum[:, :, 1, :, :].copy_(softmax_sum_updated)
        # [2, s, b, h] -> [2s, b, h]
        attn_out = attn_out.view(-1, *attn_out.shape[2:])
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                        softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                        softmax_sum.shape[-1])
    
    return [attn_out, softmax_max, softmax_sum, rng_states]


def general_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])
    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    else:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    
    return [attn_out, softmax_max, softmax_sum, rng_states]

        
class SchedulingInfo:
    def __init__(self, round_idx, recv_q_src: int = -1, recv_kv_src: int = -1, recv_o_src: list = None,
                 send_q_dst=None, send_kv_dst: list = None, send_o_dst: int = -1, comm_unit_limit=6):
        self.round_idx = round_idx
        self.recv_q_src = recv_q_src  # 下一轮计算需要的来自别处的Q，-1代表不需要
        self.recv_kv_src = recv_kv_src  # 下一轮计算需要的来自别处的KV，-1代表不需要
        self.recv_o_src = [] if recv_o_src is None else recv_o_src  # 本轮计算中哪些device帮本机算了
        self.send_q_dst = [] if send_q_dst is None else send_q_dst  # 下一轮计算中哪些device需要本机的Q
        self.send_kv_dst = [] if send_kv_dst is None else send_kv_dst  # 下一轮计算中哪些device需要本机的KV
        self.send_o_dst = send_o_dst  # 本轮计算帮哪个device算
        self.comm_unit_limit = comm_unit_limit
        self.cnt_comm_unit_forward = -1
        self.check_eligibility()

    def check_eligibility(self):
        # 检查不能同时收Q和KV
        if self.recv_q_src > -1 and self.recv_kv_src > -1:
            raise ValueError("only receive one of q and kv in a single round")
        # 检查总通信量是否符合限制
        self.count_comm_units()
        if self.cnt_comm_unit_forward > self.comm_unit_limit:
            raise ValueError(f"comm unit exceed limit: round {self.round_idx}, device {torch.npu.current_device()}")

    def count_comm_units(self):
        sum_recv_units = self.recv_q_src > -1 + (self.recv_kv_src > -1) * 2 + len(self.recv_o_src)
        sum_send_units = len(self.send_q_dst) + len(self.send_kv_dst) * 2 + self.send_o_dst > -1
        self.cnt_comm_unit_forward = sum_recv_units + sum_send_units


def coarsen_attn_mask_npu(attn_mask, coarse_ratio):
    # 输出mask中0为需要计算的，1为不需要计算的
    orig_size = attn_mask.shape[0]
    attn_mask_reshaped = (~attn_mask)
    attn_mask_reshaped = attn_mask_reshaped.view(orig_size // coarse_ratio, coarse_ratio,
                                                 orig_size // coarse_ratio, coarse_ratio).permute(0, 2, 1, 3)
    coarse_attn_mask = ~torch.any(torch.any(attn_mask_reshaped, dim=3), dim=2)
    return coarse_attn_mask


def set_scheduling_info(cp_rank, scheduling):
    global ADAPTIVE_CP_SCHEDULING_INFO
    if ADAPTIVE_CP_SCHEDULING_INFO is None or get_args().adaptive_cp_dynamic_attn_mask:
        ADAPTIVE_CP_SCHEDULING_INFO = process_scheduling_info(cp_rank, scheduling)[1:]


def get_scheduling_info():
    if ADAPTIVE_CP_SCHEDULING_INFO is None:
        raise RuntimeError("Trying to get scheduling info before setting it, ADAPTIVE_CP_SCHEDULING_INFO is still None")
    return ADAPTIVE_CP_SCHEDULING_INFO


def set_remapped_seq_order(seq_order):
    global ADAPTIVE_CP_SEQ_ORDER
    ADAPTIVE_CP_SEQ_ORDER = seq_order


def get_remapped_seq_order():
    if ADAPTIVE_CP_SEQ_ORDER is None:
        raise RuntimeError("Trying to get optimized sequence before setting it, ADAPTIVE_CP_SEQ_ORDER is still None")
    return ADAPTIVE_CP_SEQ_ORDER


def set_adaptive_cp_mask_list_by_user(mask_list):
    global ADAPTIVE_CP_MASK_LIST_SET_BY_USER
    ADAPTIVE_CP_MASK_LIST_SET_BY_USER = mask_list


def get_adaptive_cp_mask_list_by_user():
    global ADAPTIVE_CP_MASK_LIST_SET_BY_USER
    if ADAPTIVE_CP_MASK_LIST_SET_BY_USER is None:
        raise RuntimeError("Trying to get mask list before setting it, ADAPTIVE_CP_MASK_LIST_SET_BY_USER is still None")
    return ADAPTIVE_CP_MASK_LIST_SET_BY_USER


def generate_adaptive_cp_mask_list_by_user(opt_seq, scheduling_info, cp_rank, cp_size):
    mask_list = None  # replace with customized function to generate mask list
    set_adaptive_cp_mask_list_by_user(mask_list)


def set_adaptive_cp_grid_mask_by_user(grid_mask):
    global ADAPTIVE_CP_GRID_MASK_SET_BY_USER
    ADAPTIVE_CP_GRID_MASK_SET_BY_USER = grid_mask


def get_adaptive_cp_grid_mask_by_user():
    global ADAPTIVE_CP_GRID_MASK_SET_BY_USER
    if ADAPTIVE_CP_GRID_MASK_SET_BY_USER is None:
        raise RuntimeError("Trying to get grid mask before setting it, ADAPTIVE_CP_GRID_MASK_SET_BY_USER is still None")
    return ADAPTIVE_CP_GRID_MASK_SET_BY_USER


def generate_adaptive_cp_grid_mask_by_user(cp_size):
    grid_mask = None  # replace with customized function to generate grid mask
    set_adaptive_cp_grid_mask_by_user(grid_mask)


def process_scheduling_info(local_rank, orig_scheduling, comm_limit=6):
    round_num = len(orig_scheduling)
    device_num = len(orig_scheduling[0])
    processed_scheduling_info = [SchedulingInfo(round_idx=i, comm_unit_limit=comm_limit) for i in range(round_num + 1)]
    for rnd_idx in range(round_num):
        process_single_scheduling_info(local_rank, device_num, rnd_idx, orig_scheduling[rnd_idx],
                                       processed_scheduling_info)
    return processed_scheduling_info


def process_single_scheduling_info(local_rank, device_num, round_idx, round_scheduling_info, processed_scheduling_info):
    if get_args().context_parallel_algo == 'adaptive_cp_algo':
        rank_list = get_context_parallel_global_ranks()
    else:
        rank_list = get_context_parallel_for_hybrid_ring_global_ranks()
    for execute_device_id, task_id in enumerate(round_scheduling_info):  # 当前任务和实际执行当前任务的设备
        if task_id == -1:
            continue
        origin_device_id = rank_list[int(task_id / device_num)]  # 原本应该执行当前任务的设备
        kv_device_id = rank_list[task_id % device_num]  # 存储当前任务kv的设备
        execute_device_id = rank_list[execute_device_id]
        if execute_device_id != origin_device_id:  # 需要收发qo
            if execute_device_id == local_rank:  # 当前rank对应的device是执行任务的device
                processed_scheduling_info[round_idx].recv_q_src = origin_device_id
                processed_scheduling_info[round_idx + 1].send_o_dst = origin_device_id
            elif origin_device_id == local_rank:  # 当前rank对应的device是原始的device
                processed_scheduling_info[round_idx].send_q_dst.append(execute_device_id)
                processed_scheduling_info[round_idx + 1].recv_o_src.append(execute_device_id)
        else:  # 需要收发kv
            if execute_device_id == local_rank:  # 当前rank对应的device是执行任务的device
                processed_scheduling_info[round_idx].recv_kv_src = kv_device_id
            elif kv_device_id == local_rank:  # 当前rank对应的device是存储kv的device
                processed_scheduling_info[round_idx].send_kv_dst.append(execute_device_id)
    processed_scheduling_info[round_idx].check_eligibility()


def adaptive_reschedule_task(grid_mask, cp_size):
    scheduling_info = []
    total_task = torch.sum(grid_mask)
    round_idx = 0
    next_comm = np.zeros(cp_size)
    while total_task > 0:
        scheduling_info.append([-1 for _ in range(cp_size)])
        cur_comm = next_comm
        next_comm = np.zeros(cp_size)
        total_task -= execute_scheduling(grid_mask, cp_size, round_idx, cur_comm, next_comm, scheduling_info[round_idx])
        round_idx += 1
    return scheduling_info


def execute_scheduling(grid_mask, cp_size, round_idx, cur_comm, next_comm, scheduling_info):
    count = 0
    is_free = np.ones(cp_size)
    for device_id in range(cp_size):
        row, col = find_kv_task(grid_mask, cp_size, round_idx, cur_comm, device_id, is_free)
        if row != -1 and col != -1:
            scheduling_info[device_id] = row * cp_size + col
            grid_mask[row][col] = 0
            count += 1
    is_send_q = np.zeros(cp_size, dtype=int)
    for device_id in range(cp_size):
        if is_free[device_id] == 0:
            continue
        row, col = find_qo_task(grid_mask, cp_size, cur_comm, next_comm, device_id, is_send_q)
        if row != -1 and col != -1:
            scheduling_info[device_id] = row * cp_size + col
            grid_mask[row][col] = 0
            count += 1
    return count


def find_kv_task(grid_mask, cp_size, round_idx, cur_comm, device_id, is_free):
    is_free[device_id] = 0
    row = device_id
    col = (device_id + round_idx) % cp_size
    if grid_mask[row][col] == 1:
        cur_comm[row] = cur_comm[row] + 2  # recv KV
        cur_comm[col] = cur_comm[col] + 2  # send KV
        return row, col
    for i in range(1, cp_size):  # find kv task
        row = device_id
        col = (device_id - i + cp_size) % cp_size
        if grid_mask[row][col] == 1 and cur_comm[row] <= COMM_THRESHOLD - 2 and cur_comm[col] <= COMM_THRESHOLD - 2:
            cur_comm[row] += 2  # recv KV
            cur_comm[col] += 2  # send KV
            return row, col
    is_free[device_id] = 1
    return -1, -1


def find_qo_task(grid_mask, cp_size, cur_comm, next_comm, device_id, is_send_q):
    for i in range(1, cp_size):  # find qo task
        row = (device_id + i) % cp_size
        col = device_id
        if grid_mask[row][col] == 1 and cur_comm[row] <= COMM_THRESHOLD - 1 and \
                cur_comm[col] <= COMM_THRESHOLD - 1 and is_send_q[row] != 1:
            is_send_q[row] = 1
            cur_comm[row] += 1  # send Q
            cur_comm[col] += 1  # recv Q
            next_comm[row] += 1  # recv O
            next_comm[col] += 1  # send O
            return row, col
    return -1, -1


def clear_global_info():
    global CACHED_SEQ, CACHED_GRID_MASK, CACHED_MASK_LIST, CACHED_SCHEDULING, ADAPTIVE_CP_SCHEDULING_INFO
    CACHED_SEQ, CACHED_GRID_MASK, CACHED_MASK_LIST, CACHED_SCHEDULING, ADAPTIVE_CP_SCHEDULING_INFO = (None, None, [],
                                                                                                      None, None)


class AdaptiveCpOps:
    def __init__(self):
        self.ops = AdaptiveCpOpBuilder().load()

    def coarsen_attn_mask_cpu(self, attn_mask, sampling_ratio):
        if not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()
        mask_size_after_sampling = attn_mask.shape[0] // sampling_ratio
        coarse_mask = torch.ones((mask_size_after_sampling, mask_size_after_sampling), dtype=torch.bool)
        self.ops.coarsen_mask(attn_mask, mask_size_after_sampling, coarse_mask)
        return coarse_mask

    def get_grid_mask(self, attn_mask, cp_size):
        if not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()
        if get_args().attention_mask_on_cpu:
            grid_mask = torch.ones((cp_size, cp_size), dtype=torch.bool)
            self.ops.coarsen_mask(attn_mask, cp_size, grid_mask)
        else:
            grid_mask = coarsen_attn_mask_npu(attn_mask, attn_mask.shape[0] // cp_size)
        grid_mask = ~grid_mask
        return grid_mask

    def search_kmeans_cpu(self, attn_mask, reduced_mask, cp_size, num_iters=100):
        tmp_attn_mask = torch.ones_like(attn_mask)
        tmp_grid_mask = torch.ones((cp_size, cp_size), dtype=torch.bool)
        optimal_attn_mask = torch.ones_like(attn_mask)
        optimal_grid_mask = torch.ones((cp_size, cp_size), dtype=torch.bool)
        optimal_num_cluster = [-1]
        optimal_sorted_indices = self.ops.search_kmeans(attn_mask, reduced_mask, tmp_attn_mask, tmp_grid_mask,
                                                        optimal_grid_mask, optimal_attn_mask,
                                                        optimal_num_cluster, cp_size, num_iters)
        return optimal_sorted_indices, optimal_grid_mask, optimal_attn_mask, optimal_num_cluster

    def adaptive_remap(self, attn_mask, cp_size, truncated_dim=10):
        args = get_args()
        if attn_mask.dim() != 2 or attn_mask.shape[0] != attn_mask.shape[1]:
            raise RuntimeError("Only 2-dimensional self-attention mask supported in adaptive cp")

        if args.adaptive_cp_without_coarse:
            sampling_ratio = 1
            if args.attention_mask_on_cpu:
                coarse_mask = attn_mask
            else:
                coarse_mask = attn_mask.cpu()
        else:
            if attn_mask.shape[0] % ADAPTIVE_CP_DEFAULT_SHAPE != 0:
                raise RuntimeError("Shape of attention mask needs to be a multiple of 1024 if not enable "
                                   "args.adaptive_cp_without_coarse in adaptive cp")
            if args.attention_mask_on_cpu:
                sampling_ratio = attn_mask.shape[0] // ADAPTIVE_CP_DEFAULT_SHAPE
                coarse_mask = self.coarsen_attn_mask_cpu(attn_mask, sampling_ratio)
            else:
                sampling_ratio = attn_mask.shape[0] // ADAPTIVE_CP_DEFAULT_SHAPE
                coarse_mask = coarsen_attn_mask_npu(attn_mask, sampling_ratio).cpu()

        coarse_mask_np = coarse_mask.to(torch.float16).numpy()
        mean_matrix = np.mean(coarse_mask_np, axis=0)
        centered_matrix = (coarse_mask_np - mean_matrix).astype(float)
        cov_matrix = np.matmul(centered_matrix.T, centered_matrix)
        eigenvalues, eigenvectors = eigsh(cov_matrix, k=truncated_dim, which='LM')
        feature_matrix = np.matmul(coarse_mask_np, eigenvectors).tolist()

        optimal_seq, optimal_grid_mask, optimal_coarsen_attn_mask, optimal_num_cluster = (
            self.search_kmeans_cpu(coarse_mask, feature_matrix, cp_size))

        if args.adaptive_cp_without_coarse:
            final_opt_seq = optimal_seq
        else:
            final_opt_seq = sampling_ratio * torch.tensor(optimal_seq)[:, None] + torch.arange(sampling_ratio)
            final_opt_seq = final_opt_seq.view(-1).tolist()

        optimal_grid_mask = ~optimal_grid_mask

        return optimal_grid_mask, final_opt_seq

    def get_adaptive_cp_info(self, attn_mask, cp_size):
        args = get_args()
        global CACHED_GRID_MASK, CACHED_SEQ
        if args.attention_mask_on_cpu != (attn_mask.device.type == 'cpu'):
            raise RuntimeError("args.attention_mask_on_cpu does not match the device of set attention mask")

        # 生成重映射后的序列和重排后的gird mask，输出tensor(npu/cpu) opt_grid_mask和list opt_seq
        if not args.adaptive_cp_only_reschedule:
            if args.adaptive_cp_dynamic_attn_mask or CACHED_GRID_MASK is None:
                opt_grid_mask, opt_seq = self.adaptive_remap(attn_mask, cp_size)
                if not args.adaptive_cp_dynamic_attn_mask:
                    CACHED_GRID_MASK, CACHED_SEQ = opt_grid_mask, opt_seq
            else:
                opt_grid_mask, opt_seq = CACHED_GRID_MASK, CACHED_SEQ
        else:
            opt_seq = list(range(attn_mask.shape[0]))
            if args.adaptive_cp_dynamic_attn_mask or CACHED_GRID_MASK is None:
                opt_grid_mask = self.get_grid_mask(attn_mask, cp_size)
                CACHED_GRID_MASK = opt_grid_mask
            else:
                opt_grid_mask = CACHED_GRID_MASK

        # 生成调度方案
        opt_scheduling = adaptive_reschedule_task(opt_grid_mask, cp_size)

        return opt_seq, opt_scheduling

    def get_mask_list(self, attn_mask, opt_scheduling, opt_seq, cp_rank, cp_size):
        args = get_args()
        global CACHED_MASK_LIST
        if not args.adaptive_cp_dynamic_attn_mask and len(CACHED_MASK_LIST) > 0:
            return CACHED_MASK_LIST
        round_num = len(opt_scheduling)
        grid_size = attn_mask.shape[0] // cp_size
        mask_list = []

        for rnd_idx in range(round_num):
            task_id = opt_scheduling[rnd_idx][cp_rank]
            if task_id == -1:
                mask_list.append(None)
                continue
            q_device_id = task_id // cp_size
            kv_device_id = task_id % cp_size
            if args.attention_mask_on_cpu:
                mask_list.append(torch.empty((grid_size, grid_size), dtype=torch.bool, device='cpu'))
                if args.adaptive_cp_only_reschedule:
                    grid_inds = [q_device_id, kv_device_id]
                    self.ops.get_mask_list_without_remap(attn_mask, mask_list[rnd_idx], grid_inds, cp_size)
                else:
                    q_token_list = opt_seq[grid_size * q_device_id: grid_size * (q_device_id + 1)]
                    kv_token_list = opt_seq[grid_size * kv_device_id: grid_size * (kv_device_id + 1)]
                    self.ops.get_mask_list_with_remap(attn_mask, mask_list[rnd_idx], q_token_list, kv_token_list)
            else:
                q_token_list = opt_seq[grid_size * q_device_id: grid_size * (q_device_id + 1)]
                kv_token_list = opt_seq[grid_size * kv_device_id: grid_size * (kv_device_id + 1)]
                mask_list.append(attn_mask[q_token_list, :][:, kv_token_list])

        if args.attention_mask_on_cpu:
            for rnd_idx in range(round_num):
                if mask_list[rnd_idx] is not None:
                    mask_list[rnd_idx] = mask_list[rnd_idx].npu(non_blocking=True)

        CACHED_MASK_LIST = mask_list
        return mask_list
        