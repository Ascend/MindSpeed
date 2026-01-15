# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import re
from typing import (
    Optional,
    Callable,
    List
)
from dataclasses import dataclass
from copy import deepcopy
from sys import maxsize
from collections import defaultdict
from megatron.training import get_args
import torch.distributed as dist
from megatron.core import parallel_state as ps

domains = ['tp', 'dp', 'pp', 'ep', 'cp']


def is_a3():
    try:
        cmd = 'npu-smi info -t board -i 0 -c 0 | grep Chip | grep Name'
        chip_name = os.popen(cmd).read().strip()

        is_910B = bool(re.search(r'910B', chip_name, re.IGNORECASE))
        is_ascend910 = bool(re.search(r'Ascend910|Ascend 910', chip_name, re.IGNORECASE))

    except Exception as e:
        raise RuntimeError(f"Fail to get chip name : {str(e)}") from e

    if is_ascend910:
        return True
    return False


is_a3_version = is_a3()


@dataclass
class ParallelCommDomain:
    ip_list: List[List[str]]
    rank_list: List[List[int]]
    world_size: int
    parallel_type: str
    comm_amount: int
    comm_amount_no_overlap: int


def generate_masked_orthogonal_rank_groups(
        world_size: int, parallel_size: List[int],
        mask: List[bool]) -> List[List[int]]:
    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        '''
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        idx_stride_sum = sum([x * y for x, y in zip(idx, stride[:-1])])
        if idx_stride_sum != index:
            raise ValueError(
                "idx {} with shape {} mismatch the return idx {}".format(
                    index, shape, idx
                )
            )
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride) +
                inner_product(decomposed_group_idx, unmasked_stride))
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(self,
                 tp: int,
                 ep: int,
                 dp: int,
                 pp: int,
                 cp: int,
                 order: str,
                 rank_offset: int = 0) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        if 'ep' in order:
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(
                    f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                    f"specified the order ({self.order}).")
            elif name not in order:
                order = order + '-' + name

        self.order_w_ep = order
        self.order_wo_ep = '-'.join(
            [token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def generate_target_parallelism_match_mask(self, parallelism_order: str, target_parallelism_tokens: str):
        ordered_parallelism_tokens = parallelism_order.split('-')
        target_parallelism_token_list = target_parallelism_tokens.split('-')
        match_mask = [False] * len(ordered_parallelism_tokens)

        for parallelism_identifier in target_parallelism_token_list:
            match_mask[ordered_parallelism_tokens.index(parallelism_identifier)] = True

        return match_mask

    def get_ranks(self, token, independent_ep=False):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        """
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.generate_target_parallelism_match_mask(order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size,
                                                       parallel_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                rank_group[:] = [rank + self.rank_offset for rank in rank_group]
        return ranks


def RankGenerate():
    args = get_args()
    tp = args.tensor_model_parallel_size
    ep = args.expert_model_parallel_size
    dp = args.data_parallel_size
    pp = args.pipeline_model_parallel_size
    cp = args.context_parallel_size

    g = RankGenerator(
        tp=tp,
        ep=ep,
        dp=dp,
        pp=pp,
        cp=cp,
        order='tp-cp-ep-dp-pp',
        rank_offset=0,
    )
    return g;


def get_tensor_parallel_comm_domain():
    world_size = dist.get_world_size()
    args = get_args()

    rank_num = int(args.tensor_model_parallel_size)
    seq_length = int(args.seq_length)
    hidden_size = int(args.hidden_size)
    num_layers = int(args.num_layers)
    global_batch_size = int(args.global_batch_size)
    micro_batch_size = int(args.micro_batch_size)
    sequence_parallel = getattr(args, 'sequence_parallel', False)
    use_ascend_mc2 = getattr(args, 'use_ascend_mc2', False)
    use_ascend_coc = getattr(args, 'use_ascend_coc', False)
    micro_batches = global_batch_size // micro_batch_size

    # Compute communication ratios and counts
    comm_tp_num = 4  # 2 (forward and backward) * 2 (attention and mlp)
    comm_parallel_ratio = (rank_num - 1) / rank_num

    # Communication calculations
    comm_broadcast = 4 * seq_length
    comm_embedding = 4 * seq_length * hidden_size
    comm_transformer = 2 * num_layers * comm_tp_num * seq_length * hidden_size
    comm_vocab_parallel_ce = 2 * 3 * seq_length  # all_reduce

    if sequence_parallel:
        # Additional communication for sequence parallelism
        comm_embedding += seq_length * hidden_size
        comm_transformer += seq_length * hidden_size
        comm_transformer += 2 * num_layers * seq_length * hidden_size

    # Total communication amount per sample
    comm_tp_amount = (comm_broadcast + comm_embedding + comm_transformer +
                      comm_vocab_parallel_ce) * comm_parallel_ratio

    # Total communication amount for all samples
    comm_tp_groups_amount = micro_batches * comm_tp_amount * rank_num

    # overlap
    comm_non_overlap = comm_broadcast + comm_vocab_parallel_ce
    if use_ascend_mc2 or use_ascend_coc:
        comm_non_overlap += seq_length * hidden_size
    else:
        comm_non_overlap += 2 * num_layers * comm_tp_num // 2 * seq_length * hidden_size  # forward
        comm_non_overlap += 2 * num_layers * seq_length * hidden_size  # backward
    comm_non_overlap_groups_amount = micro_batches * comm_non_overlap * comm_parallel_ratio * rank_num

    g = RankGenerate()
    tp_group_ranks = g.get_ranks('tp')
    tp_groups_ips = None
    num_tp_groups = world_size // rank_num
    return ParallelCommDomain(
        tp_groups_ips, tp_group_ranks, world_size, 'tp',
        int(comm_tp_groups_amount) * num_tp_groups,
        int(comm_non_overlap_groups_amount) * num_tp_groups)


def get_pipeline_parallel_comm_domain():
    world_size = dist.get_world_size()
    args = get_args()

    rank_num = int(args.pipeline_model_parallel_size)
    micro_batch_size = int(args.micro_batch_size)
    tensor_model_parallel_size = int(args.tensor_model_parallel_size)
    pipeline_model_parallel_size = int(args.pipeline_model_parallel_size)
    seq_length = int(args.seq_length)
    hidden_size = int(args.hidden_size)
    num_layers = int(args.num_layers)
    global_batch_size = int(args.global_batch_size)
    sequence_parallel = getattr(args, 'sequence_parallel', False)
    num_layers_per_virtual_stage = getattr(
        args, 'num_layers_per_virtual_pipeline_stage', None)
    num_model_chunks = (num_layers // pipeline_model_parallel_size)
    pipeline_stage_num = pipeline_model_parallel_size
    micro_batches = global_batch_size // micro_batch_size
    if tensor_model_parallel_size > 1 and sequence_parallel:
        # Adjust sequence length for sequence parallelism
        seq_length = seq_length // tensor_model_parallel_size
    if (num_layers_per_virtual_stage is not None
            and int(num_layers_per_virtual_stage) < num_model_chunks):
        # Adjust pipeline stages for virtual pipeline stages
        pipeline_stage_num = num_layers // int(num_layers_per_virtual_stage)
    else:
        num_model_chunks = 1

    # Communication per micro-batch between pipeline stages
    comm_pp_num = 2  # Forward and backward
    # Communication parallel ratio
    comm_parallel_ratio = (rank_num - 1) / rank_num

    # Total communication between pipeline stages
    comm_recv_send = ((pipeline_stage_num - 1) * seq_length *
                      micro_batch_size * hidden_size) * comm_pp_num

    # Total communication amount per sample
    if tensor_model_parallel_size == 1:
        # Communication calculations
        comm_broadcast = 4 * seq_length
        comm_vocab_parallel_ce = 2 * 3 * seq_length  # all_reduce

        comm_pp_amount = (
                                 comm_broadcast + comm_vocab_parallel_ce
                         ) * comm_parallel_ratio * num_model_chunks + comm_recv_send
    else:
        comm_pp_amount = comm_recv_send

    # Total communication amount
    comm_pp_groups_amount = micro_batches * comm_pp_amount

    # overlap
    comm_non_overlap_groups_amount = comm_pp_groups_amount
    if (num_layers_per_virtual_stage is not None
            and int(num_layers_per_virtual_stage) < num_model_chunks
            and micro_batches > pipeline_model_parallel_size):
        comm_non_overlap = 0
        comm_non_overlap_stage = (seq_length * micro_batch_size *
                                  hidden_size) * comm_pp_num
        total_num_microbatches = num_model_chunks * micro_batches
        for rankid in range(rank_num):
            num_warmup_microbatches = (pipeline_model_parallel_size - rankid -
                                       1) * 2
            num_warmup_microbatches += (num_model_chunks -
                                        1) * pipeline_model_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches,
                                          total_num_microbatches)
            comm_non_overlap += num_warmup_microbatches * comm_non_overlap_stage
        if tensor_model_parallel_size == 1:
            comm_non_overlap_groups_amount = (
                                                     comm_broadcast + comm_vocab_parallel_ce
                                             ) * comm_parallel_ratio * num_model_chunks * micro_batches + comm_non_overlap
        else:
            comm_non_overlap_groups_amount = comm_non_overlap

    g = RankGenerate()
    pp_group_ranks = g.get_ranks('pp')
    pp_groups_ips = None
    num_pp_groups = world_size // rank_num
    return ParallelCommDomain(pp_groups_ips, pp_group_ranks, world_size, 'pp',
                              comm_pp_groups_amount * num_pp_groups,
                              comm_non_overlap_groups_amount * num_pp_groups)


def get_data_parallel_comm_domain():
    world_size = dist.get_world_size()
    args = get_args()
    rank_num = int(args.data_parallel_size)

    # Retrieve model parameters
    hidden_size = int(args.hidden_size)
    num_layers = int(args.num_layers)
    kv_channels = int(args.kv_channels)
    num_attention_heads = int(args.num_attention_heads)
    ffn_hidden_size = int(args.ffn_hidden_size)
    padded_vocab_size = int(args.padded_vocab_size)
    num_experts = int(args.num_experts) if getattr(args, 'num_experts',
                                                   False) else 1
    pipeline_model_parallel_size = int(args.pipeline_model_parallel_size)
    tensor_model_parallel_size = int(args.tensor_model_parallel_size)
    num_query_groups = (int(args.num_query_groups) if getattr(
        args, 'num_query_groups', False) else num_attention_heads)
    num_layers_per_virtual_stage = getattr(
        args, 'num_layers_per_virtual_pipeline_stage', None)
    use_distributed_optimizer = getattr(args, 'use_distributed_optimizer',
                                        False)
    overlap_grad_reduce = getattr(args, 'overlap_grad_reduce', False)
    overlap_param_gather = getattr(args, 'overlap_param_gather', False)
    group_query_attention = getattr(args, 'group_query_attention', False)
    if not group_query_attention:
        num_query_groups = num_attention_heads
    swiglu = getattr(args, 'swiglu', False)
    gated_linear_multiplier = 1.5 if swiglu else 1
    num_model_chunks = (num_layers // pipeline_model_parallel_size)
    untie_embeddings_and_output_weights = getattr(
        args, 'untie_embeddings_and_output_weights', False)

    # Query projection size and ratio
    query_projection_size = kv_channels * num_attention_heads
    query_projection_ratio = query_projection_size / hidden_size

    # Transformer parameters
    transformer_params = (
            2  # fp16
            * num_layers * hidden_size ** 2 *
            ((1 + (num_query_groups / num_attention_heads)) *
             query_projection_ratio  # Attention
             + (ffn_hidden_size / hidden_size) * num_experts *
             gated_linear_multiplier  # MLP
             + 2 / hidden_size  # LayerNorm layers
             + 1 / (num_layers * hidden_size)  # Final LayerNorm layer
             ))

    embedding_size = hidden_size * padded_vocab_size
    pp_size = pipeline_model_parallel_size
    tp_size = tensor_model_parallel_size

    pipeline_stage_num = pp_size
    if (num_layers_per_virtual_stage is not None
            and int(num_layers_per_virtual_stage) < num_model_chunks):
        pipeline_stage_num = num_layers // int(num_layers_per_virtual_stage)

    if untie_embeddings_and_output_weights:
        embedding_params = 2 * embedding_size
    else:
        embedding_params = embedding_size
    num_total_parameters = (transformer_params / pp_size +
                            embedding_params) / tp_size

    comm_dp_num = 2  # zero2: reduce_scatter + allgather
    # Compute communication amount
    comm_ratio = (rank_num - 1) / rank_num
    comm_dp_amount = comm_dp_num * num_total_parameters * comm_ratio
    comm_dp_groups_amount = comm_dp_amount * tp_size * pipeline_stage_num

    # overlap
    comm_non_overlap_groups_amount = comm_dp_groups_amount
    if (use_distributed_optimizer and overlap_grad_reduce):
        comm_non_overlap_groups_amount -= (comm_dp_groups_amount * 0.5)
    if (use_distributed_optimizer and overlap_param_gather):
        comm_non_overlap_groups_amount -= (comm_dp_groups_amount * 0.5)

    g = RankGenerate()
    dp_group_ranks = g.get_ranks('dp')
    dp_groups_ips = None
    num_dp_groups = world_size // rank_num
    return ParallelCommDomain(dp_groups_ips, dp_group_ranks, world_size, 'dp',
                              comm_dp_groups_amount * num_dp_groups,
                              comm_non_overlap_groups_amount * num_dp_groups)


def get_context_parallel_comm_domain():
    world_size = dist.get_world_size()
    args = get_args()

    rank_num = int(args.context_parallel_size)
    cp_groups_ips = None
    g = RankGenerate()
    cp_group_ranks = g.get_ranks('cp')

    seq_length = args.seq_length
    hidden_size = args.hidden_size
    micro_batch_size = args.micro_batch_size
    num_layers = args.num_layers

    if args.context_parallel_algo == 'megatron_cp_algo':
        comm_per_layer = 0
        # forward, the first factor 2 comes from k, v.
        comm_per_layer += seq_length * micro_batch_size * hidden_size * (
                rank_num - 1) / rank_num * 2
        # backward, dkv and kv both needs send recv
        comm_per_layer += seq_length * micro_batch_size * hidden_size * (
                rank_num - 1) / rank_num * 2 * 2

        comm_cp_groups_amount = comm_per_layer * num_layers

        comm_non_overlap_groups_amount = 0
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        # The factor 3 comes from q, k, v
        comm_per_layer = (
                                 rank_num -
                                 1) / rank_num * seq_length * micro_batch_size * hidden_size * 3
        # The factor 2 comes from forward and backward
        comm_cp_groups_amount = comm_per_layer * num_layers * 2
        comm_non_overlap_groups_amount = comm_cp_groups_amount
    else:
        ring_degree = rank_num // args.ulysses_degree_in_cp
        sub_seq_length = seq_length // ring_degree
        ulysses_comm_per_layer = (
                                         rank_num - 1
                                 ) / rank_num * sub_seq_length * micro_batch_size * hidden_size * 3 * 2

        ring_amount_per_layer = 0
        ring_comm_amount_per_layer = 0
        ring_comm_amount_per_layer += seq_length * micro_batch_size * hidden_size * (
                rank_num - 1) / rank_num * 2
        ring_comm_amount_per_layer += seq_length * micro_batch_size * hidden_size * (
                rank_num - 1) / rank_num * 2 * 2
        ring_comm_amount_per_layer /= args.ulysses_degree_in_cp

        comm_per_layer = ulysses_comm_per_layer + ring_comm_amount_per_layer

        comm_cp_groups_amount = comm_per_layer * num_layers
        comm_non_overlap_groups_amount = ulysses_comm_per_layer * num_layers

    return ParallelCommDomain(cp_groups_ips, cp_group_ranks, world_size, 'cp',
                              comm_cp_groups_amount,
                              comm_non_overlap_groups_amount)


def get_expert_parallel_comm_domain():
    world_size = dist.get_world_size()
    args = get_args()

    rank_num = int(args.expert_model_parallel_size)
    num_ep_groups = args.data_parallel_size // rank_num
    ep_groups_ips = None
    g = RankGenerate()
    ep_group_ranks = g.get_ranks('ep', independent_ep=True)
    num_ep_groups = world_size // rank_num
    topk = args.moe_router_topk

    seq_length = args.seq_length
    hidden_size = args.hidden_size
    micro_batch_size = args.micro_batch_size
    num_layers = args.num_layers

    if args.moe_token_dispatcher_type == "alltoall":
        num_tokens = micro_batch_size * seq_length * topk
        ep_comm_per_layer = num_tokens * (
                rank_num - 1) / rank_num * hidden_size * rank_num * 2
        comm_ep_groups_amount = ep_comm_per_layer * num_layers
        comm_non_overlap_groups_amount = comm_ep_groups_amount

    elif args.moe_token_dispatcher_type == "allgather":
        num_tokens = micro_batch_size * seq_length * topk
        ep_comm_per_layer = num_tokens * (rank_num -
                                          1) * hidden_size * rank_num * 2
        comm_ep_groups_amount = ep_comm_per_layer * num_layers
        comm_non_overlap_groups_amount = comm_ep_groups_amount

    return ParallelCommDomain(ep_groups_ips, ep_group_ranks, world_size, 'ep',
                              comm_ep_groups_amount * num_ep_groups,
                              comm_non_overlap_groups_amount * num_ep_groups)


def get_overlap_time_dict():
    time_overlap = {}

    keys = [
        (x, y)
        for x in domains
        for y in domains
    ]

    for key in keys:
        time_overlap[key] = 0

    args = get_args()
    time_overlap[('pp', 'tp')] = 1
    time_overlap[('pp', 'dp')] = 1
    time_overlap[('pp', 'cp')] = 1
    time_overlap[('pp', 'ep')] = 1

    time_overlap[('tp', 'pp')] = 1
    time_overlap[('dp', 'pp')] = 1
    time_overlap[('cp', 'pp')] = 1
    time_overlap[('ep', 'pp')] = 1

    if args.overlap_grad_reduce or args.overlap_param_gather:
        time_overlap[('dp', 'tp')] = 1
        time_overlap[('dp', 'pp')] = 1
        time_overlap[('dp', 'cp')] = 1
        time_overlap[('dp', 'ep')] = 1

        time_overlap[('tp', 'dp')] = 1
        time_overlap[('pp', 'dp')] = 1
        time_overlap[('cp', 'dp')] = 1
        time_overlap[('ep', 'dp')] = 1

    return time_overlap


def get_overlap_space_dict(domain_partition_information):
    boundary = 8

    if is_a3_version:
        boundary = os.environ.get('SuperNodeDieNum', 384)

    cross_boundary = []
    for domain in domains:
        cross_flag = is_cross_boundary(domain_partition_information[domain], boundary)
        if cross_flag:
            cross_boundary.append(domain)

    space_overlap = {}
    keys = [
        (x, y)
        for x in domains
        for y in domains
    ]
    for key in keys:
        space_overlap[key] = 0

    if not cross_boundary or len(cross_boundary) == 1:
        return space_overlap

    for domain_a in cross_boundary:
        for domain_b in cross_boundary:
            if domain_a != domain_b:
                space_overlap[(domain_a, domain_b)] = 1

    return space_overlap


def is_cross_boundary(comm_domains: list[list[int]], boundary: int) -> bool:
    # Validate boundary value (must be positive integer)
    if boundary <= 0:
        raise ValueError("Boundary value must be a positive integer")

    # Iterate through each communication domain
    for domain in comm_domains:
        # Skip empty communication domains
        if not domain:
            continue

        # Calculate machine ID for each rank in the domain (rank // boundary)
        # Machine ID represents which physical machine the rank belongs to
        machine_ids = {rank // boundary for rank in domain}

        # If multiple machine IDs exist in the same domain, it's cross-machine communication
        if len(machine_ids) > 1:
            return True

    # All communication domains are contained within single machines
    return False
