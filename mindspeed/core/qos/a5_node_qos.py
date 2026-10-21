# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math

DOMAINS = ('tp', 'dp', 'pp', 'ep', 'cp')
MESH_DOMAINS = {'tp', 'dp', 'ep'}


def simulate_node_traffic(domains_info, world_size):
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if domains_info is None:
        raise ValueError("domains_info is None")

    traffic = [{name: 0.0 for name in DOMAINS} for _ in range(world_size)]
    for domain in domains_info:
        name = getattr(domain, 'parallel_type', None)
        if name not in DOMAINS:
            continue
        groups = getattr(domain, 'rank_list', None) or []
        comm_amount = getattr(domain, 'comm_amount_no_overlap', 0) or 0
        if not groups or comm_amount <= 0:
            continue
        group_amount = comm_amount / len(groups)
        for group in groups:
            _accumulate_group_traffic(traffic, name, group, group_amount, world_size)
    return traffic


def cal_node_conflict_degree(single_comb_info, node_traffic, time_overlap):
    if not single_comb_info or not node_traffic or time_overlap is None:
        return 0

    degree = 0
    for each_queue in single_comb_info:
        if len(each_queue) < 2:
            continue
        for i, elem_i in enumerate(each_queue):
            for elem_j in each_queue[i + 1 :]:
                if elem_i >= len(DOMAINS) or elem_j >= len(DOMAINS):
                    continue
                name_a = DOMAINS[elem_i]
                name_b = DOMAINS[elem_j]
                if time_overlap.get((name_a, name_b), 0) != 1:
                    continue
                for rank_traffic in node_traffic:
                    degree += min(rank_traffic.get(name_a, 0), rank_traffic.get(name_b, 0))
    return degree


def select_min_conflict_groups(combs, node_traffic, time_overlap):
    if not combs:
        raise ValueError("combs is empty")

    min_comb = combs[0]
    min_degree = cal_node_conflict_degree(min_comb, node_traffic, time_overlap)
    for comb in combs[1:]:
        degree = cal_node_conflict_degree(comb, node_traffic, time_overlap)
        if degree < min_degree:
            min_degree = degree
            min_comb = comb
    return min_comb


def map_rank_queue_weights_to_levels(groups, node_traffic, queue_values):
    if not queue_values or len(queue_values) < 3:
        raise ValueError("queue_values must contain LOW/MIDDLE/HIGH")
    if node_traffic is None:
        raise ValueError("node_traffic is None")

    low, middle, high = queue_values[0], queue_values[1], queue_values[2]
    mapping = {name: middle for name in DOMAINS}
    if not groups:
        return mapping

    weights = []
    for group in groups:
        weight = 0.0
        for domain_idx in group:
            if domain_idx >= len(DOMAINS):
                continue
            amount = max(node_traffic.get(DOMAINS[domain_idx], 0), 0)
            weight += math.sqrt(amount)
        weights.append(weight)

    nonempty = [(idx, weight) for idx, weight in enumerate(weights) if weight > 0]
    if len(nonempty) <= 1:
        for group in groups:
            _assign_group(mapping, group, middle)
        return mapping

    if len(nonempty) == 2:
        nonempty.sort(key=lambda item: item[1], reverse=True)
        _assign_group(mapping, groups[nonempty[0][0]], high)
        _assign_group(mapping, groups[nonempty[1][0]], low)
        return mapping

    ordered = sorted(enumerate(weights), key=lambda item: item[1], reverse=True)
    _assign_group(mapping, groups[ordered[0][0]], high)
    _assign_group(mapping, groups[ordered[-1][0]], low)
    for idx, _ in ordered[1:-1]:
        _assign_group(mapping, groups[idx], middle)
    return mapping


def majority_vote_qos(rank_mappings, node_traffic):
    if not rank_mappings:
        raise ValueError("rank_mappings is empty")

    stats = {}
    for rank, mapping in enumerate(rank_mappings):
        key = tuple(sorted(mapping.items()))
        count, traffic, stored = stats.get(key, (0, 0.0, mapping))
        if node_traffic is not None and rank < len(node_traffic):
            traffic += sum(node_traffic[rank].values())
        stats[key] = (count + 1, traffic, stored)

    _, (_, _, winner) = min(stats.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
    return winner


def cal_a5_auto_qos(domain_list, time_overlap, queue_values, queue_nums=3):
    if domain_list is None:
        raise ValueError("domain_list is None")
    if time_overlap is None:
        raise ValueError("time_overlap is None")
    if queue_nums <= 0:
        raise ValueError("queue_nums must be positive")

    world_size = _infer_world_size(domain_list)
    node_traffic = simulate_node_traffic(domain_list, world_size)
    from mindspeed.core.qos.qos import generate_distributions

    combs = generate_distributions(len(DOMAINS), queue_nums)
    if not combs:
        raise ValueError("no valid queue combinations")
    min_comb = select_min_conflict_groups(combs, node_traffic, time_overlap)
    rank_mappings = [
        map_rank_queue_weights_to_levels(min_comb, node_traffic[rank], queue_values) for rank in range(world_size)
    ]
    return majority_vote_qos(rank_mappings, node_traffic)


def _infer_world_size(domain_list):
    world_size = 0
    for domain in domain_list:
        for group in getattr(domain, 'rank_list', None) or []:
            if not group:
                continue
            world_size = max(world_size, max(group) + 1)
    if world_size > 0:
        return world_size
    for domain in domain_list:
        world_size = max(world_size, getattr(domain, 'world_size', 0) or 0)
    if world_size <= 0:
        raise ValueError("cannot infer world_size")
    return world_size


def _accumulate_group_traffic(traffic, name, group, group_amount, world_size):
    if not group or group_amount <= 0:
        return
    ranks = []
    seen = set()
    for rank in group:
        if rank in seen:
            continue
        seen.add(rank)
        ranks.append(rank)
    rank_num = len(ranks)
    if rank_num <= 1:
        return
    if name in MESH_DOMAINS:
        _add_mesh_traffic(traffic, name, ranks, group_amount, world_size)
        return
    if name == 'pp':
        _add_pp_traffic(traffic, name, ranks, group_amount, world_size)
        return
    if name == 'cp':
        _add_cp_ring_traffic(traffic, name, ranks, group_amount, world_size)


def _add_mesh_traffic(traffic, name, ranks, group_amount, world_size):
    directed = len(ranks) * (len(ranks) - 1)
    if directed <= 0:
        return
    volume = group_amount / directed
    for src in ranks:
        for dst in ranks:
            if src == dst:
                continue
            _add_directed(traffic, name, src, dst, volume, world_size)


def _add_pp_traffic(traffic, name, ranks, group_amount, world_size):
    """Simulate PP traffic using the order of ranks as pipeline stage order.
    The ranks must preserve the order generated by RankGenerator.
    """
    directed = 2 * (len(ranks) - 1)
    if directed <= 0:
        return
    volume = group_amount / directed
    for idx in range(len(ranks) - 1):
        src = ranks[idx]
        dst = ranks[idx + 1]
        _add_directed(traffic, name, src=src, dst=dst, volume=volume, world_size=world_size)
        _add_directed(traffic, name, src=dst, dst=src, volume=volume, world_size=world_size)


def _add_cp_ring_traffic(traffic, name, ranks, group_amount, world_size):
    """Simulate CP ring traffic using the order of ranks as ring order.
    The ranks must preserve the order generated by RankGenerator.
    """
    directed = 2 * len(ranks)
    if directed <= 0:
        return
    volume = group_amount / directed
    rank_num = len(ranks)
    for idx, src in enumerate(ranks):
        nxt = ranks[(idx + 1) % rank_num]
        prv = ranks[(idx - 1) % rank_num]
        _add_directed(traffic, name, src, nxt, volume, world_size)
        _add_directed(traffic, name, src, prv, volume, world_size)


def _add_directed(traffic, name, src, dst, volume, world_size):
    if src == dst or volume <= 0:
        return
    if not (0 <= src < world_size and 0 <= dst < world_size):
        return
    traffic[src][name] += volume
    traffic[dst][name] += volume


def _assign_group(mapping, group, value):
    for domain_idx in group:
        if domain_idx >= len(DOMAINS):
            continue
        mapping[DOMAINS[domain_idx]] = value
