# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Preserve ordinary GroupedMLP initialization when TP ranks own whole experts."""

from functools import wraps

import torch


def _redistribute_expert_weight(weight, num_local_experts, config, group, is_fc1):
    """Exchange ordinary TP shards for the experts assigned to this TP rank.

    Both layouts have the same number of elements per rank. Sending contiguous
    expert ranges avoids gathering a full copy of every expert on every rank.
    """
    tp_size = torch.distributed.get_world_size(group)
    send = weight.detach().reshape(-1)
    if send.device.type == "cpu" and torch.distributed.get_backend(group) != "gloo":
        # HCCL/NCCL cannot exchange CPU tensors. Keep the native CPU initializer
        # and its dtype conversion, using the device only to move the shards.
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            device = torch.device("npu", npu.current_device())
        else:
            device = torch.device("cuda", torch.cuda.current_device())
        send = send.to(device=device)
    received = torch.empty_like(send)
    torch.distributed.all_to_all_single(received, send, group=group)

    if is_fc1:
        branches = 2 if config.gated_linear_unit else 1
        # Input: [source TP, expert, hidden, gate/value, FFN shard]. Each
        # activation branch must join its own TP shards before the two branches
        # are concatenated; concatenating gate/value pairs would mix them up.
        received = received.view(tp_size, num_local_experts, config.hidden_size, branches, -1)
        restored = received.permute(1, 2, 3, 0, 4).contiguous()
    else:
        # FC2 partitions the input dimension rather than the output dimension.
        received = received.view(tp_size, num_local_experts, -1, config.hidden_size)
        restored = received.permute(1, 0, 2, 3).contiguous()
    with torch.no_grad():
        weight.copy_(restored.view_as(weight))


def tp_extend_ep_grouped_mlp_init_wrapper(init):
    """Run the unchanged initializer, then redistribute its actual TP shards.

    Preserve the original CPU or expert-device RNG stream, including its state
    after construction. No layer seed or PP/VPP offset is introduced: later
    experts, shared experts and subsequent model chunks see the same RNG states
    as a run with TP-extended EP disabled and otherwise identical settings.
    """

    @wraps(init)
    def wrapper(self, num_local_experts, config):
        if not getattr(config, "moe_tp_extend_ep", False) or not config.perform_initialization:
            return init(self, num_local_experts, config)

        from megatron.core import parallel_state

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return init(self, num_local_experts, config)

        # The TP-extended expert constructors temporarily set the cached expert
        # TP size/rank to 1/0. Read the real group, not those temporary values.
        group = parallel_state.get_expert_tensor_parallel_group()
        dense_group = parallel_state.get_tensor_model_parallel_group()
        if torch.distributed.get_process_group_ranks(group) != torch.distributed.get_process_group_ranks(dense_group):
            raise ValueError("TP-extended EP initialization requires matching dense and expert TP groups.")
        reference_num_experts = num_local_experts * tp_size
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if reference_num_experts * ep_size != config.num_moe_experts:
            raise ValueError("TP-extended EP initialization received an inconsistent local expert count.")
        if config.moe_ffn_hidden_size % tp_size != 0:
            raise ValueError("Ordinary expert initialization requires the expert FFN size to be divisible by TP.")

        previous_size = parallel_state._MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
        previous_rank = parallel_state._MPU_EXPERT_TENSOR_PARALLEL_RANK
        parallel_state.set_expert_tensor_parallel_world_size(tp_size)
        parallel_state.set_expert_tensor_parallel_rank(torch.distributed.get_rank(group))
        try:
            # Ordinary EP has TP times as many local experts, each TP-sharded.
            # This constructor therefore allocates the same parameter shapes as
            # TP-extended EP, while consuming exactly the ordinary RNG stream.
            result = init(self, reference_num_experts, config)
        finally:
            parallel_state.set_expert_tensor_parallel_world_size(previous_size)
            parallel_state.set_expert_tensor_parallel_rank(previous_rank)

        self.num_local_experts = num_local_experts
        _redistribute_expert_weight(self.weight1, num_local_experts, config, group, is_fc1=True)
        _redistribute_expert_weight(self.weight2, num_local_experts, config, group, is_fc1=False)
        return result

    return wrapper
