# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch


def _gather_hccl(send_tensor, recv_tensors, data_parallel_group, return_on_all_ranks=False):
    if not return_on_all_ranks:
        from mindspeed.utils import _gather_hccl as gather_hccl

        gather_hccl(send_tensor, recv_tensors, data_parallel_group)
        return

    from megatron.training import get_args

    data_parallel_world_size = data_parallel_group.size()
    stride = get_args().hccl_slice_size
    (dim1,) = send_tensor.shape
    nums_gather = (dim1 + stride - 1) // stride

    for num in range(nums_gather):
        start_index = num * stride
        end_index = min((num + 1) * stride, dim1)

        send_part = send_tensor[start_index:end_index].npu()
        recv_part = [
            torch.empty(end_index - start_index, dtype=send_tensor.dtype, device="npu")
            for _ in range(data_parallel_world_size)
        ]

        torch.distributed.all_gather(recv_part, send_part, group=data_parallel_group)

        recv_part_cpu = [tensor.cpu() for tensor in recv_part]
        for rank in range(data_parallel_world_size):
            recv_tensors[rank][start_index:end_index].copy_(recv_part_cpu[rank])

        send_part.untyped_storage().resize_(0)
        for tensor in recv_part:
            tensor.untyped_storage().resize_(0)


def get_parameter_state_dp_zero_hccl(
    self,
    use_gloo_comm=True,
    empty_data=False,
    return_on_all_ranks=False,
):
    """Get distributed optimizer parameter state with HCCL instead of Gloo."""
    _ = use_gloo_comm
    data_parallel_group = self.data_parallel_group
    data_parallel_world_size = data_parallel_group.size()
    data_parallel_rank = data_parallel_group.rank()

    state = {"buckets_coalesced": True}
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
        dtype_state = {}
        if len(gbuf_range_maps) != 1:
            raise AssertionError("single dtype supported, for now.")

        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
            world_tensors = {}
            if data_parallel_rank == 0 or return_on_all_ranks:
                world_tensors = {
                    key: torch.zeros((buffer_numel_unpadded,), dtype=torch.float32, device="cpu")
                    for key in ("param", "exp_avg", "exp_avg_sq")
                }
                world_tensors["numel_unpadded"] = buffer_numel_unpadded

            if not empty_data:
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    if gbuf_world_numel % data_parallel_world_size != 0:
                        raise AssertionError("gbuf_world_numel must be divisible by data_parallel_world_size.")
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    gbuf_world_numel_unpadded = self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    if gbuf_world_numel_unpadded > gbuf_world_numel:
                        raise AssertionError("gbuf_world_numel_unpadded must not exceed gbuf_world_numel.")

                    local_shards = {
                        key: torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        tensors = self._get_main_param_and_optimizer_states(model_param)
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(tensors[key].detach().cpu())

                    for key, send_tensor in local_shards.items():
                        if data_parallel_rank == 0 or return_on_all_ranks:
                            recv_tensors = [
                                torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                                for _ in range(data_parallel_world_size)
                            ]
                        else:
                            recv_tensors = None

                        _gather_hccl(
                            send_tensor,
                            recv_tensors,
                            data_parallel_group,
                            return_on_all_ranks=return_on_all_ranks,
                        )
                        send_tensor = None

                        if data_parallel_rank == 0 or return_on_all_ranks:
                            recv_tensors_concatenated = torch.cat(recv_tensors)
                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded
                            world_tensors[key][start:end].copy_(recv_tensors_concatenated[:gbuf_world_numel_unpadded])

                    offset_in_world_tensors += gbuf_world_numel_unpadded

            dtype_state[dtype] = world_tensors
        state[gbuf_idx] = dtype_state

    # Keep the Megatron 0.17 root-only return contract.  In particular,
    # ChainedOptimizer.save_parameter_state() asserts that non-root DP ranks
    # receive None when the caller did not request an all-ranks state dict.
    return state if data_parallel_rank == 0 or return_on_all_ranks else None


def load_parameter_state_from_dp_zero_hccl(self, state_dict, *, update_legacy_format=False):
    """Load distributed optimizer parameter state with HCCL instead of Gloo."""
    if update_legacy_format:
        return self.load_parameter_state_from_dp_zero_legacy(state_dict)

    data_parallel_group = self.data_parallel_group
    data_parallel_world_size = data_parallel_group.size()
    data_parallel_rank = data_parallel_group.rank()
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(data_parallel_group)

    if data_parallel_rank == 0:
        self.split_state_dict_if_needed(state_dict)

    from mindspeed.utils import _scatter_hccl

    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            if data_parallel_rank == 0:
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                if buffer_numel_unpadded != checkpoint_numel_unpadded:
                    raise AssertionError(
                        f"Number of unpadded elements must be same in current run "
                        f"({buffer_numel_unpadded}) and checkpoint ({checkpoint_numel_unpadded})"
                    )

            recv_tensors = {}
            for key in ("param", "exp_avg", "exp_avg_sq"):
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    if gbuf_world_numel % data_parallel_world_size != 0:
                        raise AssertionError("gbuf_world_numel must be divisible by data_parallel_world_size.")
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    gbuf_world_numel_unpadded = self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    if gbuf_world_numel_unpadded > gbuf_world_numel:
                        raise AssertionError("gbuf_world_numel_unpadded must not exceed gbuf_world_numel.")

                    recv_tensor = torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")

                    if data_parallel_rank == 0:
                        world_tensors = state_dict[gbuf_idx][dtype][key]
                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        if not 0 <= start < end <= world_tensors.numel():
                            raise AssertionError("Invalid optimizer state tensor range.")
                        world_tensor = world_tensors[start:end]
                        offset_in_world_tensors += gbuf_world_numel_unpadded

                        world_tensor = torch.nn.functional.pad(
                            world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                        )
                        if world_tensor.numel() != gbuf_world_numel:
                            raise AssertionError("Invalid padded optimizer state tensor size.")
                        gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                        send_tensors = [world_tensor[idx : (idx + gbuf_local_numel)] for idx in gbuf_start_idxs]
                    else:
                        send_tensors = None

                    _scatter_hccl(
                        recv_tensor,
                        send_tensors,
                        data_parallel_global_ranks[0],
                        data_parallel_group,
                    )

                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        if model_param not in recv_tensors:
                            recv_tensors[model_param] = {}
                        recv_tensors[model_param][key] = recv_tensor[gbuf_local_start:gbuf_local_end]

            for model_param, tensors in recv_tensors.items():
                self._set_main_param_and_optimizer_states(model_param, tensors)

    return None
