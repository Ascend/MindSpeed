# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from functools import wraps
import torch
import torch.distributed

from megatron.core import mpu
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.training import get_args
from megatron.training.utils import unwrap_model


def save_checkpoint_ema_wrapper(func):
    @wraps(func)
    def save_checkpoint_ema(*args, **kwargs):
        model, optimizer, opt_param_scheduler = args[1:4]
        state_dict = get_ema_model(model, optimizer)
        setattr(opt_param_scheduler, 'ema_model_state_dict', state_dict)
        func(*args[:3], opt_param_scheduler, *args[4:], **kwargs)
        setattr(opt_param_scheduler, 'ema_model_state_dict', None)

    return save_checkpoint_ema


def get_ema_model(model, optimizer):
    state_dict = dict()
    global_args = get_args()
    use_dist_ckpt = global_args.use_dist_ckpt
    unwrapped_model = unwrap_model(model)
    unchained_optimizer = unchain_optimizer(optimizer)
    ema_optimizer_applier(unchained_optimizer)
    if len(unwrapped_model) == 1:
        state_dict['ema_model'] = (unwrapped_model[0].shared_state_dict()
                                   if use_dist_ckpt else
                                   unwrapped_model[0].state_dict_for_save_checkpoint())
        state_dict = ema_state_dict_to_cpu(state_dict, 'ema_model')
        ema_optimizer_restore(unchained_optimizer)
        return state_dict
    for sub_model in unwrapped_model:
        sub_model_idx = unwrapped_model.index(sub_model)
        mpu.set_virtual_pipeline_model_parallel_rank(sub_model_idx)
        state_dict['ema_model%d' % sub_model_idx] = (
            sub_model.sharded_state_dict()
            if use_dist_ckpt else
            sub_model.state_dict_for_save_checkpoint())
        state_dict = ema_state_dict_to_cpu(
            state_dict, 'ema_model%d' % sub_model_idx)
    ema_optimizer_restore(unchained_optimizer)
    return state_dict


def unchain_optimizer(chained_optimizer):
    if isinstance(chained_optimizer, ChainedOptimizer):
        return chained_optimizer.chained_optimizers
    return [chained_optimizer]


def ema_optimizer_applier(unchained_optimizer):
    for optim in unchained_optimizer:
        optim.optimizer.store(optim.optimizer.param_groups)
        optim.optimizer.copy_to()
        param_sync(optim)


def ema_optimizer_restore(unchained_optimizer):
    for optim in unchained_optimizer:
        optim.optimizer.restore(optim.optimizer.param_groups)
        param_sync(optim)
    torch.distributed.barrier()
    for optim in unchained_optimizer:
        optim.update_successful = False


def param_sync(optim):
    if hasattr(optim, "_copy_main_params_to_model_params"):
        optim._copy_main_params_to_model_params()
    if hasattr(optim, "_reset_metadata_and_sync_gather_all_model_params"):
        optim.update_successful = True
        optim._reset_metadata_and_sync_gather_all_model_params(force_sync=True)


def ema_state_dict_to_cpu(state_dict, ema_key):
    for k, v in state_dict[ema_key].items():
        if not torch.is_tensor(v):
            continue
        new_v = v.detach().cpu().clone()
        state_dict[ema_key][k] = new_v
    return state_dict


def generate_state_dict_ema_wrapper(func):
    @wraps(func)
    def generate_state_dict_ema(*args, **kwargs):
        opt_param_scheduler = args[3]
        state_dict = func(*args, **kwargs)
        if hasattr(opt_param_scheduler, 'ema_model_state_dict'):
            ema_model_state_dict = getattr(
                opt_param_scheduler, 'ema_model_state_dict', None)
            state_dict.update(ema_model_state_dict)
        return state_dict

    return generate_state_dict_ema


def get_megatron_optimizer_func_wrapper(func):
    @wraps(func)
    def get_megatron_optimizer_func(*args, **kwargs):
        chained_optimizer = func(*args, **kwargs)
        global_args = get_args()
        if hasattr(chained_optimizer, "chained_optimizers"):
            for optim in chained_optimizer.chained_optimizers:
                optim.optimizer.ema_decay = global_args.ema_decay
            return chained_optimizer
        if hasattr(chained_optimizer, "optimizer"):
            chained_optimizer.optimizer.ema_decay = global_args.ema_decay
            return chained_optimizer
        return chained_optimizer

    return get_megatron_optimizer_func


def ema_distrib_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def ema_distrib_optimizer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        self.load_parameter_state_from_dp_zero_func_temp = self.load_parameter_state_from_dp_zero
        self.load_parameter_state_from_dp_zero = types.MethodType(load_ema_state_from_dp_zero, self)
        self.get_parameter_state_dp_zero_func_temp = self.get_parameter_state_dp_zero
        self.get_parameter_state_dp_zero = types.MethodType(get_ema_state_dp_zero, self)
    return ema_distrib_optimizer_init


def load_ema_state_from_dp_zero(self, state_dict):
    """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
    using the new checkpoint format with coalesced state across buckets.

    This method performs the reverse of get_parameter_state_dp_zero():
    - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
    rank receives its relevant subset of the world buffers).
    - For each DP rank, copy param & optimizer shards from contiguous CPU
    buffers. (e.g., one buffer each for main_param, exp_avg, and
    exp_avg_sq).
    """
    self.load_parameter_state_from_dp_zero_func_temp(state_dict)
    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )

    # Scatter tensors to all DP ranks.
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            if data_parallel_rank == 0:
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                assert buffer_numel_unpadded == checkpoint_numel_unpadded, (
                    f"Number of unpadded elements must be same in current run "
                    f"({buffer_numel_unpadded}) and checkpoint ({checkpoint_numel_unpadded})"
                )
            for key in ("ema_params",):
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    )
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                    gbuf_world_numel_unpadded = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    )
                    assert gbuf_world_numel_unpadded <= gbuf_world_numel

                    # Contiguous local shards (received from DP rank 0).
                    recv_tensor = torch.empty(
                        (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                    )

                    # Scatter tensor list.
                    if data_parallel_rank == 0:
                        world_tensors = state_dict[gbuf_idx][dtype][key]

                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        assert 0 <= start < end <= world_tensors.numel()
                        world_tensor = world_tensors[start:end]
                        offset_in_world_tensors += gbuf_world_numel_unpadded

                        # Pad world_tensor to gbuf_world_numel. Don't pad at the front, pad at the back.
                        world_tensor = torch.nn.functional.pad(
                            world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                        )
                        assert world_tensor.numel() == gbuf_world_numel
                        gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                        send_tensors = [
                            world_tensor[i: (i + gbuf_local_numel)] for i in gbuf_start_idxs
                        ]
                    else:
                        send_tensors = None

                    # Scatter.
                    if get_args().disable_gloo_group:
                        from mindspeed.utils import _scatter_hccl
                        _scatter_hccl(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            self.data_parallel_group)
                    else:
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                    # Copy local contiguous shards to param/optim shards.
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][
                            group_order
                        ]

                        optim_state = self.optimizer.state[main_param]
                        if key not in self.optimizer.state[main_param].keys():
                            optim_state[key] = main_param.clone()
                        tensor_to_copy_into = optim_state[key]

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        tensor_to_copy_into.data.copy_(
                            recv_tensor[gbuf_local_start:gbuf_local_end]
                        )


def get_ema_state_dp_zero(self):
    """Get parameter state (i.e., parameter & optimizer tensors).

    This method performs two steps:
    - For each DP rank, copy param & optimizer shards to contiguous CPU
    buffers (e.g., one buffer each for main_param, exp_avg, and
    exp_avg_sq).
    - Gather contiguous buffers on DP rank 0 and concatenate to world
    buffers.
    """
    state = self.get_parameter_state_dp_zero_func_temp()
    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

        # Iterate grad buffers (by data type).
        dtype_state = state[gbuf_idx]
        assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
            # Create coalesced tensors for all state related to parameters in this buffer.
            world_tensors = {}
            if data_parallel_rank == 0:
                world_tensors = {
                    key: torch.empty(
                        (buffer_numel_unpadded,), dtype=torch.float32, device="cpu"
                    )
                    for key in ("ema_params",)
                }
                world_tensors["numel_unpadded"] = buffer_numel_unpadded
            offset_in_world_tensors = 0
            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                # Compute local DP contiguous shard's size.
                gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                assert gbuf_world_numel % data_parallel_world_size == 0
                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                gbuf_world_numel_unpadded = (
                    self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                )
                assert gbuf_world_numel_unpadded <= gbuf_world_numel

                local_shards = {
                    key: torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                    for key in ("ema_params", )
                }

                # Build contiguous DP rank shards (for param + optim states).
                for model_param, param_range_map in gbuf_range_map["param_map"].items():

                    # Main param & optimizer states.
                    group_index, group_order = self.model_param_group_index_map[model_param]
                    main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                    optim_state = self.optimizer.state[main_param]

                    tensors = {
                        "param": main_param,
                        **optim_state,
                    }

                    # Copy states into contiguous shard.
                    gbuf_local_start = param_range_map["gbuf_local"].start
                    gbuf_local_end = param_range_map["gbuf_local"].end
                    for key in local_shards:
                        local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                            tensors[key].detach().cpu()
                        )

                # Gather contiguous shards on DP rank 0.
                for key, send_tensor in local_shards.items():

                    # Gather tensor list.
                    if data_parallel_rank == 0:
                        recv_tensors = [
                            torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                            for _ in range(data_parallel_world_size)
                        ]
                    else:
                        recv_tensors = None

                    # Gather.
                    if get_args().disable_gloo_group:
                        from mindspeed.utils import _gather_hccl
                        _gather_hccl(
                            send_tensor,
                            recv_tensors,
                            self.data_parallel_group,
                        )
                    else:
                        torch.distributed.gather(
                            send_tensor,
                            recv_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                    # Concatenate.
                    if data_parallel_rank == 0:
                        recv_tensors_concatenated = torch.cat(recv_tensors)
                        # Copy this bucket's collected all-gather tensors into the right place in the
                        # tensor for the buffer. The tensor for the buffer gets rid of the padding
                        # between buckets.
                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        world_tensors[key][start:end].copy_(
                            recv_tensors_concatenated[:gbuf_world_numel_unpadded]
                        )

                offset_in_world_tensors += gbuf_world_numel_unpadded

            # Collect world state.
            dtype_state[dtype].update(world_tensors)
        state[gbuf_idx] = dtype_state

    return state


def load_parameter_state_from_dp_zero(self, state_dict):
    self.load_parameter_state_from_dp_zero_func(state_dict)
    self.first_sub_flag = False
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )
    if data_parallel_world_size == 1 or \
        not hasattr(self, "shard_main_param_res_buffers"):
        return
    for i, shard_main_param_res_buffer in enumerate(self.shard_main_param_res_buffers):
        shard_res_numel = shard_main_param_res_buffer.numel()
        recv_tensor = torch.empty((shard_res_numel,), dtype=torch.float16, device="cpu")
        if data_parallel_rank == 0:
            send_tensors = [
                state_dict["shard_main_param_res"][i][
                    dpr * shard_res_numel: (dpr + 1) * shard_res_numel] for dpr in range(data_parallel_world_size)
            ]
        else:
            send_tensors = None

        if get_args().disable_gloo_group:
            from mindspeed.utils import _scatter_hccl
            _scatter_hccl(
                recv_tensor,
                send_tensors,
                data_parallel_global_ranks[0],
                self.data_parallel_group)
        else:
            torch.distributed.scatter(
                recv_tensor,
                send_tensors,
                data_parallel_global_ranks[0],
                data_parallel_group_gloo,
            )
        recv_tensor_bf16_view = torch.tensor(recv_tensor.data.untyped_storage(), dtype=torch.bfloat16, device=recv_tensor.device)
        shard_main_param_res_buffer.copy_(recv_tensor_bf16_view)


def get_parameter_state_dp_zero(self):
    state = self.get_parameter_state_dp_zero_func()
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )
    if data_parallel_world_size == 1 or not hasattr(self, "shard_main_param_res_buffers"):
        return state

    # gather buffer res
    buffer_res_full_shard = []
    for shard_main_param_res_buffer in self.shard_main_param_res_buffers:
        if get_args().disable_gloo_group:
            recv_tensors = [torch.empty(shard_main_param_res_buffer.numel(), dtype=torch.float16, device="cpu") for _
                            in range(data_parallel_world_size)]
        else:
            if data_parallel_rank == 0:
                recv_tensors = [torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu") for _ in range(data_parallel_world_size)]
            else:
                recv_tensors = None

        send_tensor = torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu")
        send_tensor_bf16_view = torch.tensor(send_tensor.data.untyped_storage(), dtype=torch.bfloat16, device=send_tensor.device)
        send_tensor_bf16_view.copy_(shard_main_param_res_buffer.detach().cpu())
        if get_args().disable_gloo_group:
            from mindspeed.utils import _gather_hccl
            _gather_hccl(
                send_tensor,
                recv_tensors,
                self.data_parallel_group,
            )
        else:
            torch.distributed.gather(
                send_tensor,
                recv_tensors,
                data_parallel_global_ranks[0],
                data_parallel_group_gloo,
            )
        if data_parallel_rank == 0:
            buffer_res_full_shard.append(torch.cat(recv_tensors))

    state['shard_main_param_res'] = buffer_res_full_shard
    return state