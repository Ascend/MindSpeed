# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import List, Dict, Tuple
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as MegatronDistributedOptimizer
from megatron.core import tensor_parallel
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2


class SwapDistributedOptimizer(MegatronDistributedOptimizer):
    ALL_OPTIMIZER = []

    swap_to_device_stream = None
    swap_to_host_stream = None

    swap_to_device_events_map = {}
    swap_to_host_events_map = {}
    copy_to_model_param_events_map = {}

    param_to_cpu_states_map = {}
    param_to_device_states_map = {}
    main_param_to_model_param_map = {}

    state_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']

    def __init__(self, *args, **kwargs):
        super(SwapDistributedOptimizer, self).__init__(*args, **kwargs)
        self.is_distributed_optimizer = hasattr(self, 'per_model_buffers')
        self.optimizer.is_swap_optimizer = True
        if SwapDistributedOptimizer.swap_to_device_stream is None:
            SwapDistributedOptimizer.swap_to_device_stream = torch.cuda.Stream()
            SwapDistributedOptimizer.swap_to_host_stream = torch.cuda.Stream()
        SwapDistributedOptimizer.ALL_OPTIMIZER.append(self)

        # create all parameters list for step
        self.optimizer.param_to_group_map = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.optimizer.param_to_group_map[p] = group

        # initialization optimizer states
        self.opt_states_initialization()

        # print swap param num and size
        swap_num = sum([key.numel() for key in self.main_param_to_model_param_map.keys()])
        self.optimizer.swap_numel = swap_num // 4
        total_num = sum([sum([p.numel() for p in group['params']]) for group in self.optimizer.param_groups])
        swap_memory = swap_num * 12 / 1024 / 1024
        print('[Rank {}] swap optimizer: {} ({} MB)/{}\n'.format(torch.cuda.current_device(), swap_num, swap_memory,
                                                                 total_num), end='')

    def opt_states_initialization(self):
        for group in self.shard_fp32_from_float16_groups:
            for main_param in group:
                device_state = self.optimizer.state[main_param]
                cpu_state = self.param_to_cpu_states_map[main_param]
                self.param_to_device_states_map[main_param] = device_state

                amsgrad = self.optimizer.param_to_group_map[main_param]['amsgrad']

                for key in self.state_keys:
                    if key in device_state:
                        continue
                    if key == 'max_exp_avg_sq' and not amsgrad:
                        device_state[key] = None
                        cpu_state[key] = None
                    else:
                        device_state[key] = torch.zeros_like(main_param, memory_format=torch.contiguous_format)
                        cpu_state[key] = torch.empty_like(main_param, pin_memory=True, device='cpu')
                        cpu_state[key].copy_(device_state[key], non_blocking=True)
                        device_state[key].storage().resize_(0)

    @classmethod
    def create_tensor_maps(cls, main_param, model_param):
        # optimizer parameter
        cpu_state = {'param': torch.empty_like(main_param, pin_memory=True, device='cpu')}
        cls.param_to_cpu_states_map[main_param] = cpu_state
        cls.main_param_to_model_param_map[main_param] = model_param
        cls.swap_to_host_events_map[main_param] = None
        cls.copy_to_model_param_events_map[main_param] = None

    def swap_to_host(self):
        for param in self.param_to_cpu_states_map.keys():
            self.swap_tensors_to_host(param)

    def swap_to_device(self):
        for param in self.param_to_cpu_states_map.keys():
            self.swap_tensors_to_device(param)

    @classmethod
    def copy_tensor_to_model_param(cls, param):
        cls.main_param_to_model_param_map[param].data.copy_(param)
        cls.copy_to_model_param_events_map[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def wait_copy_to_model_event(cls, param):
        event = cls.copy_to_model_param_events_map[param]
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            cls.copy_to_model_param_events_map[param] = None

    @classmethod
    def swap_tensors_to_device(cls, param):
        if param.storage().size() == 0:
            cpu_state = cls.param_to_cpu_states_map[param]
            param.storage().resize_(cpu_state['param'].storage().size())
            param.copy_(cpu_state['param'], non_blocking=True)

            if param in cls.param_to_device_states_map:
                device_state = cls.param_to_device_states_map[param]
                for key in cls.state_keys:
                    if device_state[key] is not None and device_state[key].storage().size() == 0:
                        device_state[key].storage().resize_(cpu_state[key].storage().size())
                        device_state[key].copy_(cpu_state[key], non_blocking=True)

            cls.swap_to_device_events_map[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def wait_swap_to_device_event(cls, param):
        event = cls.swap_to_device_events_map.get(param, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            cls.swap_to_device_events_map[param] = None

    @classmethod
    def swap_tensors_to_host(cls, param):
        if param.storage().size() != 0:
            cpu_state = cls.param_to_cpu_states_map[param]
            cpu_state['param'].copy_(param, non_blocking=True)
            param.storage().resize_(0)

            if param in cls.param_to_device_states_map:
                device_state = cls.param_to_device_states_map[param]
                for key in cls.state_keys:
                    if key in device_state and device_state[key] is not None and device_state[key].storage().size() != 0:
                        cpu_state[key].copy_(device_state[key], non_blocking=True)
                        device_state[key].storage().resize_(0)

            cls.swap_to_host_events_map[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def swap_all_to_device(cls):
        for op in SwapDistributedOptimizer.ALL_OPTIMIZER:
            with torch.cuda.stream(cls.swap_to_device_stream):
                op.swap_to_device()

    @classmethod
    def swap_all_to_host(cls):
        for op in SwapDistributedOptimizer.ALL_OPTIMIZER:
            with torch.cuda.stream(cls.swap_to_host_stream):
                op.swap_to_host()

    @classmethod
    def _build_model_and_main_param_groups(
            cls,
            gbuf_ranges: List[Dict],
            param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
            opt_group_ranges: List,
    ):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Allocate (or slice) each group's param shard.
        for group_range in opt_group_ranges:

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

            for model_param in group_range["params"]:

                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[
                                        param_range.start: param_range.end
                                        ]
                    shard_main_param = shard_model_param.clone().float()
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                    SwapDistributedOptimizer.create_tensor_maps(shard_main_param, shard_model_param)
                    SwapDistributedOptimizer.swap_tensors_to_host(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start: param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

            # Update optimizer's params.
            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )

    def load_parameter_state_from_dp_zero(self, state_dict):
        """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
        using the new checkpoint format with coalesced state across buckets.

        This method performs the reverse of get_parameter_state_dp_zero():
        - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
          rank receives its relevant subset of the world buffers).
        - For each DP rank, copy param & optimizer shards from contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        """

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
                for key in ("param", "exp_avg", "exp_avg_sq"):
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
                        recv_tensor = torch.zeros(
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
                            if key == "param":
                                tensor_to_copy_into = main_param
                            else:
                                optim_state = self.optimizer.state[main_param]
                                tensor_to_copy_into = optim_state[key]

                            # Copy states into contiguous shard.
                            gbuf_local_start = param_range_map["gbuf_local"].start
                            gbuf_local_end = param_range_map["gbuf_local"].end

                            if tensor_to_copy_into.storage().size() != 0:
                                tensor_to_copy_into.data.copy_(recv_tensor[gbuf_local_start:gbuf_local_end])
                            else:
                                cpu_state = SwapDistributedOptimizer.param_to_cpu_states_map[main_param]

                                # resize h2d
                                tensor_to_copy_into.storage().resize_(cpu_state[key].storage().size())

                                # copy data
                                tensor_to_copy_into.data.copy_(recv_tensor[gbuf_local_start:gbuf_local_end])

                                # swap d2h
                                cpu_state[key].copy_(tensor_to_copy_into)
                                tensor_to_copy_into.storage().resize_(0)

    def get_parameter_state_dp_zero(self):
        """Get parameter state (i.e., parameter & optimizer tensors).

        This method performs two steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Gather contiguous buffers on DP rank 0 and concatenate to world
          buffers.
        """

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Collect param states.
        state = {
            "buckets_coalesced": True,
        }
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                # Create coalesced tensors for all state related to parameters in this buffer.
                world_tensors = {}
                if data_parallel_rank == 0:
                    world_tensors = {
                        key: torch.zeros(
                            (buffer_numel_unpadded,), dtype=torch.float32, device="cpu"
                        )
                        for key in ("param", "exp_avg", "exp_avg_sq")
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
                        key: torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    # Build contiguous DP rank shards (for param + optim states).
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]

                        if main_param in self.param_to_cpu_states_map:
                            tensors = self.param_to_cpu_states_map[main_param]
                        else:
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
                                torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                                for _ in range(data_parallel_world_size)
                            ]
                        else:
                            recv_tensors = None

                        # Gather.
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
                dtype_state[dtype] = world_tensors
            state[gbuf_idx] = dtype_state

        return state

    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start: param_range.end]

                    if shard_main_param.storage().size() != 0:
                        shard_main_param.data.copy_(shard_model_param)
                    else:
                        cpu_state = SwapDistributedOptimizer.param_to_cpu_states_map[shard_main_param]
                        shard_main_param.storage().resize_(cpu_state['param'].storage().size())
                        shard_main_param.data.copy_(shard_model_param)
                        cpu_state['param'].copy_(shard_main_param)
                        shard_main_param.storage().resize_(0)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)

    def _copy_main_params_to_model_params(self):
        pass


def swap_adamw_step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    swap_count = 0
    params_list = list(self.param_to_group_map.keys())
    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError('AdamW does not support sparse gradients')

        group = self.param_to_group_map[param]
        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']
        state = self.state[param]

        if 'step' in state:
            state['step'] += 1
        else:
            state['step'] = torch.tensor(1, dtype=torch.int64, device=torch.cuda.current_device())

        # State initialization
        if len(state) == 0:
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format) if amsgrad else None

        # Swap adamw
        if swap_count == 0:
            torch.cuda.current_stream().wait_stream(SwapDistributedOptimizer.swap_to_host_stream)
            with torch.cuda.stream(SwapDistributedOptimizer.swap_to_device_stream):
                torch.cuda.current_stream().wait_stream(SwapDistributedOptimizer.swap_to_host_stream)
                while i < len(params_list) and (swap_count + params_list[i].numel() <= self.swap_numel or swap_count <= 0):
                    SwapDistributedOptimizer.swap_tensors_to_device(params_list[i])
                    swap_count += params_list[i].numel()
                    i += 1

        SwapDistributedOptimizer.wait_swap_to_device_event(param)
        npu_apply_fused_adamw_v2(param, param.grad, state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq'],
                                 state['step'], group['lr'], beta1, beta2, group['weight_decay'],
                                 group['eps'], amsgrad, group['maximize'])

        SwapDistributedOptimizer.copy_tensor_to_model_param(param)
        with torch.cuda.stream(SwapDistributedOptimizer.swap_to_host_stream):
            SwapDistributedOptimizer.wait_copy_to_model_event(param)
            swap_count -= param.numel()
            SwapDistributedOptimizer.swap_tensors_to_host(param)

    return loss
