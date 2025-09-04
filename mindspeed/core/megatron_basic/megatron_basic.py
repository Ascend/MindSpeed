# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

import time
from functools import wraps
from logging import getLogger
import torch
from torch import _C
from torch_npu.npu import _lazy_call, device as device_ctx_manager
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import HAVE_APEX_OR_TE
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D


logger = getLogger(__name__)
param_group_identifier_keys = ('wd_mult', 'lr_mult', 'is_expert_parallel', 'is_decoupled_lr')


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]

            # if graph capturing, set the rng state in a cudagraphable way
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers
        compile_helpers()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


def add_layer_norm_sp_support(config, instance):
    setattr(instance, 'config', config)
    sequence_parallel = False if not hasattr(config, 'sequence_parallel') else config.sequence_parallel
    persist_layer_norm = False if not hasattr(config, 'persist_layer_norm') else config.persist_layer_norm
    setattr(instance, 'sequence_parallel', sequence_parallel)
    setattr(instance.weight, 'sequence_parallel', sequence_parallel)
    setattr(instance.bias, 'sequence_parallel', sequence_parallel)
    setattr(instance, 'persist_layer_norm', persist_layer_norm)



class PTNorm:

    def __new__(cls, config, hidden_size: int, eps: float = 1e-5):
        if config.normalization == "LayerNorm":
            if getattr(config, "tp_2d", False):
                instance = LayerNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
            else:
                try:
                    # using apex implementation
                    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
                    instance = FusedLayerNorm(config=config, hidden_size=hidden_size, eps=eps)
                except ImportError:
                    # using torch implementation
                    instance = torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps)
                    add_layer_norm_sp_support(config, instance)
        elif config.normalization == "RMSNorm":
            if getattr(config, "tp_2d", False):
                instance = RMSNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
                instance.use_fused_rmsnorm = False
            else:
                from mindspeed.core.fusions.fused_rms_norm import RMSNorm
                instance = RMSNorm(dim=hidden_size, eps=eps, sequence_parallel=config.sequence_parallel, config=config)
                instance.config.use_fused_rmsnorm = True
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance


def get_device_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = torch.distributed.get_backend()
        local_rank = args[0]
        if backend == 'hccl':
            if local_rank is None:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{local_rank}')
        else:
            device = func(*args, **kwargs)
        return device
    return wrapper


def get_device_arch_version():
    return 8


@staticmethod
def preload_tensors(write_buckets, non_blocking=True):
    """
    Preloads tensors in `state_dict` to host memory via CPU memory.

    Args:
        write_buckets (List): List of `WriteBucket` objects that define what to
            save in a checkpoint.
        non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
    """
    result = []

    for bucket in write_buckets:
        file_name, storage_key, (bytes_data, tensor_data) = bucket
        tensor_data = [
            (item, tensor.to("cpu", non_blocking=False) if not tensor.is_cpu else tensor.clone()) for item, tensor in tensor_data
        ]
        result.append((file_name, storage_key, (bytes_data, tensor_data)))
    if non_blocking:
        torch.cuda.synchronize()
    return result


def dist_optim_load_state_dict(self, state_dict):
    """Load the state dict.

    As detailed in state_dict(), the state dict contains all non-
    parameter-related variables. This method is notably longer than
    state_dict(), because the Torch optimizers state has yet to be
    allocated at this point, and so we must do a cross referencing between
    the optimizers state (and the ordering it expects for parameter state)
    and this DP rank's shards. The optimizer at this point does not contain
    any tensor dimension information, so we must get these dimensions from
    the DP shards mapped during DistributedOptimizer.__init__().

    The tensor parameter state is loaded via load_parameter_state(), and
    so this method also must populate the loaded state dict with dummy
    tensor data (i.e., via torch.empty() below). This will be overwritten
    during load_parameter_state().

    ** Note: Torch optimizer's state structure. **
    The Torch optimizer stores its state in two levels. The top level is a
    list of groups, where each group contains a list of integer indexes
    (corresponding to parameters) that index into a master parameter list
    that is shared by all groups. As such, three values are necessary for
    maintaining this ordering:

    - group_index : The group to which a parameter belongs.
    - group_order : The index of a parameter within its group.
    - state_order : The index of a parameter within the shared parameter
        list.
    """
    if len(self.optimizer.state) == 0:
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer.dummy_step()
        elif self.ddp_config.use_custom_fsdp:
            # Initializes optimizer states with dummy values.

            # This step is necessary to ensure that the optimizers' states are
            # initialized correctly. These dummy states will be replaced in-place
            # during the loading of distributed checkpoints.
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.numel() == 0:
                        # Avoid FusedAdam errors on empty tensor input.
                        continue
                    param.grad = torch.randn_like(param)
            self.optimizer.step()
            self.optimizer.zero_grad()

    # Get the Torch optimizers' state dict.
    # - This 'inner' optimizer at this point is unallocated, and only
    #   contains an integer ordering of parameters within each group, and
    #   the ordering of parameters within its flattened parameter state
    #   list.
    def make_needed_groups(param_group):
        needed_groups = []
        for key in param_group_identifier_keys:
            # NeMo changes these variable names from `lr_mult` and `wd_mult`
            # to `pre_lr_mult` and `pre_wd_mult`, so we need to check both.
            if key in param_group:
                pass
            elif f"pre_{key}" in param_group:
                key = f"pre_{key}"
            else:
                raise ValueError(
                    f"Key {key} (or pre_{key}) not found in param_group {param_group}."
                )
            needed_groups.append(param_group[key])
        needed_groups = tuple(needed_groups)
        return needed_groups

    param_groups_map = {}
    for param_group in state_dict["optimizer"]["param_groups"]:
        needed_groups = make_needed_groups(param_group)
        param_groups_map[needed_groups] = param_group
    inner_state_dict = self.optimizer.state_dict()
    state_dict_param_groups = []
    for inner_param_group in inner_state_dict["param_groups"]:
        needed_groups = make_needed_groups(inner_param_group)
        state_dict_param_groups.append(
            {**param_groups_map[needed_groups], "params": inner_param_group['params']}
        )

    # Allocate or retrieve optimizer state (i.e., tensors).
    if len(self.optimizer.state) == 0:
        # Allocate empty optimizer state if not previously initialized.
        # - If len(self.optimizer.state) == 0, this means that the optimizer
        #   state has not been previously initialized. Once it has been
        #   initialized, we skip this code block to avoid reallocating
        #   empty tensors (i.e., torch.empty), which in turn reduces memory
        #   fragmentation.
        # - Real data is overwritten during load_parameter_state().
        state_dict_state = []
        for gbuf_range_maps in self.gbuf_ranges:
            for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Get parameter ordering information (see method docstring
                        # for details).
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        state_order = inner_state_dict["param_groups"][group_index]["params"][
                            group_order
                        ]

                        # Allocate dummy tensors.
                        numel = len(param_range_map["gbuf_world"])

                        def init_shard(elements_count):
                            return torch.empty(
                                (elements_count,), dtype=torch.float32, device=torch.cuda.current_device()
                            )

                        tensors = {"exp_avg": init_shard(numel), "exp_avg_sq": init_shard(numel)}
                        if self.config.use_precision_aware_optimizer:
                            tensors["master_param"] = init_shard(numel)
                        state_dict_state.append((state_order, tensors))

        # Sort by state order (see method docstring for details).
        state_dict_state.sort(key=lambda s: s[0])
        state_dict_state = {s[0]: s[1] for s in state_dict_state}

    else:
        # Retrieve existing optimizer state.
        state_dict_state = inner_state_dict["state"]

    # Extract 'step', for non-Apex/TE support.
    if not HAVE_APEX_OR_TE:
        steps = list(set([g["step"] for g in state_dict["optimizer"]["param_groups"]]))
        if len(steps) != 1:
            raise AssertionError(f"Expect exactly one kind of step, but detect {len(steps)} kinds of steps")
        step = torch.tensor(steps[0], dtype=torch.float)

        for s in state_dict_state.values():
            # Native PyTorch state dict requires step (i.e., iteration).
            s["step"] = step
    elif isinstance(self.optimizer, HybridDeviceOptimizer):
        # Handle Torch AdamW special case, which, unlike FusedAdam, Torch AdamW
        # has an extra optimizer state “step”.
        steps = list(
            set([g["step"] for g in state_dict["optimizer"]["param_groups"] if "step" in g])
        )
        if len(steps) != 0:
            if len(steps) != 1:
                raise AssertionError(f"steps: {steps}")
            step = torch.tensor(steps[0], dtype=torch.float32, device="cpu")
            for v in self.optimizer.state.values():
                v["step"] = step.detach().clone()

    # Optimizer.
    self.optimizer.load_state_dict(
        {"state": state_dict_state, "param_groups": state_dict_param_groups}
    )

    # Grad scaler.
    if 'grad_scaler' not in state_dict:
        if self.config.fp16:
            logger.info(
                '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
            )
    else:
        if self.grad_scaler:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
        else:
            logger.info(
                '***WARNING*** found the grad scaler in the '
                'checkpoint but it is None in the class. '
                'Skip loading grad scaler ...'
            )

    if 'param_state' in state_dict:
        if 'param_state_sharding_type' not in state_dict:
            raise AssertionError(
                f"Could not find 'param_state_sharding_type' in state_dict."
                f"Current state_dict.key(): {state_dict.key()}")
        param_state = state_dict['param_state']
        sharding_type = state_dict['param_state_sharding_type']
        if self.ddp_config.use_custom_fsdp:
            if sharding_type != "fully_sharded_model_space":
                raise AssertionError("Only fully sharded model space is supported")
        logger.info(f'Loading distributed optimizer sharded state of type {sharding_type}')
        if sharding_type == 'dp_zero_gather_scatter':
            self.load_parameter_state_from_dp_zero(param_state)
        elif sharding_type == 'fully_sharded_bucket_space':
            self.load_parameter_state_from_fs_bucket_space(param_state)
        elif sharding_type == 'fully_sharded_model_space':
            self.load_parameter_state_from_fs_model_space(param_state)
        else:
            raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')