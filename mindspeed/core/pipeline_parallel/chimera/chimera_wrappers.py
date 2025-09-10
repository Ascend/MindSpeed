# Copyright (c) 2024; NVIDIA CORPORATION.
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.

from functools import wraps
import os
from datetime import timedelta
import warnings
import torch
import torch.distributed
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
    get_tensor_model_parallel_world_size,
    RankGenerator
)
from megatron.core.utils import get_model_config, prepare_input_tensors_for_wgrad_compute
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as DDP
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.training import get_args, print_rank_0
from megatron.training.training import build_train_valid_test_data_loaders
from megatron.legacy.model import Float16Module
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core import parallel_state
from mindspeed.core.pipeline_parallel.chimera.weight_grad_store import WeightGradStore
from mindspeed.core.parallel_state import (
    set_virtual_data_parallel_group, 
    set_virtual_data_parallel_world_size, 
    get_virtual_data_parallel_rank, 
    get_virtual_data_parallel_world_size, 
    set_virtual_data_parallel_rank, 
    get_virtual_data_parallel_group, 
    get_virtual_data_modulo_expert_parallel_group,
    get_embedding_group, 
    set_embedding_group, 
    get_embedding_ranks, 
    set_embedding_ranks,
    set_virtual_data_modulo_expert_parallel_group,
)


def init_chimera_parallel_state(args):
    ## NOTE: construct virtual data parallel communication group for milti-pipeline
    set_virtual_data_parallel_world_size(args.virtual_data_parallel_size)
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    data_parallel_size: int = world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size)
    if args.virtual_data_parallel_size is not None:
        rank_generator = RankGenerator(
            tp=args.tensor_model_parallel_size, 
            ep=args.expert_model_parallel_size, 
            dp=data_parallel_size, 
            pp=args.pipeline_model_parallel_size, 
            cp=args.context_parallel_size, 
            order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp'
        )
        pp_group_size = world_size // args.pipeline_model_parallel_size
        pp_rank = global_rank // pp_group_size
        num_dp_groups_per_stage = pp_group_size // data_parallel_size
        num_dp_cp_groups_per_stage = pp_group_size // (data_parallel_size * args.context_parallel_size)
        
        ep_world_size = args.expert_model_parallel_size
        num_experts = args.num_experts
        if num_experts is not None:
            num_dp_modulo_ep_groups_per_stage = pp_group_size // (data_parallel_size // ep_world_size)
            num_dp_cp_modulo_ep_groups_per_stage = pp_group_size // (data_parallel_size * args.context_parallel_size // ep_world_size)
            num_tp_ep_groups_per_stage = pp_group_size // (args.tensor_model_parallel_size * ep_world_size)
            num_ep_groups_per_stage = pp_group_size // ep_world_size
            args.expert_model_parallel_size = ep_world_size * args.virtual_data_parallel_size
        offset_unit = args.pipeline_model_parallel_size // (args.virtual_data_parallel_size // 2)

        all_data_parallel_group_ranks = []
        all_data_parallel_group_ranks_with_cp = []
        all_data_modulo_expert_group_ranks = []
        all_data_modulo_expert_group_with_cp_ranks = []
        all_tp_ep_group_ranks = []
        all_ep_group_ranks = []
        timeout = timedelta(minutes=args.distributed_timeout_minutes)
        # find the dp groups and dp-cp dp_moduloep, dp_modulo_ep_with_cp groups
        for ranks in rank_generator.get_ranks("dp"):
            all_data_parallel_group_ranks.append(ranks)
        for ranks in rank_generator.get_ranks("dp-cp"):
            all_data_parallel_group_ranks_with_cp.append(ranks)
        if num_experts is not None:
            for ranks in rank_generator.get_ranks("dp", independent_ep=True):
                all_data_modulo_expert_group_ranks.append(ranks)
            for ranks in rank_generator.get_ranks("dp-cp", independent_ep=True):
                all_data_modulo_expert_group_with_cp_ranks.append(ranks)
            for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):
                all_tp_ep_group_ranks.append(ranks)
            for ranks in rank_generator.get_ranks('ep', independent_ep=True):
                all_ep_group_ranks.append(ranks)
        self_vdp_mod_ep_ranks = None
        self_vdp_mod_ep_with_cp = None
        self_vdp_tp_ep_ranks = None
        self_vdp_ep_ranks = None
        # find all pp ranks include same model-partition, because different virtual data parallel ranks have the same virtual data parallel rank group, here only considers the virtual data parallel rank 0
        for pp_rank in range(args.pipeline_model_parallel_size // args.virtual_data_parallel_size):
            pp_ranks = [(pp_rank - offset_unit * j + args.pipeline_model_parallel_size) % args.pipeline_model_parallel_size for j in range(args.virtual_data_parallel_size // 2)] # [0, 4]
            dual_pp_rank = args.pipeline_model_parallel_size - 1 - pp_rank
            dual_pp_ranks = [(dual_pp_rank - offset_unit * j + args.pipeline_model_parallel_size) % args.pipeline_model_parallel_size for j in range(args.virtual_data_parallel_size // 2)] # [3, 7]
            vdp_pp_ranks = pp_ranks + dual_pp_ranks
            # find all dp groups in above pp groups
            for dp_innner_rank in range(num_dp_groups_per_stage):
                vdp_group_ranks = []
                for k in vdp_pp_ranks:
                    vdp_group_ranks += all_data_parallel_group_ranks[k * num_dp_groups_per_stage + dp_innner_rank]
                group = torch.distributed.new_group(vdp_group_ranks, timeout=timeout)
                if global_rank in vdp_group_ranks:
                    print("d log : 111")
                    self_vdp_groups = vdp_group_ranks
                    set_virtual_data_parallel_group(group)
            # find all dp_cp groups in above pp groups
            for dp_cp_inner_rank in range(num_dp_cp_groups_per_stage):
                vdp_with_cp_group_ranks = []
                for k in vdp_pp_ranks:
                    vdp_with_cp_group_ranks += all_data_parallel_group_ranks_with_cp[k * num_dp_cp_groups_per_stage + dp_cp_inner_rank]
                group = torch.distributed.new_group(vdp_with_cp_group_ranks, timeout=timeout)
                if global_rank in vdp_with_cp_group_ranks:  
                    self_vdp_cp_groups = vdp_with_cp_group_ranks
                    set_virtual_data_parallel_group(group, with_context_parallel=True)
            # if ep, find all dp/ep groups and dp-cp/ep groups in above pp groups
            if num_experts is not None:
                for dp_mod_ep_inner_rank in range(num_dp_modulo_ep_groups_per_stage):
                    vdp_mod_ep_ranks = []
                    for k in vdp_pp_ranks:
                        vdp_mod_ep_ranks += all_data_modulo_expert_group_ranks[k * num_dp_modulo_ep_groups_per_stage + dp_mod_ep_inner_rank]
                    group = torch.distributed.new_group(vdp_mod_ep_ranks, timeout=timeout)
                    if global_rank in vdp_mod_ep_ranks:
                        self_vdp_mod_ep_ranks = vdp_mod_ep_ranks
                        set_virtual_data_modulo_expert_parallel_group(group)
                for dp_mod_ep_with_cp_inner_rank in range(num_dp_cp_modulo_ep_groups_per_stage):
                    vdp_mod_ep_with_cp_ranks = []
                    for k in vdp_pp_ranks:
                        vdp_mod_ep_with_cp_ranks += all_data_modulo_expert_group_with_cp_ranks[k * num_dp_cp_modulo_ep_groups_per_stage + dp_mod_ep_with_cp_inner_rank]
                    group = torch.distributed.new_group(vdp_mod_ep_with_cp_ranks, timeout=timeout)
                    if global_rank in vdp_mod_ep_with_cp_ranks:
                        self_vdp_mod_ep_with_cp = vdp_mod_ep_with_cp_ranks
                        set_virtual_data_modulo_expert_parallel_group(group, with_context_parallel=True)
                for tp_ep_inner_rank in range(num_tp_ep_groups_per_stage):
                    vdp_tp_ep_ranks = []
                    for k in vdp_pp_ranks:
                        vdp_tp_ep_ranks += all_tp_ep_group_ranks[k * num_tp_ep_groups_per_stage + tp_ep_inner_rank]
                    group = torch.distributed.new_group(vdp_tp_ep_ranks, timeout=timeout)
                    if global_rank in vdp_tp_ep_ranks:
                        self_vdp_tp_ep_ranks = vdp_tp_ep_ranks
                        parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group
                for ep_inner_rank in range(num_ep_groups_per_stage):
                    vdp_ep_ranks = []
                    for k in vdp_pp_ranks:
                        vdp_ep_ranks += all_ep_group_ranks[k * num_ep_groups_per_stage + ep_inner_rank]
                    group = torch.distributed.new_group(vdp_ep_ranks, timeout=timeout)
                    if global_rank in vdp_ep_ranks:
                        self_vdp_ep_ranks = vdp_ep_ranks
                        parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

        print(f"[Rank {global_rank}]: virtual data parallel ranks {self_vdp_groups} | virtual data parallel with context parallel is {self_vdp_cp_groups} | virtual data modulo expert ranks is {self_vdp_mod_ep_ranks} | virtual data modulo expert with cp ranks is {self_vdp_mod_ep_with_cp} | tensor and expert parallel ranks is {self_vdp_tp_ep_ranks} | expert parallel ranks is {self_vdp_ep_ranks}")
            
        # find all embedding group
        offset_unit = args.pipeline_model_parallel_size // (args.virtual_data_parallel_size // 2)
        for vdp_rank in range(args.virtual_data_parallel_size // 2):
            if vdp_rank < args.virtual_data_parallel_size // 2:
                first_stage_rank = offset_unit * vdp_rank
                last_stage_rank = (vdp_rank * offset_unit + args.pipeline_model_parallel_size - 1) % args.pipeline_model_parallel_size
            else:
                dual_vdp_rank = vdp_rank - args.virtual_data_parallel_size // 2
                first_stage_rank = args.pipeline_model_parallel_size - 1 - offset_unit * dual_vdp_rank
                last_stage_rank = (-dual_vdp_rank * offset_unit + args.pipeline_model_parallel_size) % args.pipeline_model_parallel_size
            for pp_inner_rank in range(pp_group_size):
                first_stage_global_rank = first_stage_rank * pp_group_size + pp_inner_rank
                last_stage_global_rank = last_stage_rank * pp_group_size + pp_inner_rank
                group = torch.distributed.new_group([first_stage_global_rank, last_stage_global_rank], timeout=timeout)
                if global_rank in [first_stage_global_rank, last_stage_global_rank]:
                    print(f"[Rank {global_rank}]: embedding group ranks: {[first_stage_global_rank, last_stage_global_rank]}")
                    set_embedding_group(group)
                    set_embedding_ranks([first_stage_global_rank, last_stage_global_rank])


def is_pipeline_first_stage_wrapper(is_pipeline_first_stage):
    @wraps(is_pipeline_first_stage)
    def wrapper(ignore_virtual=False, vdp_rank: int = None):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            vdp_world_size = get_virtual_data_parallel_world_size()
            if vdp_world_size is not None and vdp_world_size > 1:
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                if vdp_rank is None:
                    vdp_rank = get_virtual_data_parallel_rank()
                pp_world_size = mpu.get_pipeline_model_parallel_world_size()
                offset_unit = pp_world_size // (vdp_world_size // 2)
                if vdp_rank < vdp_world_size // 2:
                    offset = offset_unit * vdp_rank
                else:
                    dual_vdp_rank = vdp_rank - (vdp_world_size // 2)
                    offset = pp_world_size - 1 - offset_unit * dual_vdp_rank
                return pp_rank == offset
            if (
                mpu.get_virtual_pipeline_model_parallel_world_size() is not None
                and mpu.get_virtual_pipeline_model_parallel_rank() != 0
            ):
                return False
        return mpu.get_pipeline_model_parallel_rank() == 0
    return wrapper


def is_pipeline_last_stage_wrapper(is_pipeline_last_stage):
    @wraps(is_pipeline_last_stage)
    def wrapper(ignore_virtual=False, vdp_rank: int = None):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            virtual_pipeline_model_parallel_world_size = (
                mpu.get_virtual_pipeline_model_parallel_world_size()
            )
            vdp_world_size = get_virtual_data_parallel_world_size()
            if vdp_world_size is not None and vdp_world_size > 1:
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                if vdp_rank is None:
                    vdp_rank = get_virtual_data_parallel_rank()
                pp_world_size = mpu.get_pipeline_model_parallel_world_size()
                offset_unit = pp_world_size // (vdp_world_size // 2)
                if vdp_rank < vdp_world_size // 2:
                    offset = (vdp_rank * offset_unit + pp_world_size - 1) % pp_world_size
                else:
                    dual_vdp_rank = vdp_rank - vdp_world_size // 2
                    offset = (-dual_vdp_rank * offset_unit + pp_world_size) % pp_world_size
                return pp_rank == offset
            if (
                virtual_pipeline_model_parallel_world_size is not None
                and mpu.get_virtual_pipeline_model_parallel_rank()
                != (virtual_pipeline_model_parallel_world_size - 1)
            ):
                return False
        return mpu.get_pipeline_model_parallel_rank() == (mpu.get_pipeline_model_parallel_world_size() - 1)
    return wrapper


def get_data_parallel_group_wrapper(get_data_parallel_group):
    @wraps(get_data_parallel_group)
    def wrapper(with_context_parallel=False):
        return get_virtual_data_parallel_group(with_context_parallel=with_context_parallel)
    return wrapper


def get_embedding_group_wrapper(fn):
    @wraps(fn)
    def wrapper():
        return get_embedding_group()
    return wrapper


def is_rank_in_embedding_group_wrapper(is_rank_in_embedding_group):
    @wraps(is_rank_in_embedding_group)
    def wrapper(ignore_virtual=False):
        rank = torch.distributed.get_rank()
        ranks = get_embedding_ranks()
        if ranks is None:
            return False
        else:
            return rank in ranks
    return wrapper


def get_model_wrapper(get_model):
    @wraps(get_model)
    def wrapper(model_provider_func, model_type, wrap_with_ddp=True):
        def get_llm_model(model_provider_func):
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
            )
            this_model.model_type = model_type
            return this_model
        
        def get_mm_model(model_provider_func):
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = pre_process
            add_decoder = True
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder
            )
            this_model.model_type = model_type
            return this_model

        """Initialize virtual data parallel groups for chimera"""
        args = get_args()
        init_chimera_parallel_state(args)
        """Build the model."""
        args.model_type = model_type

        # Build model.
        vdp_world_size = args.virtual_data_parallel_size
        pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_world_size > 1 and vdp_world_size is not None:
            model = []
            for i in range(vdp_world_size):
                set_virtual_data_parallel_rank(i)
                if model_type == ModelType.encoder_and_decoder: # no use
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    add_encoder = True
                    add_decoder = True
                    if mpu.get_pipeline_model_parallel_world_size() > 1:
                        if args.pipeline_model_parallel_split_rank is None:
                            raise ValueError(
                                "Split rank needs to be specified for model with both encoder and decoder"
                            )
                        rank = mpu.get_pipeline_model_parallel_rank()
                        split_rank = args.pipeline_model_parallel_split_rank
                        world_size = mpu.get_pipeline_model_parallel_world_size()
                        pre_process = rank == 0 or rank == split_rank
                        post_process = (rank == (split_rank - 1)) or (
                                rank == (world_size - 1))
                        add_encoder = mpu.is_pipeline_stage_before_split()
                        add_decoder = mpu.is_pipeline_stage_after_split()
                    this_model = model_provider_func(
                        pre_process=pre_process,
                        post_process=post_process,
                        add_encoder=add_encoder,
                        add_decoder=add_decoder)
                    model.model_type = model_type
                else:
                    ## NOTE: multimodal, only support pipeline model parallel size = 1 for encoder part, and in the encoder rank, there must be at least one layer of decoder
                    if getattr(args, "multimodal", False):
                        this_model = get_mm_model(model_provider_func)
                    else:
                        this_model = get_llm_model(model_provider_func)
                model.append(this_model)
        else:
            if getattr(args, "multimodal", False):
                model = get_mm_model(model_provider_func)
            else:
                model = get_llm_model(model_provider_func)
        if not isinstance(model, list):
            model = [model]
        total_param_count = 0
        for module in model:
            param_count = sum(p.numel() for p in module.parameters())
            total_param_count += param_count
        print(f">>> total paramaters is {total_param_count}")

        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in model:
            for param in model_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)


        # GPU allocation.
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if getattr(args, "preserve_orig_param_dtype", False):
            model = [model_module for model_module in model]
        elif args.fp16 or args.bf16:
            model = [Float16Module(model_module, args) for model_module in model]

        if getattr(args, "multimodal", False):
            from pangu.training.utils import freeze_module
            model = freeze_module(model)

        if mpu.get_data_parallel_rank() == 0:
            print_rank_0('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            for model_module in model:
                for name, parameters in model_module.named_parameters():
                    print('{} : {} : {} : {}'.format(name, parameters.dtype, parameters.size(), parameters.requires_grad))

        if wrap_with_ddp:
            config = get_model_config(model[0])
            ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=args.ddp_average_in_collective)
            model = [DDP(config,
                        ddp_config,
                        model_chunk,
                        # Turn off the bucket_size, because there is no appropriate method to deal with the conflict between decouple_bw and overlap_grad_reduce in chimera schedule,
                        # current solution is that place all the param into a bucket, then manual call the start_sync function for each model chunk in correct time. 
                        disable_bucketing=True)
                    for (model_chunk_idx, model_chunk) in enumerate(model)]
            for model_chunk in model:
                model_chunk.config.decouple_bw = args.chimera_decouple_bw

            # Broadcast params from data parallel src rank to other data parallel ranks.
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        print(f"[Rank {mpu.get_pipeline_model_parallel_rank()}]: {model}")
        return model
    return wrapper


def build_pretraining_data_loader_wrapper(build_pretraining_data_loader):
    @wraps(build_pretraining_data_loader)
    def wrapper(dataset, consumed_samples):
        """Build dataloader given an input dataset."""

        if dataset is None:
            return None
        args = get_args()
        dp_rank = mpu.get_data_parallel_rank()
        dp_world_size = mpu.get_data_parallel_world_size()
        vdp_world_size = get_virtual_data_parallel_world_size()
        dataloaders = []
        for i in range(vdp_world_size):
            # Megatron sampler
            if args.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=args.micro_batch_size,
                    data_parallel_rank=dp_rank * vdp_world_size + i,
                    data_parallel_size=dp_world_size * vdp_world_size)
            elif args.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    dataset,
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=args.micro_batch_size,
                    data_parallel_rank=dp_rank * vdp_world_size + i,
                    data_parallel_size=dp_world_size * vdp_world_size,
                    data_sharding=args.data_sharding)
            elif args.dataloader_type == "external":
                # External dataloaders are passed through. User is expected to provide a
                # torch-compatible dataloader and define samplers, if needed.
                return dataset
            else:
                raise Exception('{} dataloader type is not supported.'.format(
                        args.dataloader_type))

            # Torch dataloader.
            loader = torch.utils.data.DataLoader(dataset,
                                            batch_sampler=batch_sampler,
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            persistent_workers=True if args.num_workers > 0 else False,
                                            )
            dataloaders.append(loader)
        if len(dataloaders) == 1:
            dataloaders = dataloaders[0]
        return dataloaders
    return wrapper


def build_train_valid_test_data_iterators_wrapper(build_train_valid_test_data_iterators):
    """Build pretraining data iterators."""

    @wraps(build_train_valid_test_data_iterators)
    def wrapper(build_train_valid_test_datasets_provider):
        def data_generator(loader):
            for x in loader:
                yield x
        args = get_args()

        # Build loaders.
        train_dataloader, valid_dataloader, test_dataloader = \
            build_train_valid_test_data_loaders(
                build_train_valid_test_datasets_provider)
        # Build iterators.
        dl_type = args.dataloader_type
        if dl_type not in ['single', 'cyclic', 'external']:
            raise ValueError(
                f"Invalid dl_type: {dl_type}. "
                "Expected one of: 'single', 'cyclic', 'external'"
            )

        def _get_iterator(dataloader_type, dataloader):
            """Return dataset iterator."""
            if dataloader_type == "single":
                if isinstance(dataloader, list):
                    return [iter(loader) for loader in dataloader]
                return iter(dataloader)
            elif dataloader_type == "cyclic":
                if isinstance(dataloader, list):
                    return [data_generator(loader) for loader in dataloader]
                return iter(data_generator(dataloader))
            elif dataloader_type == "external":
                # External dataloader is passed through. User is expected to define how to iterate.
                return dataloader
            else:
                raise RuntimeError("unexpected dataloader type")

        if train_dataloader is not None:
            train_data_iterator = _get_iterator(dl_type, train_dataloader)
        else:
            train_data_iterator = None

        if valid_dataloader is not None:
            valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
        else:
            valid_data_iterator = None

        if test_dataloader is not None:
            test_data_iterator = _get_iterator(dl_type, test_dataloader)
        else:
            test_data_iterator = None

        return train_data_iterator, valid_data_iterator, test_data_iterator
    return wrapper


def broadcast_params_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)
            if is_expert_parallel:
                data_parallel_group = get_virtual_data_modulo_expert_parallel_group(
                    with_context_parallel=True
                )
            else:
                data_parallel_group = get_virtual_data_parallel_group(
                    with_context_parallel=True
                )
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )
        return wrapper


def linear_backward_wgrad_detach_wrapper(fn):
    def wrapper(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel and not WeightGradStore.is_decoupleBlock:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input_.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input_.dtype, "mpu"
                )
                handle = torch.distributed._all_gather_base(
                    all_gather_buffer, input_, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input_
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute and not WeightGradStore.is_decoupleBlock:
            handle.wait()

        if wgrad_compute and not WeightGradStore.is_decoupleBlock:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            if ctx.allreduce_dgrad:
                raise RuntimeError(
                    "Allreduce_dgrad should be none"
                )
            dim_size = list(input_.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input_.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation


        if WeightGradStore.is_decoupleBlock:
            WeightGradStore.put(
                total_input.clone().detach(),
                grad_output.clone().detach(),
                weight,
                ctx.sequence_parallel,
                in_row=not ctx.sequence_parallel
            )
            if hasattr(weight, 'grad_added_to_main_grad') and get_args().overlap_grad_reduce:
                weight.skip_grad_accum = True
            grad_weight = None
        else:
            if ctx.gradient_accumulation_fusion:
                if wgrad_compute:
                    if weight.main_grad.dtype == torch.float32:
                        from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
                        npu_matmul_add_fp32(total_input, grad_output, weight.main_grad)
                    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                        raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

                if hasattr(weight, 'grad_added_to_main_grad'):
                    # When overlap_grad_reduce is True, need to ensure that backward hooks
                    # are all run on the main backprop thread to prevent deadlocks. Setup
                    # dummy grad_weight tensor to prevent backward hooks from being run
                    # in a background thread.
                    if getattr(weight, 'zero_out_wgrad', False):
                        grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input_.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        grad_weight = torch.empty(
                            weight.main_grad.shape,
                            dtype=input_.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    weight.grad_added_to_main_grad = True
                else:
                    grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
    return wrapper


def make_param_hook_wrapper(fn):
    @wraps(fn)
    def wrapper(
            self,
            param: torch.nn.Parameter,
            param_to_buffer,
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):
            # When enable WeightGradStore, the backward will return a None grad_weight, this hook will cause a bug, disable the hook, 
            # and call the start_sync to achieve the async allreduce 
            if not getattr(param, 'skip_grad_accum', False):
                if param.requires_grad:
                    if self.ddp_config.overlap_grad_reduce:
                        if param.grad is None:
                            raise RuntimeError('param.grad being None is not safe when overlap_grad_reduce is True')
                    if param.grad is not None and (
                            not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                    ):
                        param.main_grad.add_(param.grad.data)
                    param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    param_to_buffer[param].register_grad_ready(param)
                if getattr(param, 'skip_grad_accum', False):
                    param.skip_grad_accum = False
        return param_hook
    return wrapper
