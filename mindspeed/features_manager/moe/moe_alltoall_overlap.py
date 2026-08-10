# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
from functools import wraps
from mindspeed.features_manager.feature import MindSpeedFeature


def alltoall_overlap_ddp_init_wrapper(fn):
    """Register DDP post hooks for only the TE experts whose wgrad is delayed."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if self.ddp_config.delay_wgrad_compute:
            raise RuntimeError(
                'MindSpeed alltoall overlap owns the TE expert delayed-wgrad '
                'schedule and cannot be combined with global DDP '
                'delay_wgrad_compute.'
            )

        marker = '_mindspeed_alltoall_overlap_delayed_wgrad_hook_registered'
        for module in self.module.modules():
            module_config = getattr(module, 'config', None)
            if not (
                getattr(module_config, 'moe_alltoall_overlap_comm', False)
                and getattr(module, 'delay_wgrad_compute', False)
                and hasattr(module, 'register_wgrad_accumulation_and_reduce_hooks')
            ):
                continue

            for param in module.parameters(recurse=False):
                if not (
                    param.requires_grad
                    and getattr(param, 'skip_backward_post_hook', False)
                    and getattr(param, marker, None) != id(self)
                ):
                    continue

                ddp_post_hook = self._make_backward_post_hook(param)

                def delayed_wgrad_post_hook(
                    param=param,
                    module=module,
                    ddp_post_hook=ddp_post_hook,
                ):
                    # With gradient-accumulation fusion, TE has already updated
                    # main_grad in-place and deliberately leaves param.grad as
                    # None. Megatron's regular hook requires a non-None grad
                    # when overlap_grad_reduce is enabled, so register the
                    # bucket directly instead of manufacturing a dummy wgrad.
                    if param.grad is None and getattr(module, 'fuse_wgrad_accumulation', False):
                        if not hasattr(param, 'main_grad'):
                            raise RuntimeError('TE delayed fused wgrad completed without param.main_grad.')
                        param.grad_added_to_main_grad = True
                        if self.ddp_config.overlap_grad_reduce and param in self.param_to_bucket_group:
                            self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)
                        return
                    ddp_post_hook()

                module.register_wgrad_accumulation_and_reduce_hooks(delayed_wgrad_post_hook)
                setattr(param, marker, id(self))
        return result

    return wrapper


class MoEAlltoAllOverLapFeature(MindSpeedFeature):
    '''
    MoE Layer AllToAll or alltoall_seq OverLap spec.
    This spec supports "alltoall" and "alltoall_seq" dispatcher.
    '''

    def __init__(self):
        super().__init__('moe-alltoall-overlap-comm', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--moe-alltoall-overlap-comm',
            action='store_true',
            default=False,
            help='Use async communication&swap to overlap compute in alltoall or alltoall_seq. In alltoall dispatcher, \
                           if with share_expert, will open `--moe-shared-expert-overlap` automatically.',
        )

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        if args.moe_alltoall_overlap_comm and args.moe_token_dispatcher_type not in ('alltoall', 'alltoall_seq'):
            raise AssertionError(
                '`--moe-alltoall-overlap-comm` only support with `--moe-token-dispatcher-type alltoall` or `--moe-token-dispatcher-type alltoall_seq`.'
            )
        if args.moe_alltoall_overlap_comm:
            if args.expert_model_parallel_size == 1:
                raise AssertionError(
                    '`--moe-alltoall-overlap-comm` only support with `--expert-model-parallel-size` > 1.'
                )

            if args.moe_token_dispatcher_type == 'alltoall':  # nosec B105
                if not args.moe_grouped_gemm:
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-grouped-gemm`.'
                    )
                if getattr(args, 'delay_wgrad_compute', False):
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` owns the TE expert '
                        'delayed-wgrad schedule and cannot be combined with '
                        'global `delay_wgrad_compute`.'
                    )
                if getattr(args, 'overlap_dispatch_backward_with_experts_wgrad', False):
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` cannot be combined with '
                        '`overlap_dispatch_backward_with_experts_wgrad` because '
                        'both schedules call TE expert backward_dw().'
                    )
                if getattr(args, 'overlap_moe_expert_parallel_comm', False):
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` cannot be combined with '
                        'Megatron `overlap_moe_expert_parallel_comm`; use only '
                        'the MindSpeed AlltoAll overlap scheduler.'
                    )
                if args.moe_zero_memory == 'level1':
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` with the `alltoall` dispatcher '
                        'and TEGroupedMLP currently supports `--moe-zero-memory` '
                        '`disable` or `level0`, but not `level1`.'
                    )
                if args.moe_tp_extend_ep:
                    raise AssertionError(
                        '`alltoall` not support `--moe-tp-extend-ep` for now. With`--moe-tp-extend-ep`, the dispatcher should be `alltoall_seq`.'
                    )
                if (
                    args.n_shared_experts or args.moe_shared_expert_intermediate_size
                ) and not args.moe_shared_expert_overlap:
                    args.moe_shared_expert_overlap = True
                    print('Warning: with `alltoall` dispatcher and share_expert, open `--moe-shared-expert-overlap`.')

            elif args.moe_token_dispatcher_type == 'alltoall_seq':  # nosec B105
                if not args.moe_permutation_async_comm:
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` with `alltoall_seq` dispatcher needs `--moe-permutation-async-comm`.'
                    )
                if not args.moe_grouped_gemm:
                    raise AssertionError(
                        '`--moe-alltoall-overlap-comm` with `alltoall_seq` dispatcher needs `--moe-grouped-gemm`.'
                    )
                if not args.moe_tp_extend_ep and args.moe_alltoall_overlap_comm and args.tensor_model_parallel_size > 1:
                    raise AssertionError(
                        '`When tp > 1, --moe-alltoall-overlap-comm` with `alltoall_seq` needs `moe_tp_extend_ep`.'
                    )

            # Convert Megatron Shared_experts to MindSpeed version. This convert operation only for some judge.
            if args.n_shared_experts is None and args.moe_shared_expert_intermediate_size is not None:
                args.n_shared_experts = args.moe_shared_expert_intermediate_size // (
                    args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
                )

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import (
            MindSpeedAlltoAllOverlapMoeLayerAdaptor,
            MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor,
        )
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
            mlp_init,
            parallel_transformer_layer_init_wrapper,
            core_mlp_forward_wrapper,
        )

        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward', core_mlp_forward_wrapper)
        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_alltoall_overlap_comm:
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                parallel_transformer_layer_init_wrapper,
            )
            if args.moe_token_dispatcher_type == 'alltoall':  # nosec B105
                patch_manager.register_patch(
                    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                    alltoall_overlap_ddp_init_wrapper,
                )
                patch_manager.register_patch(
                    'megatron.core.transformer.moe.moe_layer.MoELayer', MindSpeedAlltoAllOverlapMoeLayerAdaptor
                )
            elif args.moe_token_dispatcher_type == 'alltoall_seq':  # nosec B105
                patch_manager.register_patch(
                    'megatron.core.transformer.moe.moe_layer.MoELayer', MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor
                )
