# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
from functools import wraps
from inspect import signature

from megatron.core.transformer.cuda_graphs import is_graph_capturing

from mindspeed.features_manager.feature import MindSpeedFeature


def _get_alltoall_overlap_delayed_wgrad_params(model):
    """Map each AllToAll-overlap delayed-wgrad parameter to its TE linear."""
    from mindspeed.core.transformer.moe.moe_feature.adaptor import (
        MindSpeedAlltoALLOverLapGmmExperts,
    )

    delayed_params = {}
    for experts in model.modules():
        if not isinstance(experts, MindSpeedAlltoALLOverLapGmmExperts):
            continue

        for module in (experts.linear_fc1, experts.linear_fc2):
            if not (
                hasattr(module, 'need_backward_dw')
                and module.need_backward_dw()
                and hasattr(module, 'register_wgrad_accumulation_and_reduce_hooks')
            ):
                raise RuntimeError(
                    f'MindSpeed AllToAll overlap requires delayed-wgrad hooks on {type(module).__name__}.'
                )

            module_params = [
                param
                for param in module.parameters(recurse=False)
                if param.requires_grad and getattr(param, 'skip_backward_post_hook', False)
            ]
            if not module_params:
                raise RuntimeError(
                    f'MindSpeed AllToAll overlap found no delayed-wgrad parameters on {type(module).__name__}.'
                )
            for param in module_params:
                if param in delayed_params:
                    raise RuntimeError('A delayed-wgrad parameter belongs to multiple TE linears.')
                delayed_params[param] = module

    return delayed_params


def _make_alltoall_overlap_delayed_wgrad_hook(self, param, module, ddp_post_hook):
    """Run the DDP post hook only after TENPU has completed delayed wgrad."""

    def hook(*unused):
        if is_graph_capturing():
            return

        if param.grad is not None:
            ddp_post_hook()
            return

        if not getattr(module, 'fuse_wgrad_accumulation', False):
            raise RuntimeError(
                'TENPU delayed wgrad completed without param.grad while gradient-accumulation fusion is disabled.'
            )
        if not hasattr(param, 'main_grad'):
            raise RuntimeError('TENPU delayed fused wgrad completed without param.main_grad.')
        if param not in self.param_to_bucket_group:
            raise RuntimeError('TENPU delayed-wgrad parameter is not assigned to a DDP bucket.')

        # TENPU has already accumulated this wgrad directly into main_grad.
        # Calling Megatron's regular hook would either assert on param.grad or
        # add the gradient a second time, so only publish readiness here.
        param.grad_added_to_main_grad = True
        if self.ddp_config.overlap_grad_reduce:
            self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)

    return hook


def _make_alltoall_overlap_autograd_hook(param):
    """Ignore the early AccumulateGrad callback for a delayed TE wgrad."""

    def hook(*unused):
        if is_graph_capturing():
            return

        # A delayed GroupedLinear returns no parameter gradient from its
        # autograd backward. The real wgrad is produced later by backward_dw(),
        # so DDP must not publish this parameter at the AccumulateGrad boundary.
        if param.grad is None:
            return

        raise RuntimeError(
            'TENPU marked this expert parameter for delayed wgrad, but its '
            'autograd callback received param.grad before backward_dw().'
        )

    return hook


def alltoall_overlap_ddp_init_wrapper(fn):
    """Register DDP hooks at the TENPU delayed-wgrad completion boundary."""

    fn_signature = signature(fn)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound_args = fn_signature.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        ddp_config = bound_args.arguments['ddp_config']
        model = bound_args.arguments['module']

        if ddp_config.delay_wgrad_compute:
            raise RuntimeError(
                'MindSpeed alltoall overlap owns the TE expert delayed-wgrad '
                'schedule and cannot be combined with global DDP '
                'delay_wgrad_compute.'
            )

        delayed_params = _get_alltoall_overlap_delayed_wgrad_params(model)
        if not delayed_params:
            return fn(*bound_args.args, **bound_args.kwargs)

        unexpected_delayed_params = [
            param
            for param in model.parameters()
            if getattr(param, 'skip_backward_post_hook', False) and param not in delayed_params
        ]
        if unexpected_delayed_params:
            raise RuntimeError(
                'MindSpeed AllToAll overlap found delayed-wgrad parameters outside its TEGroupedMLP experts.'
            )

        original_make_backward_post_hook = self._make_backward_post_hook
        registered_params = set()

        def make_backward_post_hook(param):
            module = delayed_params.get(param)
            if module is None:
                return original_make_backward_post_hook(param)
            if param in registered_params:
                raise RuntimeError('A TENPU delayed-wgrad DDP hook was registered more than once.')
            registered_params.add(param)
            return _make_alltoall_overlap_autograd_hook(param)

        # Let Megatron register its normal AccumulateGrad callbacks, but replace
        # only the delayed expert callbacks with an early no-op. TENPU does not
        # produce their real gradients until MindSpeed calls backward_dw().
        had_instance_hook_factory = '_make_backward_post_hook' in self.__dict__
        instance_hook_factory = self.__dict__.get('_make_backward_post_hook')
        self._make_backward_post_hook = make_backward_post_hook
        try:
            result = fn(*bound_args.args, **bound_args.kwargs)
        finally:
            if had_instance_hook_factory:
                self._make_backward_post_hook = instance_hook_factory
            else:
                self.__dict__.pop('_make_backward_post_hook', None)

        if registered_params != set(delayed_params):
            missing = len(set(delayed_params) - registered_params)
            raise RuntimeError(
                'Megatron DDP did not register every TENPU delayed-wgrad parameter '
                f'for AllToAll overlap: {missing} parameter(s) are missing.'
            )

        # Publish gradient readiness only from TENPU's actual delayed-wgrad
        # completion boundary. This hook is distinct from the early autograd
        # callback above, so fused and non-fused accumulation remain unambiguous.
        for param, module in delayed_params.items():
            module.register_wgrad_accumulation_and_reduce_hooks(
                _make_alltoall_overlap_delayed_wgrad_hook(
                    self,
                    param,
                    module,
                    original_make_backward_post_hook(param),
                )
            )
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
