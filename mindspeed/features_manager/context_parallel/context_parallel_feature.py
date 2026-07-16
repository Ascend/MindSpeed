# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature


class ContextParallelFeature(MindSpeedFeature):
    _CP_ALGO_TO_COMM_TYPE = {
        'megatron_cp_algo': 'p2p',
        'ulysses_cp_algo': 'a2a',
        'kvallgather_cp_algo': 'all_gather',
        'hybrid_cp_algo': 'a2a+p2p',
    }
    _CP_COMM_TYPE_TO_ALGO = {
        'p2p': 'megatron_cp_algo',
        'a2a': 'ulysses_cp_algo',
        'all_gather': 'kvallgather_cp_algo',
        'allgather': 'kvallgather_cp_algo',
        'a2a+p2p': 'hybrid_cp_algo',
    }

    def __init__(self):
        super().__init__('context-parallel-size')

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return (
            self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, 1)
        ) or self.default_patches

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--context-parallel-algo',
            type=str,
            default=None,
            choices=list(self._CP_ALGO_TO_COMM_TYPE),
            help=(
                'Deprecated MindSpeed CP spelling. It is translated to Megatron '
                '--cp-comm-type and takes precedence when both spellings are supplied.'
            ),
        )

        group.add_argument(
            '--attention-mask-type',
            type=str,
            default='causal',
            choices=['causal', 'general'],
            help='context parallel attention mask type',
        )
        group.add_argument(
            '--use-cp-send-recv-overlap', action='store_true', help='use this flag to enable cp send-recv-overlap.'
        )
        group.add_argument('--cp-window-size', type=int, default=1)

    @classmethod
    def _sync_context_parallel_algo_to_cp_comm_type(cls, args):
        is_args_namespace = isinstance(args, Namespace)
        current_cp_comm_type = getattr(args, 'cp_comm_type', None)
        legacy_algo = getattr(args, 'context_parallel_algo', None)

        if legacy_algo is not None:
            legacy_cp_comm_type = cls._CP_ALGO_TO_COMM_TYPE[legacy_algo]
            # Megatron 0.17 assigns ``['p2p']`` before MindSpeed can inspect
            # parser provenance, so an old MindSpeed option must take priority
            # over that default. Users should not provide both spellings.
            args.cp_comm_type = [legacy_cp_comm_type] if is_args_namespace else legacy_cp_comm_type
            return

        if current_cp_comm_type is None:
            args.cp_comm_type = ['p2p'] if is_args_namespace else 'p2p'
            args.context_parallel_algo = 'megatron_cp_algo'
            return

        current_values = current_cp_comm_type if isinstance(current_cp_comm_type, list) else [current_cp_comm_type]
        canonical_algos = {cls._CP_COMM_TYPE_TO_ALGO.get(value, value) for value in current_values}
        if len(canonical_algos) == 1:
            args.context_parallel_algo = next(iter(canonical_algos))
        else:
            # Heterogeneous generic CP is owned by Megatron/TENPU.  EOD rejects
            # it during validation because one batch cannot use two layouts.
            args.context_parallel_algo = None

    def pre_validate_args(self, args):
        self._sync_context_parallel_algo_to_cp_comm_type(args)

    def validate_args(self, args):
        from mindspeed.core.context_parallel.model_parallel_utils import (
            UNSUPPORTED_CP,
            get_cp_backend_route,
            get_resolved_cp_comm_type,
        )

        if args.context_parallel_size <= 1:
            return

        if args.context_parallel_algo in ('adaptive_cp_algo', 'hybrid_adaptive_cp_algo'):
            raise AssertionError(
                'MindSpeed adaptive CP has been retired; use --cp-comm-type p2p, a2a, all_gather, or a2a+p2p.'
            )

        cp_comm_type = get_resolved_cp_comm_type(args)
        if getattr(args, 'reset_attention_mask', False) and cp_comm_type is None:
            raise AssertionError('EOD CP currently requires one homogeneous --cp-comm-type for all layers.')

        route = get_cp_backend_route(args)
        if route == UNSUPPORTED_CP:
            raise AssertionError(
                'Unsupported CP capability cell: '
                f'reset_attention_mask={getattr(args, "reset_attention_mask", False)}, '
                f'cp_comm_type={getattr(args, "cp_comm_type", None)}, '
                f'attention_mask_type={getattr(args, "attention_mask_type", None)}, '
                f'transformer_impl={getattr(args, "transformer_impl", None)}. '
                'TENPU support is not available and no MindSpeed EOD-only fallback exists.'
            )

        if getattr(args, 'use_fused_ring_attention_update', False):
            raise AssertionError(
                'TENPU does not yet provide fused ring-attention update; '
                '--use-fused-ring-attention-update is unavailable.'
            )

        cp_window_size = int(getattr(args, 'cp_window_size', 1))
        use_cp_send_recv_overlap = bool(getattr(args, 'use_cp_send_recv_overlap', False))
        if cp_window_size != 1 or use_cp_send_recv_overlap:
            if cp_comm_type not in {'p2p', 'a2a+p2p'}:
                raise AssertionError(
                    '--cp-window-size and --use-cp-send-recv-overlap are supported only with '
                    '--cp-comm-type p2p or a2a+p2p.'
                )
            ring_cp_size = args.context_parallel_size
            if cp_comm_type == 'a2a+p2p':
                hierarchy = getattr(args, 'hierarchical_context_parallel_sizes', None)
                if hierarchy:
                    if len(hierarchy) != 2:
                        raise AssertionError(
                            'a2a+p2p requires two --hierarchical-context-parallel-sizes: '
                            '[a2a_degree, p2p_degree].'
                        )
                    ring_cp_size = hierarchy[1]
                else:
                    ulysses_degree = getattr(args, 'ulysses_degree_in_cp', None)
                    if ulysses_degree is None or ulysses_degree <= 0:
                        raise AssertionError(
                            'a2a+p2p requires --hierarchical-context-parallel-sizes or '
                            '--ulysses-degree-in-cp when enabling CP window/overlap.'
                        )
                    if args.context_parallel_size % ulysses_degree != 0:
                        raise AssertionError(
                            '--context-parallel-size must be divisible by --ulysses-degree-in-cp.'
                        )
                    ring_cp_size = args.context_parallel_size // ulysses_degree

            if not 1 <= cp_window_size < ring_cp_size:
                raise AssertionError(
                    '--cp-window-size must be in [1, ring context-parallel size) for p2p CP.'
                )
            if ring_cp_size % cp_window_size != 0:
                raise AssertionError(
                    'ring context-parallel size must be divisible by --cp-window-size for p2p CP.'
                )

    def register_patches(self, patch_manager, args):
        """Pass MindSpeed double-ring options to the TENPU attention instance."""
        # ``register_patches`` is called once while MindSpeed is imported, before
        # Megatron registers and type-converts its own arguments.  Therefore
        # command-line values such as ``--context-parallel-size 2`` can be the
        # string ``'2'`` here.  Normalize only the values used for this early
        # feature gate; Megatron remains the authoritative parser/validator.
        try:
            context_parallel_size = int(getattr(args, 'context_parallel_size', 1))
            cp_window_size = int(getattr(args, 'cp_window_size', 1))
        except (TypeError, ValueError):
            # Leave malformed values to Megatron's regular argument parser.
            return

        if (
            context_parallel_size > 1
            and (
                cp_window_size != 1
                or getattr(args, 'use_cp_send_recv_overlap', False)
            )
        ):
            from mindspeed.core.context_parallel.tenpu_adaptor import te_dot_product_attention_init_wrapper

            patch_manager.register_patch(
                'megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__',
                te_dot_product_attention_init_wrapper,
            )
