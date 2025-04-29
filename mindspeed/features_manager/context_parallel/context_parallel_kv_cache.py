# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ContextParallelKvCacheFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('context-parallel-kv-cache-policy')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        # context parallelism kv cache
        group.add_argument('--context-parallel-kv-cache-policy', type=str, default=None,
                           choices=['full', 'half'],
                           help='Selectivity cache K, V in process of cp.'
                                'Default is None, means not used cache K, V.'
                                'If para is full, cache all K, V.'
                                'If para is half, cache only K')
        group.add_argument('--context-parallel-cache-interval', type=int, default=0,
                           help='Set the interval of cache layers in cp.'
                                'Default is 0, means cache K, V in all layers.')
        group.add_argument('--use-ulysses-allgather-kv', action='store_true',
                           help='use this flag to enable allgather kv + repeat all2all q in ulysses cp.')


    def validate_args(self, args):
        # context parallelism kv cache
        if args.context_parallel_kv_cache_policy:
            if args.context_parallel_size == 1:
                raise AssertionError(
                    'context parallel size must larger than 1 when --context-parallel-kv-cache-policy is set.')
            if not args.use_flash_attn:
                raise AssertionError(
                    '--context-parallel-kv-cache-policy only support use flash attention.'
                )

        if args.context_parallel_cache_interval != 0:
            if not args.context_parallel_kv_cache_policy:
                raise AssertionError(
                    '--context-parallel-cache-interval only can be used when --context-parallel-kv-cache-policy is set.'
                )
            if args.context_parallel_cache_interval >= args.num_layers:
                raise AssertionError(
                    '--context-parallel-cache-interval should be smaller than the number of layers.'
                )
            if args.context_parallel_cache_interval < 0:
                raise AssertionError(
                    '--context-parallel-cache-interval cannot be negative number.'
                )

        if args.use_ulysses_allgather_kv:
            if args.context_parallel_size == 1:
                raise AssertionError(
                    'context parallel size must larger than 1 when --use-ulysses-allgather-kv is set.')
            if args.context_parallel_algo != 'ulysses_cp_algo':
                raise AssertionError(
                    '--context_parallel-algo should be ulysses_cp_algo when using --use-ulysses-allgather-kv.'
                )
            if not args.group_query_attention:
                raise AssertionError(
                    '--use-ulysses-allgather-kv needs to enable --group-query-attention.'
                )
