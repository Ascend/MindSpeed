# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class AdaptiveContextParallelFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('context-parallel-size')

    def register_args(self, parser: ArgumentParser):
        self.add_parser_argument_choices_value(
            parser,
            "--context-parallel-algo",
            'adaptive_cp_algo'
        )
        group = parser.add_argument_group(title=self.feature_name)

        # adaptive context parallel
        group.add_argument('--attention-mask-on-cpu', action='store_true',
                           help='store full attention mask on CPU instead of NPU')
        group.add_argument('--adaptive-cp-without-coarse', action='store_true',
                           help='does not coarse the attention mask in adaptive_cp feature, only recommended when full'
                                'sequence length is less than 8K and dynamic attention mask is not feasible')
        group.add_argument('--adaptive-cp-dynamic-attn-mask', action='store_true',
                           help='if the attention mask is dynamic across batches')
        group.add_argument('--adaptive-cp-only-reschedule', action='store_true',
                           help='not apply remapping but only rescheduling process in adaptive-cp feature')
        group.add_argument('--adaptive-cp-manually-set-mask-list', action='store_true',
                           help='manually set pre-cooked attention mask list')


    def validate_args(self, args):
        # adaptive context parallel
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'adaptive_cp_algo':
            if args.seq_length % args.context_parallel_size != 0:
                raise AssertionError("sequence length must be divisible by context_parallel_size")
            args.use_flash_attn = True
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
            if args.ulysses_degree_in_cp is None:
                raise AssertionError("--ulysses-degree-in-cp must be specified in hybrid_adaptive_cp_algo")
            ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
            if not (ring_degree > 1 and remainder == 0):
                raise AssertionError("--ulysses-degree-in-cp must be devisible by --context-parallel-size")
            head, remainder = divmod(args.num_attention_heads,
                                     args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
            if not (head >= 1 and remainder == 0):
                raise AssertionError("num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp")
            if args.seq_length % args.context_parallel_size != 0:
                raise AssertionError("sequence length must be divisible by context_parallel_size in hybrid cp")
            args.use_flash_attn = True