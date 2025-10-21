# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import os
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class AutoSettingsFeature(MindSpeedFeature):
    """
    auto settings
    """

    def __init__(self):
        super().__init__('auto-settings', optimization_level=2)

    def is_need_apply(self, args):
        return args.auto_settings or os.getenv("OOTB_OPTIMIZER_PARSE_ARGS", "FALSE") == "TRUE" \
            or os.getenv("OOTB_OPTIMIZER_PARSE_MODEL", "FALSE") == "TRUE" \
            or os.getenv("OOTB_OPTIMIZER_PROFILING", "FALSE") == "TRUE"

    def register_args(self, parser: ArgumentParser):
        self._add_auto_settings_args(parser)

    def validate_args(self, args):
        if args.auto_settings:
            if args.load or args.save:
                raise AssertionError('--save or --load not support when --auto-settings')

    def register_patches(self, pm, args):
        from mindspeed.training import pretrain
        pm.register_patch('megatron.training.training.pretrain', pretrain)

        from mindspeed.tokenizer import build_tokenizer_wrapper
        from mindspeed.core.training import pretrain_decorator, setup_model_and_optimizer_decorator
        pm.register_patch('megatron.training.training.pretrain', pretrain_decorator)
        pm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_decorator)
        pm.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer_wrapper)

    def _add_auto_settings_args(self, parser):
        group = parser.add_argument_group(title="auto_settings")

        group.add_argument(
            "--auto-settings",
            action="store_true",
            help="Enable auto settings."
        )
        group.add_argument(
            "--auto-settings-work-dir",
            type=str,
            default=os.getcwd(),
            help="Auto setting's working directory. By default current directory."
        )
        group.add_argument(
            "--auto-settings-ranks",
            type=int,
            default=8,
            help="The world size (# of ranks) for auto settings to search in."
        )
        group.add_argument(
            "--auto-settings-log-level",
            type=str,
            default="info",
            help="The world size (# of ranks) for auto settings to search in."
        )
        group.add_argument(
            "--target-nnodes",
            type=int,
            default=1,
            help="Target search nnodes for auto_settings."
        )
        group.add_argument(
            "--nnodes",
            type=int,
            default=1,
            help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>."
                 "Will be passed into torchrun."
        )
        group.add_argument(
            "--nproc-per-node",
            type=int,
            default=1,
            help="Number of workers per node; supported values: [auto, cpu, gpu, int]."
                 "Will be passed into torchrun."
        )
        group.add_argument(
            "--node-rank",
            type=int,
            default=0,
            help="Rank of the node for multi-node distributed training."
                 "Will be passed into torchrun."
        )
        group.add_argument(
            "--auto-settings-type",
            type=str,
            default="white",
            help="You should select one of [mixed, white, black]."
        )
        group.add_argument(
            "--prof-file",
            type=str,
            default=None,
            help=''
        )
        group.add_argument(
            "--master-addr",
            default="127.0.0.1",
            type=str,
            help="Address of the master node (rank 0) that only used for static rendezvous. It should "
                 "be either the IP address or the hostname of rank 0. For single node multi-proc training "
                 "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
                 "`[0:0:0:0:0:0:0:1]`."
                 "Will be passed into torchrun."
        )
        group.add_argument(
            "--master-port",
            default=29500,
            type=int,
            help="Port on the master node (rank 0) to be used for communication during distributed "
                 "training. It is only used for static rendezvous."
                 "Will be passed into torchrun."
        )

        # add default automated-pipeline-allocation params
        group.add_argument('--automated-pipeline',
                           default=False,
                           action='store_true',
                           help='To enable automated pipeline memory saving process'
                           )
        group.add_argument('--automated-pipeline-perf',
                           default=False,
                           action='store_true',
                           help='To enable automated pipeline performance acceleration process'
                           )
        group.add_argument('--recompute-module-list',
                           type=str, help='To store the recompute policy of automated pipeline'
                           )

        # add default jit params
        group.add_argument('--jit-compile', action='store_true', default=False,
                           help='Setting jit compile mode to True')
        return parser
