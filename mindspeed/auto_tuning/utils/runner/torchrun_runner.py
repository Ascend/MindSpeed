import os
import sys
import subprocess

from megatron.training import get_args

from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.utils.runner.irunner import _Env, IRunner

_AUTO_TUNING_ARGS = "--auto-tuning"
_logger = get_logger("runner")


class TorchRunRunner(IRunner):

    def __init__(self) -> None:
        super().__init__()

    def get_base_env(self) -> _Env:
        return os.environ.copy()

    def run(self, env: _Env) -> int:

        args = get_args()
        argv: list = sys.argv[1:]
        auto_tuning_filter_args_switch = ["--use-ascend-mc2", "--swap-attention",
                                          "--ampipe-tp-sp-comm-overlap",
                                          "--use-pipe-experts", "--pipe-experts-multi-stream",
                                          "--recompute-in-advance", "--recompute-in-bubble", "--use-nanopipe"]
        auto_tuning_filter_args_config = ["--ampipe-degree", "--pipe-experts-multi-data"]

        if _AUTO_TUNING_ARGS in sys.argv:
            argv[argv.index("--tensor-model-parallel-size") + 1] = '8'
            argv[argv.index("--pipeline-model-parallel-size") + 1] = '1'
            argv[argv.index("--context-parallel-size") + 1] = '1'
            if "--num-layers-per-virtual-pipeline-stage" in argv:
                vpp_index = argv.index("--num-layers-per-virtual-pipeline-stage")
                argv.pop(vpp_index + 1)
                argv.pop(vpp_index)
            if "--expert-model-parallel-size" in argv:
                argv[argv.index("--expert-model-parallel-size") + 1] = '1'
            if "--use-ascend-mc2" in argv:
                argv.pop(argv.index("--use-ascend-mc2"))
            for feature_args in auto_tuning_filter_args_switch:
                if feature_args in argv:
                    argv.pop(argv.index(feature_args))
            for feature_args in auto_tuning_filter_args_config:
                if feature_args in argv:
                    args_index = argv.index(feature_args)
                    argv.pop(args_index + 1)
                    argv.pop(args_index)

        while _AUTO_TUNING_ARGS in argv:
            pos = argv.index(_AUTO_TUNING_ARGS)
            argv.pop(pos)

        command = [
            'torchrun',
            '--nproc_per_node', str(args.nproc_per_node),
            '--nnodes', str(args.nnodes),
            '--node-rank', str(args.node_rank),
            '--master_addr', str(args.master_addr),
            '--master_port', str(args.master_port),
            str(sys.argv[0])
        ] + argv
        process = subprocess.Popen(command, shell=False, preexec_fn=lambda: os.setpgrp(), env=env)
        process.wait()
        returncode = process.returncode
        _logger.info(returncode)

        return returncode
