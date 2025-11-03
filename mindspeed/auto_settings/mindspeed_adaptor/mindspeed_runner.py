# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Dict, List
from argparse import Namespace
import os
import subprocess
import sys

from mindspeed.auto_settings.utils.logger import get_logger


class MindSpeedRunner:

    def __init__(self, args: Namespace):
        self.nnodes: str = args.nnodes
        self.nproc_per_node: str = args.nproc_per_node
        self.node_rank: int = args.node_rank
        self.master_addr: str = args.master_addr
        self.master_port: int = args.master_port
        self._logger = get_logger("runner")

    @staticmethod
    def get_base_argv() -> List[str]:
        return sys.argv.copy()
    
    @staticmethod
    def get_base_env() -> Dict[str, str]:
        return os.environ.copy()

    def run(
        self,
        modified_argv: List[str],
        modified_env: Dict[str, str]
    ) -> int:
        cmd = [
            "torchrun",
            "--nnodes", str(self.nnodes),
            "--nproc-per-node", str(self.nproc_per_node),
            "--node-rank", str(self.node_rank),
            "--master-addr", str(self.master_addr),
            "--master-port", str(self.master_port)
        ] + modified_argv
        self._logger.debug(f"Next job command: {cmd} with env {modified_env}")

        process = subprocess.Popen(
            cmd,
            preexec_fn=os.setpgrp,
            env=modified_env
        )
        process.wait()
        returncode = process.returncode
        self._logger.info("Last job returns %d.", returncode)

        return returncode
