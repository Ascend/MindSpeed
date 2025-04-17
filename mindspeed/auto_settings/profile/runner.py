from typing import Dict, List
from argparse import Namespace
import os
import subprocess
import sys

from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.utils.logger import get_logger


class Runner:

    def __init__(self):
        system_config = get_system_config()
        self.nnodes: str = system_config.nnodes
        self.nproc_per_node: str = system_config.nproc_per_node
        self.node_rank: int = system_config.node_rank
        self.master_addr: str = system_config.master_addr
        self.master_port: int = system_config.master_port
        self._logger = get_logger("runner")

    def get_base_argv(self) -> List[str]:
        return sys.argv.copy()

    def get_base_env(self) -> Dict[str, str]:
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
        return_code = process.returncode
        self._logger.info("Last job returns %d.", return_code)

        return return_code
