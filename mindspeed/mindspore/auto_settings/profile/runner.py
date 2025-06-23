import os
import subprocess
import sys
import random
import time
from typing import Dict, List
running_index = 0


def runner_run(
        self,
        modified_argv: List[str],
        modified_env: Dict[str, str]
) -> int:
    global running_index
    running_index = running_index + 1
    cmd = [
              "msrun",
              "--local_worker_num", str(self.nproc_per_node),
              "--worker_num", str(self.nproc_per_node * self.nnodes),
              "--node_rank", str(self.node_rank),
              "--master_addr", str(self.master_addr),
              "--master_port", str(self.master_port),
              "--log_dir", f"msrun_log_v{str(running_index)}",
              "--tail_worker_log", str(99),
              "--join", "True",
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
