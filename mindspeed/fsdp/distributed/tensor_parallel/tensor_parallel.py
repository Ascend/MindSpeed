import fnmatch
import logging

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module

from mindspeed.fsdp.parallel_engine_config import TPPlanConfig
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def tensor_parallel_modules(model: torch.nn.Module, tp_mesh: DeviceMesh, tp_plan: TPPlanConfig):
    torch_tp_parallelize_plan = {}
    for name, _ in model.named_modules():
        parallel_style = get_parallel_style(name, tp_plan)
        if parallel_style is not None:
            print_rank(logger.debug, f'[TP]: Apply module <{name}> with parallel style {parallel_style}')
            torch_tp_parallelize_plan[name] = parallel_style()
    return parallelize_module(model, tp_mesh, torch_tp_parallelize_plan)


def get_parallel_style(module_name, tp_plan: TPPlanConfig):
    parallel_style = []
    if any([module_name_match(pattern, module_name) for pattern in tp_plan.colwise_parallel]):
        parallel_style.append(ColwiseParallel)
    elif any([module_name_match(pattern, module_name) for pattern in tp_plan.rowwise_parallel]):
        parallel_style.append(RowwiseParallel)
    elif any([module_name_match(pattern, module_name) for pattern in tp_plan.sequence_parallel]):
        parallel_style.append(SequenceParallel)

    if len(parallel_style) > 1:
        raise RuntimeError(f'[TP] More than one parallel style with {module_name}, pattern: {tp_plan}')
    elif len(parallel_style) == 1:
        return parallel_style[0]
    return None
