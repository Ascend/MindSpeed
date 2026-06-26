import torch


def _to_step_int(step):
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)


def _distributed_group_rank(group):
    if hasattr(group, 'rank') and callable(group.rank):
        return group.rank()
    return torch.distributed.get_rank(group)
