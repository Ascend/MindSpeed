import torch
from einops import rearrange


def allgather_head_dim(data_ag, tp, tp_group, layout='BSND'):
    if layout == 'TND':
        data_ag = rearrange(data_ag, 't h d -> h t d')
    else:
        data_ag = rearrange(data_ag, 's b h d -> h s b d')
    data_ag_shape = list(data_ag.size())
    data_ag_shape[0] *= tp
    total_data_ag = torch.empty(data_ag_shape, dtype=data_ag.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(total_data_ag, data_ag.contiguous(), group=tp_group)
    if layout == 'TND':
        total_data_ag = rearrange(total_data_ag, 'h t d -> t h d')
    else:
        total_data_ag = rearrange(total_data_ag, 'h s b d -> s b h d')
    return total_data_ag
