import torch_npu


def npu_rotary_position_embedding(contiguous_tensor, cos_, sin_, mode):
    return torch_npu.npu_rotary_position_embedding(contiguous_tensor, cos_, sin_, mode)