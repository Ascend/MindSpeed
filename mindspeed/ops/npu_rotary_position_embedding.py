import torch_npu

__all__ = ["npu_rotary_position_embedding"]


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    if mode not in (0, 1):
        raise ValueError("mode must be 0 (half) or 1 (interleave)")
    return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="half" if mode == 0 else "interleave")
