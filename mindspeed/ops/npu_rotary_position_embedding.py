import torch
from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder
rope_op_builder = RotaryPositionEmbeddingOpBuilder()


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    rope_ops = rope_op_builder.load()
    return rope_ops.npu_rotary_position_embedding(x, cos, sin, mode)
