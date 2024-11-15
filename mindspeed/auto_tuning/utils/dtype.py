from enum import Enum


class DTYPE(Enum):
    fp16 = ("fp16", 2)
    fp32 = ("fp32", 4)
    bf16 = ("bf16", 2)
