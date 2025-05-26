# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import math
from argparse import Namespace

import torch


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
