# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    if numerator % denominator != 0:
        raise ValueError("{} is not divisible by {}".format(numerator, denominator))
    return numerator // denominator
