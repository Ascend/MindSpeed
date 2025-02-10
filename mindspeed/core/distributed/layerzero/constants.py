# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

AUTO_CAST_INPUT = True
AUTO_CAST_OUTPUT = True


def set_auto_cast_input(state: bool):
    global AUTO_CAST_INPUT
    if not isinstance(state, bool):
        raise AssertionError("state must be a boolean")
    AUTO_CAST_INPUT = state


def set_auto_cast_output(state: bool):
    global AUTO_CAST_OUTPUT
    if not isinstance(state, bool):
        raise AssertionError("state must be a boolean")
    AUTO_CAST_OUTPUT = state