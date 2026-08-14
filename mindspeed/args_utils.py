# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Argument helpers re-exported from MegatronAdaptor.

The feature list is unified in MA's FeaturesManager. These symbols are re-exported
here to keep MindSpeed decoupled from MA's internal module layout, so that callers
can depend on the stable `mindspeed.args_utils` surface rather than MA's private paths.
"""

from megatron_adaptor.utils.args_utils import (
    add_args,
    parser_unknown_args,
    get_mindspeed_args,
    get_full_args,
    set_full_args,
)

__all__ = ['add_args', 'parser_unknown_args', 'get_mindspeed_args', 'get_full_args', 'set_full_args']
