# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import re


def replace_first_segment_numbers(module_name):
    """Replaces the first occurrence of consecutive digits in a dot-separated string with '*'.

    This is particularly useful for creating pattern matches that ignore specific layer indices
    while preserving the rest of the module hierarchy.

    Args:
        module_name (str): The input module name string to process. Expected to be a dot-separated
            path representing a module hierarchy (e.g., 'model.layer.0.mlp.experts.0.up_proj').

    Returns:
        str: The modified string where the first segment containing digits has all consecutive
            digit sequences replaced by '*'. If no segments contain digits, the original string
            is returned unchanged.

    Examples:
        >>> replace_first_segment_numbers('model.layer.0.mlp.experts.0.up_proj')
        'model.layer.*.mlp.experts.0.up_proj'

        >>> replace_first_segment_numbers('encoder.block.12.attention.weight')
        'encoder.block.*.attention.weight'

        >>> replace_first_segment_numbers('just.text.without.digits')
        'just.text.without.digits'

        >>> replace_first_segment_numbers('layer1.conv2d.weights')
        'layer*.conv2d.weights'

    Note:
        The function only modifies the first segment that contains digits. Subsequent segments
        with digits remain unchanged. Each sequence of consecutive digits within the target
        segment is replaced by a single '*'.
    """
    parts = module_name.split('.')

    for i, part in enumerate(parts):
        if any(char.isdigit() for char in part):
            parts[i] = re.sub(r'\d+', '*', part)
            break

    return '.'.join(parts)
