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


def compile_extended_pattern(pattern: str):
    """
    Convert extended pattern (e.g., "*.layer.{0-4}") into a regex and range specs.
    Returns: (compiled_regex, specs)
      - specs: list of either (low, high) or None (for {*})
    """
    specs = []
    placeholder = "__NUM__"

    # Replace {*}, {a-b} with placeholder and record spec
    def replace_brace(match):
        inner = match.group(1)
        if inner == '*':
            specs.append(None)  # no range
            return placeholder
        elif '-' in inner:
            parts = inner.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid brace pattern: {inner}")
            try:
                low = int(parts[0])
                high = int(parts[1])
            except ValueError as e:
                raise ValueError(f"Non-integer in range: {inner}") from e
            if low > high:
                raise ValueError(f"Invalid range: {inner} (low > high)")
            specs.append((low, high))
            return placeholder
        else:
            raise ValueError(f"Unrecognized brace pattern: {inner}")

    # Match {...} but avoid matching escaped or invalid braces
    temp_pattern = re.sub(r'\{([^}]*)\}', replace_brace, pattern)

    # Convert fnmatch wildcards to regex manually
    regex_parts = []
    i = 0
    while i < len(temp_pattern):
        if temp_pattern.startswith(placeholder, i):
            regex_parts.append(r'(\d+)')  # capture digits
            i += len(placeholder)
        else:
            c = temp_pattern[i]
            if c == '*':
                regex_parts.append(r'.*')
            elif c == '?':
                regex_parts.append(r'.')
            elif c == '.':
                regex_parts.append(r'\.')
            else:
                regex_parts.append(re.escape(c))
            i += 1

    full_regex = '^' + ''.join(regex_parts) + '$'
    return re.compile(full_regex), specs


def module_name_match(pattern: str, string):
    """
    Match a string against an extended pattern.
    """
    regex, specs = compile_extended_pattern(pattern)
    m = regex.match(string)
    if not m:
        return False

    groups = m.groups()
    for num_str, spec in zip(groups, specs):
        try:
            num = int(num_str)
        except ValueError:
            return False

        # If a range is specified, check bounds
        if spec is not None:
            low, high = spec
            if not (low <= num <= high):
                return False
    return True
