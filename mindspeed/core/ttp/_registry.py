# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Shared registry to break cyclic imports between processor and exception_handler.

Both modules need access to the TTPProcessor singleton.  By placing
get_processor/set_processor in this module, both can import from here
without depending on each other.
"""

from typing import Optional

_processor = None


def set_processor(processor) -> None:
    """Register the TTPProcessor singleton (called once during init)."""
    global _processor
    _processor = processor


def get_processor() -> Optional[object]:
    """Return the registered TTPProcessor singleton, or None.

    Falls back to TTPProcessor.get_instance() if set_processor() hasn't
    been called yet (e.g., during early initialization or edge cases).
    """
    if _processor is not None:
        return _processor
    try:
        from ..comm.processor import TTPProcessor

        return TTPProcessor.get_instance()
    except ImportError:
        return None
