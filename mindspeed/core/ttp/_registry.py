# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
