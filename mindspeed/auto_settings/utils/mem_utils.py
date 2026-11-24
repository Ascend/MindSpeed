# Copyright 2026 Huawei Technologies Co., Ltd.
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

from typing import Callable, List


def mem_b_to_kb(n: float, d: int = 2) -> float:
    return round(n / pow(1024, 1), d)


def mem_kb_to_b(n: float, d: int = 2) -> float:
    return round(n * pow(1024, 1), d)


def mem_b_to_mb(n: float, d: int = 2) -> float:
    return round(n / pow(1024, 2), d)


def mem_mb_to_b(n: float, d: int = 2) -> float:
    return round(n * pow(1024, 2), d)


def mem_b_to_gb(n: float, d: int = 2) -> float:
    return round(n / pow(1024, 3), d)


def mem_gb_to_b(n: float, d: int = 2) -> float:
    return round(n * pow(1024, 3), d)


def mem_convert_list(ns: List[float], func: Callable[[float, int], float], d: int = 2) -> List[float]:
    return [func(n, d) for n in ns]
