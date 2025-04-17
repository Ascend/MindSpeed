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
