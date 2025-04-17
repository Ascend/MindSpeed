"""
环境变量参数配置工具
"""
from typing import List, Optional, Any


def update_param(argv: List[str], arg_name: str, arg_value: Optional[Any]):
    while arg_name in argv:
        i = argv.index(arg_name)
        argv.pop(i + 1)
        argv.pop(i)

    if arg_value:
        argv.extend((arg_name, str(arg_value)))


def update_flag(argv: List[str], arg_name: str, switch: bool):
    while arg_name in argv:
        argv.remove(arg_name)

    if switch:
        argv.append(arg_name)