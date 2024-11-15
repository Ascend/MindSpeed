from typing import Dict
import abc

_Env = Dict[str, str]


class IRunner(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if cls is IRunner:
            return hasattr(subclass, "get_base_env") and \
                hasattr(subclass, "run")
        return NotImplemented

    @abc.abstractmethod
    def get_base_env(self) -> _Env:
        return NotImplemented

    @abc.abstractmethod
    def run(self, env: _Env) -> int:
        return NotImplemented
