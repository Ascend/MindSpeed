from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParam:
    name: str
    num_parameters: int

    @staticmethod
    def cmp(left: object, right: object) -> int:
        if isinstance(left, ModelParam) and isinstance(right, ModelParam):
            if left == right:
                return 1
            elif left.name == right.name:
                return -1
        return 0
