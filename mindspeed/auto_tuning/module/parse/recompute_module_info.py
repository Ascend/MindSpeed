from typing import Dict


class ModuleRecomputeInfo:
    def __init__(self, context: Dict):
        self.name = context.get("name")
        self.prefix_name = context.get("prefix_name")
        self.full_name = self.prefix_name + '.' + self.name
        self.memory = context.get("memory")
        self.input_size = context.get("input")
        self.time = context.get("time")
        self.recompute = False
