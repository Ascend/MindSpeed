from __future__ import annotations

from mindspeed.auto_tuning.utils.singleton import Singleton


class Hardware(metaclass=Singleton):
    """
    hardware modeling
    """
    HARDWARE_PARSE_FILENAME = "auto_tuning_hardware.json"

    def __init__(self) -> None:
        self.device_type: str = "910"
        self.host_ip: str = "localhost"
        self.user_name: str = "root"

        self.cube_performance: float = 363.7248
        self.vector_performance: float = 11.3664
        self.cube_utilization_ratio: float = 0.742
        self.cube_time_ratio: float = 0.62
        self.memory_limit: float = 60.0 * 1024

        # intra-node config
        self.devices_per_node: int = 8
        self.intra_node_bandwidth: int = 196
        self.intra_node_bandwidth_utilization_ratio: float = 0.65

        # inter-node config
        self.num_nodes: int = 2
        self.node_rank: int = 0
        self.inter_node_bandwidth: int = 25
        self.inter_node_bandwidth_utilization_ratio: float = 0.7

    def __str__(self):
        rt = []
        rt.append(f"{'Device Type':<30}{str(self.device_type):<40}")
        rt.append(f"{'Host IP':<30}{str(self.host_ip):<40}")
        rt.append(f"{'Devices Per Node':<30}{str(self.devices_per_node):<40}")
        rt.append(f"{'Number Nodes':<30}{str(self.num_nodes):<40}")
        rt.append(f"{'Node rank':<30}{str(self.node_rank):<40}")
        return '\n'.join(rt)

    @property
    def num_devices(self) -> int:
        return self.devices_per_node * self.num_nodes

    def load(self, hardware: Hardware) -> None:
        for k in self.__dict__.keys():
            if k in hardware.__dict__:
                self.__dict__[k] = hardware.__dict__[k]
