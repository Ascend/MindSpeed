from dataclasses import dataclass

from mindspeed.auto_settings.config.model_config import ModelConfig


@dataclass
class PostInfo:
    FILENAME = "auto_settings_post_info.pkl"

    model_config: ModelConfig = None
    devices_per_node: int = None  # type: ignore
    nnodes: int = None  # type: ignore
    node_rank: int = None  # type: ignore
    device_type: str = None  # type: ignore
    wait_timeout: int = None  # type: ignore
    memory_cap: float = None  # type: ignore
    driver_version: str = None  # type: ignore
    cann_version: str = None  # type: ignore
    mm_model: str = None # type: ignore