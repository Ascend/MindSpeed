# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    DEFAULT_SERVER_IP,
    DEFAULT_SERVER_PORT,
    DEFAULT_HEARTBEAT_INTERVAL_MS,
    DEFAULT_HEARTBEAT_TIMEOUT_MS,
    DEFAULT_MAX_MISSED_COUNT,
    DEFAULT_DUMP_DIR,
    DEFAULT_DUMP_TIMEOUT_SECONDS,
    DEFAULT_WAIT_DUMP_FILE_TIMEOUT,
    MAX_DUMP_CKPT_TO_KEEP,
)


@dataclass
class HeartbeatConfig:
    """Heartbeat configuration"""

    interval_ms: int = DEFAULT_HEARTBEAT_INTERVAL_MS
    timeout_ms: int = DEFAULT_HEARTBEAT_TIMEOUT_MS
    max_missed_count: int = DEFAULT_MAX_MISSED_COUNT


@dataclass
class TTPConfig:
    """TTP configuration"""

    enabled: bool = False

    server_ip: str = DEFAULT_SERVER_IP
    server_port: int = DEFAULT_SERVER_PORT

    dump_dir: str = DEFAULT_DUMP_DIR
    max_dump_ckpt_to_keep: int = MAX_DUMP_CKPT_TO_KEEP
    save_optimizer: bool = True
    save_rng: bool = True

    dump_timeout_seconds: int = DEFAULT_DUMP_TIMEOUT_SECONDS
    wait_dump_file_timeout: int = DEFAULT_WAIT_DUMP_FILE_TIMEOUT

    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)

    def validate(self) -> bool:
        """Validate configuration parameters"""
        if not self.enabled:
            return True

        if self.heartbeat.interval_ms <= 0:
            raise ValueError(f"heartbeat.interval_ms must be positive, got {self.heartbeat.interval_ms}")

        if self.heartbeat.timeout_ms <= 0:
            raise ValueError(f"heartbeat.timeout_ms must be positive, got {self.heartbeat.timeout_ms}")

        if self.heartbeat.max_missed_count <= 0:
            raise ValueError(f"heartbeat.max_missed_count must be positive, got {self.heartbeat.max_missed_count}")

        if self.server_port <= 0 or self.server_port > 65535:
            raise ValueError(f"server_port must be in range [1, 65535], got {self.server_port}")

        return True


_global_ttp_config: Optional[TTPConfig] = None


def get_ttp_config() -> TTPConfig:
    """Get TTP configuration from global cache or environment variables.

    The env var fallback exists because Ray worker processes only receive
    config via os.environ (set by set_ttp_config() in the main process).
    Workers never call set_ttp_config() directly.
    """
    if _global_ttp_config is not None:
        return _global_ttp_config

    enabled = os.environ.get('TTP_ENABLED', 'false').lower() == 'true'

    config = TTPConfig(
        enabled=enabled,
        server_ip=os.environ.get('TTP_SERVER_IP', DEFAULT_SERVER_IP),
        server_port=int(os.environ.get('TTP_SERVER_PORT', str(DEFAULT_SERVER_PORT))),
        dump_dir=os.environ.get('TTP_DUMP_DIR', DEFAULT_DUMP_DIR),
        max_dump_ckpt_to_keep=int(os.environ.get('TTP_MAX_DUMP_CKPT_TO_KEEP', str(MAX_DUMP_CKPT_TO_KEEP))),
        save_optimizer=os.environ.get('TTP_SAVE_OPTIMIZER', 'true').lower() == 'true',
        save_rng=os.environ.get('TTP_SAVE_RNG', 'true').lower() == 'true',
        dump_timeout_seconds=int(os.environ.get('TTP_DUMP_TIMEOUT_SECONDS', str(DEFAULT_DUMP_TIMEOUT_SECONDS))),
        wait_dump_file_timeout=int(os.environ.get('TTP_WAIT_DUMP_FILE_TIMEOUT', str(DEFAULT_WAIT_DUMP_FILE_TIMEOUT))),
        heartbeat=HeartbeatConfig(
            interval_ms=int(os.environ.get('TTP_HEARTBEAT_INTERVAL_MS', str(DEFAULT_HEARTBEAT_INTERVAL_MS))),
            timeout_ms=int(os.environ.get('TTP_HEARTBEAT_TIMEOUT_MS', str(DEFAULT_HEARTBEAT_TIMEOUT_MS))),
            max_missed_count=int(os.environ.get('TTP_MAX_MISSED_COUNT', str(DEFAULT_MAX_MISSED_COUNT))),
        ),
    )

    return config


def set_ttp_config(config: TTPConfig) -> None:
    """Set global TTP configuration and sync to environment variables for Ray workers."""
    global _global_ttp_config
    config.validate()
    _global_ttp_config = config

    # Sync to env vars so Ray actor processes can read them via get_ttp_config()
    os.environ['TTP_ENABLED'] = str(config.enabled).lower()
    os.environ['TTP_SERVER_IP'] = config.server_ip
    os.environ['TTP_SERVER_PORT'] = str(config.server_port)
    os.environ['TTP_DUMP_DIR'] = config.dump_dir
    os.environ['TTP_MAX_DUMP_CKPT_TO_KEEP'] = str(config.max_dump_ckpt_to_keep)
    os.environ['TTP_SAVE_OPTIMIZER'] = str(config.save_optimizer).lower()
    os.environ['TTP_SAVE_RNG'] = str(config.save_rng).lower()
    os.environ['TTP_DUMP_TIMEOUT_SECONDS'] = str(config.dump_timeout_seconds)
    os.environ['TTP_WAIT_DUMP_FILE_TIMEOUT'] = str(config.wait_dump_file_timeout)
    os.environ['TTP_HEARTBEAT_INTERVAL_MS'] = str(config.heartbeat.interval_ms)
    os.environ['TTP_HEARTBEAT_TIMEOUT_MS'] = str(config.heartbeat.timeout_ms)
    os.environ['TTP_MAX_MISSED_COUNT'] = str(config.heartbeat.max_missed_count)
