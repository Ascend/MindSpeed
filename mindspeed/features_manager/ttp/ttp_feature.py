# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""TTP feature — CLI args and patch registration.

Patch registration goes through the standard MindSpeedFeature lifecycle:
  - Phase 1 (import time): patch_features() → is_need_apply(args).
    args.ttp_enabled is typically False here (no --ttp-enabled on verl CLI).
  - Phase 2 (repatch): verl's _init_device_mesh() calls repatch() with
    override_transformer_config (including ttp_enabled=True) → re-registers
    patches → optimizer __new__ patch is applied in workers.

Env vars (TTP_ENABLED, TTP_SERVER_IP, etc.) are exported in the bash launcher
before calling verl, so workers inherit them via os.environ and get_ttp_config()
can build the full TTPConfig at optimizer construction time.
"""

import logging
import os
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature

logger = logging.getLogger(__name__)


class TTPFeature(MindSpeedFeature):
    """TTP feature — CLI args and patch registration.

    optimization_level=2 (standard): is_need_apply() checks getattr(args, 'ttp_enabled', False)
    via the base class. In Phase 1 this is False (no --ttp-enabled in verl sys.argv);
    in Phase 2 repatch sets ttp_enabled=True on the args namespace and patches are re-registered.
    """

    def __init__(self):
        super().__init__('ttp_enabled', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='ttp')
        group.add_argument('--ttp-enabled', action='store_true', default=False, help='Enable TTP fault recovery')
        group.add_argument('--ttp-server-ip', type=str, default='0.0.0.0', help='TTP server IP')
        group.add_argument('--ttp-server-port', type=int, default=29500, help='TTP server port')
        group.add_argument('--ttp-dump-dir', type=str, default='ttp_dump', help='Dump checkpoint directory')
        group.add_argument('--ttp-max-dump-ckpt-to-keep', type=int, default=1, help='Max dump checkpoints to keep')
        group.add_argument(
            '--ttp-save-optimizer', action='store_true', default=True, help='Save optimizer state in dump'
        )
        group.add_argument('--ttp-save-rng', action='store_true', default=True, help='Save RNG state in dump')
        group.add_argument('--ttp-dump-timeout-seconds', type=int, default=60, help='Dump timeout in seconds')
        group.add_argument('--ttp-wait-dump-file-timeout', type=int, default=60, help='Wait dump file timeout')
        group.add_argument('--ttp-heartbeat-interval-ms', type=int, default=1000, help='Heartbeat interval in ms')
        group.add_argument('--ttp-heartbeat-timeout-ms', type=int, default=3000, help='Heartbeat timeout in ms')
        group.add_argument('--ttp-max-missed-count', type=int, default=3, help='Max missed heartbeats before fault')

    def register_patches(self, patch_manager, args):
        if not getattr(args, 'ttp_enabled', False):
            logger.debug("[TTP] ttp_enabled=False, skipping patch registration")
            return

        logger.warning("[TTP] Registering patches (ttp_enabled=True)")

        # Sync TTP config from args to global singleton + os.environ.
        # During Phase 2 (repatch), this ensures workers have the full config.
        _try_set_ttp_config(args)

        # Permanent patch: DistributedOptimizer.__new__ → TTPReplicaOptimizer
        from mindspeed.core.ttp.adaptor import patched_optimizer_new

        patch_manager.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__new__',
            patched_optimizer_new,
        )
        logger.info("[TTP] Registered optimizer __new__ patch")

        # Permanent patch: ChainedOptimizer.__init__ → store parent on sub-optimizers.
        # The save callback accesses the full ChainedOptimizer (not a sub-optimizer)
        # to produce a complete sharded_state_dict.  Saving the original __init__ first
        # so our wrapper can delegate to it.
        from megatron.core.optimizer.optimizer import ChainedOptimizer
        from mindspeed.core.ttp.adaptor import patched_chained_optimizer_init
        import mindspeed.core.ttp.adaptor as _adaptor

        _adaptor._ORIGINAL_CHAINED_OPTIMIZER_INIT = ChainedOptimizer.__init__
        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.ChainedOptimizer.__init__',
            patched_chained_optimizer_init,
        )
        logger.info("[TTP] Registered ChainedOptimizer __init__ patch")

        # Permanent patch: ChainedOptimizer.step → @ttp_exception_handler.
        # The count_zero_fix patched step() calls clip_grad_by_total_norm_fp32
        # BEFORE step_with_ready_grads().  If stop_device fires during gradient
        # clipping, FORCE STOP is NOT caught by the decorators on either
        # step_with_ready_grads or forward_backward.  This wrapper adds
        # exception handling at the ChainedOptimizer.step level.
        from mindspeed.core.ttp.adaptor import patched_chained_optimizer_step_wrapper

        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.ChainedOptimizer.step',
            patched_chained_optimizer_step_wrapper,
        )
        logger.info("[TTP] Registered ChainedOptimizer step patch")

        # Permanent patch: forward_backward schedule → check PAUSE at every step.
        # Without this, PAUSE arriving during forward/backward goes undetected
        # because @ttp_exception_handler only wraps optimizer methods.
        from mindspeed.core.ttp.adaptor import patched_forward_backward_wrapper

        patch_manager.register_patch(
            'megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining',
            patched_forward_backward_wrapper,
        )
        logger.info("[TTP] Registered forward_backward_no_pipelining patch")


def _try_set_ttp_config(args):
    """Build TTPConfig from parsed args and sync to global singleton + os.environ."""
    try:
        from mindspeed.core.ttp.config import TTPConfig, HeartbeatConfig, set_ttp_config

        # Prefer args (from repatch → override_transformer_config) over env vars
        # so that ttp_enabled is controlled solely via Hydra config.
        # Env vars are kept as fallback for Ray workers (set by set_ttp_config()).
        config = TTPConfig(
            enabled=getattr(args, 'ttp_enabled', False) or os.environ.get('TTP_ENABLED', 'false').lower() == 'true',
            server_ip=os.environ.get('TTP_SERVER_IP', getattr(args, 'ttp_server_ip', '0.0.0.0')),
            server_port=int(os.environ.get('TTP_SERVER_PORT', getattr(args, 'ttp_server_port', 29500))),
            dump_dir=os.environ.get('TTP_DUMP_DIR', getattr(args, 'ttp_dump_dir', 'ttp_dump')),
            max_dump_ckpt_to_keep=int(
                os.environ.get('TTP_MAX_DUMP_CKPT_TO_KEEP', getattr(args, 'ttp_max_dump_ckpt_to_keep', 1))
            ),
            save_optimizer=os.environ.get('TTP_SAVE_OPTIMIZER', str(getattr(args, 'ttp_save_optimizer', True))).lower()
            == 'true',
            save_rng=os.environ.get('TTP_SAVE_RNG', str(getattr(args, 'ttp_save_rng', True))).lower() == 'true',
            dump_timeout_seconds=int(
                os.environ.get('TTP_DUMP_TIMEOUT_SECONDS', getattr(args, 'ttp_dump_timeout_seconds', 60))
            ),
            wait_dump_file_timeout=int(
                os.environ.get('TTP_WAIT_DUMP_FILE_TIMEOUT', getattr(args, 'ttp_wait_dump_file_timeout', 60))
            ),
            heartbeat=HeartbeatConfig(
                interval_ms=int(
                    os.environ.get('TTP_HEARTBEAT_INTERVAL_MS', getattr(args, 'ttp_heartbeat_interval_ms', 1000))
                ),
                timeout_ms=int(
                    os.environ.get('TTP_HEARTBEAT_TIMEOUT_MS', getattr(args, 'ttp_heartbeat_timeout_ms', 3000))
                ),
                max_missed_count=int(os.environ.get('TTP_MAX_MISSED_COUNT', getattr(args, 'ttp_max_missed_count', 3))),
            ),
        )
        set_ttp_config(config)
        logger.info(
            "[TTP] Config synced: server=%s:%d dump_dir=%s", config.server_ip, config.server_port, config.dump_dir
        )
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as e:
        logger.warning("[TTP] set_ttp_config failed: %s", e)
