# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""CP capability routing owned by the MindSpeed/Megatron boundary.

MindSpeed no longer creates CP groups or installs a CP attention fallback.  The
only responsibility left here is to translate the deprecated MindSpeed naming
to Megatron's public ``cp_comm_type`` and reject capability cells that TENPU
does not implement.
"""

from mindspeed.core.context_parallel import get_args


TENPU_NATIVE = 'tenpu'
MINDSPEED_EOD_BASELINE = 'mindspeed_eod_baseline'
UNSUPPORTED_CP = 'unsupported'

_CP_ALGO_TO_COMM_TYPE = {
    'megatron_cp_algo': 'p2p',
    'ulysses_cp_algo': 'a2a',
    'kvallgather_cp_algo': 'all_gather',
    'hybrid_cp_algo': 'a2a+p2p',
}


def get_resolved_cp_comm_type(args=None):
    """Return the one CP communication type selected for this run."""
    if args is None:
        args = get_args()

    cp_comm_type = getattr(args, 'cp_comm_type', None)
    if isinstance(cp_comm_type, (list, tuple)):
        values = list(dict.fromkeys(cp_comm_type))
        if len(values) != 1:
            return None
        cp_comm_type = values[0]

    if cp_comm_type == 'allgather':
        return 'all_gather'
    if cp_comm_type is not None:
        return cp_comm_type
    return _CP_ALGO_TO_COMM_TYPE.get(getattr(args, 'context_parallel_algo', None), 'p2p')


def is_te_mode(args=None):
    """Whether the TE-compatible TENPU backend is selected."""
    if args is None:
        args = get_args()
    return getattr(args, 'transformer_impl', 'transformer_engine') == 'transformer_engine'


def get_cp_backend_route(args=None):
    """Resolve the supported CP/EOD capability cell without changing layout."""
    if args is None:
        args = get_args()

    cp_size = int(getattr(args, 'context_parallel_size', 1))
    is_eod = bool(getattr(args, 'reset_attention_mask', False))
    if cp_size <= 1:
        return MINDSPEED_EOD_BASELINE if is_eod else TENPU_NATIVE

    if not is_te_mode(args):
        return UNSUPPORTED_CP

    raw_cp_comm_type = getattr(args, 'cp_comm_type', None)
    if isinstance(raw_cp_comm_type, (list, tuple)) and len(set(raw_cp_comm_type)) > 1:
        supported_types = {'p2p', 'a2a', 'all_gather', 'allgather', 'a2a+p2p'}
        return TENPU_NATIVE if not is_eod and set(raw_cp_comm_type) <= supported_types else UNSUPPORTED_CP

    cp_comm_type = get_resolved_cp_comm_type(args)
    if cp_comm_type is None:
        return UNSUPPORTED_CP

    if not is_eod:
        return TENPU_NATIVE if cp_comm_type in {'p2p', 'a2a', 'all_gather', 'a2a+p2p'} else UNSUPPORTED_CP

    mask_type = getattr(args, 'attention_mask_type', 'causal')
    if mask_type == 'causal' and cp_comm_type in {'p2p', 'all_gather'}:
        return TENPU_NATIVE

    # No MindSpeed A2A/Ring/Ulysses fallback is retained for EOD.  TENPU-side
    # enablement remains explicitly outside this PR.
    return UNSUPPORTED_CP
