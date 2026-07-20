"""Pytest conftest for QAD mindspeed tests.

Sets up megatron module mocks in sys.modules *before* test modules are imported,
so that test files can place all imports at the top of the file (no ``# noqa: E402``
needed).  The module under test (``logits_kl_loss``) imports megatron at module
level, but megatron is not installed in the unit-test environment.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Megatron module mocks (must be set up before any mindspeed imports)
# ---------------------------------------------------------------------------

_mock_parallel_state = MagicMock()
_mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 1

_mock_megatron_core = MagicMock()
_mock_megatron_core.parallel_state = _mock_parallel_state
_mock_megatron_core.mpu = MagicMock()
_mock_megatron_core.mpu.get_tensor_model_parallel_world_size.return_value = 1

_MEGATRON_MOCKS = {
    "megatron": MagicMock(),
    "megatron.core": _mock_megatron_core,
    "megatron.core.mpu": _mock_megatron_core.mpu,
    "megatron.core.parallel_state": _mock_parallel_state,
    "megatron.training": MagicMock(),
}

for _name, _mock in _MEGATRON_MOCKS.items():
    sys.modules.setdefault(_name, _mock)

# ---------------------------------------------------------------------------
# 2. Shared fixtures
# ---------------------------------------------------------------------------

_PS_PATH = "mindspeed.core.distill.logits_kl_loss.parallel_state"


@pytest.fixture
def tp1_env():
    """Patch parallel_state for TP=1 (single device, no all_reduce needed)."""
    with patch(_PS_PATH) as mock:
        mock.get_tensor_model_parallel_world_size.return_value = 1
        yield mock


@pytest.fixture
def tp2_env():
    """Patch parallel_state for TP=2 (multi-device, all_reduce path active)."""
    with patch(_PS_PATH) as mock:
        mock.get_tensor_model_parallel_world_size.return_value = 2
        mock.get_tensor_model_parallel_group.return_value = MagicMock()
        yield mock
