"""Pytest conftest for QAD feature tests.

Sets up megatron module mocks in sys.modules *before* test modules are imported,
so that test files can place all imports at the top of the file (no ``# noqa: E402``
needed).  The modules under test (``logits_kl_loss``, ``teacher_model_manager``,
``config``) import megatron at module level, but megatron is not installed in the
unit-test environment.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import mindspeed
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
    "megatron.training.arguments": MagicMock(),
    "megatron.core.models": MagicMock(),
    "megatron.core.models.gpt": MagicMock(),
    "megatron.core.models.gpt.gpt_layer_specs": MagicMock(),
    "megatron.core.transformer": MagicMock(),
    "megatron.core.transformer.spec_utils": MagicMock(),
    "megatron.legacy.model": MagicMock(),
}

for _name, _mock in _MEGATRON_MOCKS.items():
    sys.modules.setdefault(_name, _mock)

# ---------------------------------------------------------------------------
# 2. Load QADQuantEngineFeature via importlib.util (avoids importing the full
#    mindspeed.features_manager package which pulls in torch_npu)
# ---------------------------------------------------------------------------

# mindspeed is imported at the top of this file; its __init__.py is empty so
# the import is cheap and safe (no torch_npu).  We use mindspeed.__file__ to
# locate the package directory.
_mindspeed_root = os.path.dirname(mindspeed.__file__)


class _StubFeature:
    """Minimal stand-in for MindSpeedFeature with the methods we need."""

    def __init__(self, name, optimization_level=2):
        self.feature_name = name.lower().strip().replace('-', '_')
        self.optimization_level = optimization_level
        self.default_patches = optimization_level == 0

    def is_need_apply(self, args):
        return getattr(args, self.feature_name, None)


_MOCKED_MODULES = [
    "mindspeed.features_manager",
    "mindspeed.features_manager.feature",
    "mindspeed.patch_utils",
]
_saved_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULES}

_mock_feature_module = MagicMock()
_mock_feature_module.MindSpeedFeature = _StubFeature
sys.modules["mindspeed.features_manager"] = MagicMock()
sys.modules["mindspeed.features_manager.feature"] = _mock_feature_module

_mock_patch_utils_module = MagicMock()
_mock_patch_utils_module.MindSpeedPatchesManager = MagicMock()
sys.modules["mindspeed.patch_utils"] = _mock_patch_utils_module

_engine_path = os.path.join(_mindspeed_root, "features_manager", "qad", "qad_quant_engine.py")
_spec = importlib.util.spec_from_file_location("qad_quant_engine", _engine_path)
_qad_quant_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qad_quant_engine)

# Restore sys.modules so we don't pollute other test modules.
for _name in _MOCKED_MODULES:
    _original = _saved_modules.get(_name)
    if _original is not None:
        sys.modules[_name] = _original
    else:
        sys.modules.pop(_name, None)

QADQuantEngineFeature = _qad_quant_engine.QADQuantEngineFeature

# ---------------------------------------------------------------------------
# 3. Shared fixtures
# ---------------------------------------------------------------------------

_PS_PATH = "mindspeed.core.distill.logits_kl_loss.parallel_state"


@pytest.fixture
def tp1():
    """Patch parallel_state for TP=1 (single device)."""
    with patch(_PS_PATH) as mock:
        mock.get_tensor_model_parallel_world_size.return_value = 1
        yield mock


@pytest.fixture
def config():
    """A fresh MagicMock config for each test."""
    return MagicMock()


@pytest.fixture
def manager():
    """A TeacherModelManager with a mock config, reset between tests."""
    from mindspeed.core.distill.teacher_model_manager import TeacherModelManager

    return TeacherModelManager(MagicMock())


@pytest.fixture
def feature():
    """Shared QADQuantEngineFeature instance."""
    return QADQuantEngineFeature()
