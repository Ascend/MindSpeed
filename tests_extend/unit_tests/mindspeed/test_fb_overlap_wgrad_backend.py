# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU contract checks for backend selection, not NPU numerical precision."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _layers_source():
    """Locate production code even when tests_extend is copied into Megatron-LM."""
    try:
        package_spec = importlib.util.find_spec("mindspeed")
    except (ImportError, ValueError):
        package_spec = None
    package_dirs = []
    if package_spec is not None:
        package_dirs.extend(Path(path) for path in package_spec.submodule_search_locations or ())
    package_dirs.append(Path(__file__).resolve().parents[3] / "mindspeed")
    for package_dir in package_dirs:
        source = package_dir / "core/transformer/moe/moe_feature/fb_overlap/modules/layers.py"
        if source.is_file():
            return source
    raise FileNotFoundError(
        "MindSpeed sources are required. Install the MindSpeed checkout with "
        "'python -m pip install -e /path/to/MindSpeed' or add it to PYTHONPATH. "
        "Copied tests_extend files alone do not contain the production code."
    )


@pytest.fixture
def backward_env(monkeypatch):
    # Resolve the real package before replacing training dependencies with stubs.
    source = _layers_source()
    torch = ModuleType("torch")
    torch.distributed = ModuleType("torch.distributed")
    torch.float32, torch.float16, torch.bfloat16 = "float32", "float16", "bfloat16"
    torch.cuda = SimpleNamespace(current_device=lambda: 0)
    torch.empty = Mock(return_value=object())
    torch.zeros = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)

    feature_name = "mindspeed.core.transformer.moe.moe_feature"
    feature = ModuleType(feature_name)
    feature.prepare_input_tensors_for_wgrad_compute = lambda dy, x: (dy, x)
    for name in ("get_global_memory_buffer", "get_tensor_model_parallel_group", "get_tensor_model_parallel_world_size"):
        setattr(feature, name, Mock())
    feature.get_args = lambda: SimpleNamespace(overlap_grad_reduce=False)
    monkeypatch.setitem(sys.modules, feature_name, feature)

    store = ModuleType("_fb_overlap_wgrad_backend_test.weight_grad_store")
    store.WeightGradStore = SimpleNamespace(is_decoupleBlock=False)
    monkeypatch.setitem(sys.modules, store.__name__, store)

    backend = ModuleType("fused_weight_gradient_mlp_cuda")
    backend.wgrad_gemm_accum_fp32 = Mock()
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    legacy_backend = ModuleType("mindspeed.ops.npu_matmul_add")
    legacy_backend.npu_matmul_add_fp32 = Mock(
        side_effect=AssertionError("FB backward bypassed the adapter's registered backend")
    )
    monkeypatch.setitem(sys.modules, legacy_backend.__name__, legacy_backend)

    spec = importlib.util.spec_from_file_location("_fb_overlap_wgrad_backend_test.layers", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    input_ = SimpleNamespace(dtype=torch.bfloat16)
    main_grad = SimpleNamespace(dtype=torch.float32, shape=(2, 3), value=7)
    weight = SimpleNamespace(main_grad=main_grad)
    grad_output = Mock()
    ctx = SimpleNamespace(
        saved_tensors=(input_, weight),
        use_bias=False,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        sequence_parallel=False,
        allreduce_dgrad=False,
        gradient_accumulation_fusion=True,
    )
    return SimpleNamespace(
        backward=module.linear_backward_wgrad_detach,
        torch=torch,
        backend=backend.wgrad_gemm_accum_fp32,
        ctx=ctx,
        grad_output=grad_output,
        input=input_,
        weight=weight,
    )


@pytest.mark.parametrize("zero_out_wgrad", [None, False, True])
def test_fused_backward_uses_registered_backend(backward_env, zero_out_wgrad):
    env = backward_env
    if zero_out_wgrad is not None:
        env.weight.grad_added_to_main_grad = False
        env.weight.zero_out_wgrad = zero_out_wgrad

    def accumulate(input_, grad_output, main_grad):
        assert input_ is env.input
        assert grad_output is env.grad_output
        assert main_grad is env.weight.main_grad
        main_grad.value += 5

    env.backend.side_effect = accumulate
    result = env.backward(env.ctx, env.grad_output)

    env.backend.assert_called_once_with(env.input, env.grad_output, env.weight.main_grad)
    assert env.weight.main_grad.value == 12
    assert result[0] is env.grad_output.matmul.return_value
    env.grad_output.t.assert_not_called()
    if zero_out_wgrad is None:
        assert result[1] is None
        env.torch.empty.assert_not_called()
        env.torch.zeros.assert_not_called()
    else:
        factory = env.torch.zeros if zero_out_wgrad else env.torch.empty
        assert result[1] is factory.return_value
        factory.assert_called_once_with((2, 3), dtype=env.torch.bfloat16, device=0, requires_grad=False)
        assert env.weight.grad_added_to_main_grad is True


def test_unfused_backward_returns_weight_gradient(backward_env):
    env = backward_env
    env.ctx.gradient_accumulation_fusion = False
    env.weight.grad_added_to_main_grad = False

    result = env.backward(env.ctx, env.grad_output)

    env.backend.assert_not_called()
    env.grad_output.t.return_value.matmul.assert_called_once_with(env.input)
    assert result[1] is env.grad_output.t.return_value.matmul.return_value
    assert env.weight.main_grad.value == 7
    assert env.weight.grad_added_to_main_grad is False
    env.torch.empty.assert_not_called()
    env.torch.zeros.assert_not_called()
