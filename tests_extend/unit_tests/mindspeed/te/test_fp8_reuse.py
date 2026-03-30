import torch

from mindspeed.te.pytorch.fp8.reuse import (
    clear_weight_quantization_reuse_cache,
    generate_weight_reuse_key,
    get_weight_quantization_reuse_stats,
    optimizer_step_reuse_cleanup_wrapper,
    reuse_or_quantize,
)
from mindspeed.te.pytorch.fp8.state_manager import FP8GlobalStateManager


def _fake_quantizer(tensor: torch.Tensor, **kwargs):
    scale_value = kwargs.get("axis", 0) + kwargs.get("bias", 0)
    return tensor.clone(), torch.full((1,), scale_value, dtype=torch.float32)


def test_generate_weight_reuse_key_matches_tensor_views():
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    view_a = weight.view(2, 8)
    view_b = weight.reshape(2, 8)

    key_a = generate_weight_reuse_key(view_a, "fake_quantizer", {"axis": -1})
    key_b = generate_weight_reuse_key(view_b, "fake_quantizer", {"axis": -1})

    assert key_a == key_b


def test_generate_weight_reuse_key_changes_after_weight_update():
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).requires_grad_()
    weight_view = weight.view(2, 8)

    first_key = generate_weight_reuse_key(weight_view, "fake_quantizer", {"axis": -1})
    with torch.no_grad():
        weight.add_(1.0)
    second_key = generate_weight_reuse_key(weight_view, "fake_quantizer", {"axis": -1})

    assert first_key != second_key


def test_reuse_or_quantize_only_reuses_weight_tensors():
    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = True
    clear_weight_quantization_reuse_cache()

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    first_quant, first_scale = reuse_or_quantize(
        weight.view(2, 8),
        "weight",
        _fake_quantizer,
        op_name="fake_quantizer",
        axis=-1,
    )
    second_quant, second_scale = reuse_or_quantize(
        weight.reshape(2, 8),
        "weight",
        _fake_quantizer,
        op_name="fake_quantizer",
        axis=-1,
    )

    input_tensor = torch.ones_like(weight)
    reuse_or_quantize(
        input_tensor,
        "inputs",
        _fake_quantizer,
        op_name="fake_quantizer",
        axis=-1,
    )
    reuse_or_quantize(
        input_tensor,
        "inputs",
        _fake_quantizer,
        op_name="fake_quantizer",
        axis=-1,
    )

    stats = get_weight_quantization_reuse_stats()
    assert first_quant.data_ptr() == second_quant.data_ptr()
    assert first_scale.data_ptr() == second_scale.data_ptr()
    assert stats == {"hits": 1, "misses": 1}

    clear_weight_quantization_reuse_cache()
    FP8GlobalStateManager.FP8_ENABLED = False
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = False


def test_optimizer_step_wrapper_clears_cached_weights():
    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = True
    clear_weight_quantization_reuse_cache()

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    reuse_or_quantize(
        weight,
        "weight",
        _fake_quantizer,
        op_name="fake_quantizer",
        axis=-1,
    )

    step_calls = []

    @optimizer_step_reuse_cleanup_wrapper
    def fake_step():
        step_calls.append("called")
        return "ok"

    assert fake_step() == "ok"
    assert step_calls == ["called"]
    assert get_weight_quantization_reuse_stats() == {"hits": 0, "misses": 0}

    clear_weight_quantization_reuse_cache()
    FP8GlobalStateManager.FP8_ENABLED = False
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = False


def test_optimizer_step_wrapper_clears_cached_weights_after_step_work():
    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = True
    clear_weight_quantization_reuse_cache()

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    @optimizer_step_reuse_cleanup_wrapper
    def fake_step():
        reuse_or_quantize(
            weight,
            "weight",
            _fake_quantizer,
            op_name="fake_quantizer",
            axis=-1,
        )
        return "ok"

    assert fake_step() == "ok"
    assert get_weight_quantization_reuse_stats() == {"hits": 0, "misses": 0}

    clear_weight_quantization_reuse_cache()
    FP8GlobalStateManager.FP8_ENABLED = False
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = False


def test_weight_release_is_skipped_when_reuse_is_enabled():
    from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8Tensor2D

    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = True

    tensor_2d = Float8Tensor2D(torch.float8_e4m3fn, torch.Size([2, 2]), torch.device('cpu'), key='weight')
    data = torch.ones((2, 2), dtype=torch.uint8)
    scale = torch.ones((1, 1), dtype=torch.float32)
    tensor_2d.release(data, scale)

    assert data.untyped_storage().size() > 0
    assert scale.untyped_storage().size() > 0

    FP8GlobalStateManager.FP8_ENABLED = False
    FP8GlobalStateManager.FP8_REUSE_QUANTIZED_WEIGHT = False


def test_reuse_is_disabled_without_runtime_flag():
    calls = {"count": 0}

    def counted_quantizer(tensor: torch.Tensor, **kwargs):
        calls["count"] += 1
        return tensor.clone(), torch.ones((1,), dtype=torch.float32)

    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.set_weight_quantization_reuse_enabled(False)

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    reuse_or_quantize(weight, "weight", counted_quantizer, op_name="counted", axis=-1)
    reuse_or_quantize(weight, "weight", counted_quantizer, op_name="counted", axis=-1)

    assert calls["count"] == 2
    assert get_weight_quantization_reuse_stats() == {"hits": 0, "misses": 0}

    clear_weight_quantization_reuse_cache()
    FP8GlobalStateManager.FP8_ENABLED = False


def test_disabling_reuse_clears_existing_cached_weights():
    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.set_weight_quantization_reuse_enabled(True)

    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    reuse_or_quantize(weight, "weight", _fake_quantizer, op_name="fake_quantizer", axis=-1)
    assert get_weight_quantization_reuse_stats() == {"hits": 0, "misses": 1}

    FP8GlobalStateManager.set_weight_quantization_reuse_enabled(False)
    assert get_weight_quantization_reuse_stats() == {"hits": 0, "misses": 0}

    FP8GlobalStateManager.FP8_ENABLED = False


def test_reuse_is_disabled_for_non_persistent_weight_tensor():
    calls = {"count": 0}

    def counted_quantizer(tensor: torch.Tensor, **kwargs):
        calls["count"] += 1
        return tensor.clone(), torch.ones((1,), dtype=torch.float32)

    FP8GlobalStateManager.FP8_ENABLED = True
    FP8GlobalStateManager.set_weight_quantization_reuse_enabled(True)

    base_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).requires_grad_()
    stacked_weight = torch.stack((base_weight, base_weight), dim=0)
    reuse_or_quantize(stacked_weight, "weight", counted_quantizer, op_name="counted", axis=-1)
    reuse_or_quantize(stacked_weight, "weight", counted_quantizer, op_name="counted", axis=-1)

    assert calls["count"] == 2
    assert get_weight_quantization_reuse_stats() == {"hits": 0, "misses": 0}

    clear_weight_quantization_reuse_cache()
    FP8GlobalStateManager.FP8_ENABLED = False
    FP8GlobalStateManager.set_weight_quantization_reuse_enabled(False)
