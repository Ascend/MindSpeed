from types import SimpleNamespace

import pytest
import torch

from mindspeed.core.transformer.experimental_attention_variant import dsa_rope


def test_dsa_config_wrapper_hides_rope_fusion_only_during_validation():
    observed = []

    def post_init(config):
        observed.append(config.apply_rope_fusion)

    config = SimpleNamespace(
        experimental_attention_variant="dsa", apply_rope_fusion=True
    )

    dsa_rope.dsa_transformer_config_post_init_wrapper(post_init)(config)

    assert observed == [False]
    assert config.apply_rope_fusion is True


def test_dsa_config_wrapper_restores_rope_fusion_after_validation_error():
    def post_init(_config):
        raise RuntimeError("validation failed")

    config = SimpleNamespace(
        experimental_attention_variant="dsa", apply_rope_fusion=True
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        dsa_rope.dsa_transformer_config_post_init_wrapper(post_init)(config)

    assert config.apply_rope_fusion is True


def test_dsa_indexer_fused_rope_preserves_nope_slice(monkeypatch):
    calls = []

    def fake_apply_fused_rope(
        x,
        rotary_pos_emb,
        mscale,
        rotary_interleaved,
        multi_latent_attention,
    ):
        calls.append(
            (
                x.clone(),
                rotary_pos_emb,
                mscale,
                rotary_interleaved,
                multi_latent_attention,
            )
        )
        return x + 10

    monkeypatch.setattr(dsa_rope, "apply_fused_rope", fake_apply_fused_rope)
    indexer = SimpleNamespace(
        config=SimpleNamespace(
            apply_rope_fusion=True,
            rotary_interleaved=True,
            multi_latent_attention=True,
        ),
        index_head_dim=6,
        qk_pos_emb_head_dim=2,
    )
    x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 2, 6)
    freqs = torch.zeros(1, 1, 1, 2)

    wrapped = dsa_rope.dsa_indexer_apply_rope_wrapper(lambda *_args, **_kwargs: None)
    actual = wrapped(indexer, x, freqs, 0.5)

    torch.testing.assert_close(actual[..., :4], x[..., :4])
    torch.testing.assert_close(actual[..., 4:], x[..., 4:] + 10)
    assert calls[0][2:] == (0.5, True, True)


def test_dsa_indexer_uses_native_path_when_fusion_is_disabled():
    expected = torch.tensor([1.0])

    def native(*_args, **_kwargs):
        return expected

    indexer = SimpleNamespace(config=SimpleNamespace(apply_rope_fusion=False))
    wrapped = dsa_rope.dsa_indexer_apply_rope_wrapper(native)

    assert wrapped(indexer, torch.empty(0), torch.empty(0), 1.0) is expected


def test_apply_fused_rope_builds_scaled_cos_sin_and_mode(monkeypatch):
    captured = {}

    def fake_npu_fused_rope(x, cos, sin, mode):
        captured.update(x=x, cos=cos, sin=sin, mode=mode)
        return x

    monkeypatch.setattr(dsa_rope, "_npu_fused_rope", fake_npu_fused_rope)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 4).transpose(0, 1)
    freqs = torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]).reshape(
        1, 1, 1, 4
    )

    actual = dsa_rope.apply_fused_rope(
        x,
        freqs,
        mscale=0.5,
        rotary_interleaved=True,
        multi_latent_attention=True,
    )

    assert actual.is_contiguous()
    assert captured["mode"] == 1
    torch.testing.assert_close(
        captured["x"], torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    )
    torch.testing.assert_close(captured["cos"], torch.cos(freqs) * 0.5)
    torch.testing.assert_close(captured["sin"], torch.sin(freqs) * 0.5)
