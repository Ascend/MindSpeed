# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer, BertTokenizer

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase


@pytest.fixture(params=["adaptor", "legacy"])
def builder(request):
    if request.param == "adaptor":
        module = importlib.import_module("mindspeed.tokenizer.build_tokenizer.adaptor")
        return module.build_tokenizer_HF
    module = importlib.import_module("mindspeed.tokenizer.tokenizer")

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("PretrainedFromHF must use the MindSpeed HF loader")

    return module.build_tokenizer_wrapper(unexpected_fallback)


@pytest.fixture
def tokenizer_path(tmp_path):
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_file), eos_token="[SEP]")
    tokenizer.save_pretrained(tmp_path)
    return str(tmp_path)


def make_args(tokenizer_path, **overrides):
    values = dict(
        tokenizer_type="PretrainedFromHF",
        tokenizer_name_or_path=tokenizer_path,
        tokenizer_kwargs=[],
        vocab_extra_ids=0,
        seq_length=32,
        tokenizer_not_use_fast=True,
        padded_vocab_size=None,
        make_vocab_size_divisible_by=8,
        tensor_model_parallel_size=2,
        rank=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("use_fast", [False, True])
@pytest.mark.parametrize("extra_ids", [0, 2])
@pytest.mark.parametrize("pad_token", [None, "[PAD]"])
def test_pretrained_hf_preserves_loading_and_encoding(builder, tokenizer_path, use_fast, extra_ids, pad_token):
    args = make_args(
        tokenizer_path,
        tokenizer_not_use_fast=use_fast,
        vocab_extra_ids=extra_ids,
        tokenizer_kwargs=["pad_token", pad_token, "do_lower_case", False],
    )
    expected_kwargs = dict(
        pad_token=pad_token,
        do_lower_case=False,
        model_max_length=32,
        use_fast=use_fast,
        trust_remote_code=False,
        local_files_only=True,
    )
    if extra_ids:
        expected_kwargs["additional_special_tokens"] = [f"<extra_id_{i}>" for i in range(extra_ids)]

    with patch.object(AutoTokenizer, "from_pretrained", wraps=AutoTokenizer.from_pretrained) as load:
        tokenizer = builder(args)
    load.assert_called_once_with(tokenizer_path, **expected_kwargs)
    reference = AutoTokenizer.from_pretrained(tokenizer_path, **expected_kwargs)

    assert isinstance(tokenizer, MegatronTokenizerBase)
    assert tokenizer.path == tokenizer_path
    assert tokenizer.tokenizer.is_fast == use_fast
    assert tokenizer.tokenizer.model_max_length == 32
    for text in ("hello world", "Hello world", "<extra_id_0> hello"):
        assert tokenizer.tokenize(text) == reference.encode(text)
        assert tokenizer.detokenize(reference.encode(text)) == reference.decode(reference.encode(text))
    assert tokenizer.vocab == reference.get_vocab()
    assert tokenizer.inv_vocab == {v: k for k, v in reference.get_vocab().items()}
    assert tokenizer.vocab_size == len(reference) == 7 + extra_ids
    assert tokenizer.encoder == tokenizer.vocab
    assert tokenizer.decoder == tokenizer.inv_vocab
    assert tokenizer.eod == tokenizer.eos == tokenizer.eos_token_id == reference.eos_token_id
    assert tokenizer.pad == (reference.eos_token_id if pad_token is None else reference.pad_token_id)
    assert tokenizer.cls == reference.cls_token_id
    assert tokenizer.sep == reference.sep_token_id
    assert tokenizer.mask == reference.mask_token_id
    assert tokenizer.additional_special_tokens_ids == reference.additional_special_tokens_ids
    assert args.padded_vocab_size == 16

    conversation = [{"role": "user", "content": "hello world"}]
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    for tokenize in (False, True):
        kwargs = dict(chat_template=template, tokenize=tokenize, add_generation_prompt=False)
        assert tokenizer.apply_chat_template(conversation, **kwargs) == reference.apply_chat_template(
            conversation, **kwargs
        )


@pytest.mark.parametrize(
    "changed_options",
    [
        {"vocab_extra_ids": 2},
        {"seq_length": 64},
        {"tokenizer_not_use_fast": False},
        {"tokenizer_kwargs": ["eos_token", "[CLS]"]},
    ],
)
def test_dataset_cache_identifies_effective_tokenizer_options(builder, tokenizer_path, changed_options):
    tokenizer = builder(make_args(tokenizer_path))
    repeated = builder(make_args(tokenizer_path))
    changed = builder(make_args(tokenizer_path, **changed_options))

    # MegatronDataset serializes its config.tokenizer using this callback.
    description = json.dumps({"tokenizer": tokenizer}, default=lambda obj: obj.unique_identifiers)
    assert json.loads(description)["tokenizer"] == json.loads(tokenizer.unique_description)
    assert tokenizer.unique_identifiers == repeated.unique_identifiers
    assert tokenizer.unique_identifiers != changed.unique_identifiers


def test_checkpoint_padded_vocab_size_is_preserved(builder, tokenizer_path):
    args = make_args(tokenizer_path, padded_vocab_size=128)
    builder(args)
    assert args.padded_vocab_size == 128


def test_missing_tokenizer_path_is_rejected(builder):
    with pytest.raises(ValueError, match="Missing tokenizer_name_or_path"):
        builder(make_args(None))


def test_unpaired_tokenizer_kwargs_are_rejected(builder, tokenizer_path):
    with pytest.raises(ValueError, match="must be entered in pairs"):
        builder(make_args(tokenizer_path, tokenizer_kwargs=["eos_token"]))


def test_legacy_wrapper_delegates_other_tokenizer_types():
    module = importlib.import_module("mindspeed.tokenizer.tokenizer")
    args = make_args(None, tokenizer_type="HuggingFaceTokenizer")
    sentinel = object()

    def fallback(received_args, **kwargs):
        assert received_args is args
        assert kwargs == {"custom_option": True}
        return sentinel

    assert module.build_tokenizer_wrapper(fallback)(args, custom_option=True) is sentinel
