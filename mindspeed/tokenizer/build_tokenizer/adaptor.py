# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024; HUAWEI CORPORATION.
# Copyright (c) 2024; HUAWEI CORPORATION. All rights reserved.
from transformers import AutoTokenizer
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


def build_tokenizer_HF(args, **kwargs):
    """Pure HF tokenizer builder, to be patched when tokenizer_type=PretrainedFromHF"""
    if args.rank == 0:
        print(' > building PretrainFromHF tokenizer. Vocab file is un-used, '
              'loading tokenizer from pre-trained model', flush=True)

    if args.tokenizer_name_or_path is None:
        raise ValueError("Missing tokenizer_name_or_path while building PretrainFromHF tokenizer.")

    from pathlib import Path
    if not Path(args.tokenizer_name_or_path).exists():
        raise FileNotFoundError(f"Tokenizer path not found: {args.tokenizer_name_or_path}")

    hf_tokenizer_kwargs = dict()
    if hasattr(args, "tokenizer_kwargs") and args.tokenizer_kwargs:
        if len(args.tokenizer_kwargs) % 2 != 0:
            raise ValueError("The token name and token value must be entered in pairs.")

        for i in range(0, len(args.tokenizer_kwargs), 2):
            hf_tokenizer_kwargs[args.tokenizer_kwargs[i]] = \
                args.tokenizer_kwargs[i + 1]

    tokenizer = _AutoTokenizer(
        args.tokenizer_name_or_path,
        vocab_extra_ids=args.vocab_extra_ids,
        model_max_length=args.seq_length,
        use_fast=args.tokenizer_not_use_fast,
        **hf_tokenizer_kwargs
    )

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


class _AutoTokenizer(MegatronTokenizer):
    """AutoTokenizer for Hf Pretrained model loading."""

    def __init__(self, tokenizer_name_or_path, vocab_extra_ids, model_max_length, use_fast, **kwargs):
        name = tokenizer_name_or_path
        super().__init__(name)
        hf_tokenizer_kwargs = kwargs
        if vocab_extra_ids > 0:
            hf_tokenizer_kwargs["additional_special_tokens"] = [f"<extra_id_{_id}>" for _id in range(vocab_extra_ids)]

        hf_tokenizer_kwargs["model_max_length"] = model_max_length
        hf_tokenizer_kwargs["use_fast"] = use_fast
        hf_tokenizer_kwargs["trust_remote_code"] = False
        hf_tokenizer_kwargs["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **hf_tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer)  # vocab_size doesn't contain additional tokens

    @property
    def vocab(self):
        if not hasattr(self, '_cached_vocab'):
            self._cached_vocab = {
                **{special_token: self.tokenizer.convert_tokens_to_ids(special_token)
                   for special_token in self.tokenizer.additional_special_tokens},
                **self.tokenizer.vocab,
            }
        return self._cached_vocab

    @property
    def inv_vocab(self):
        if not hasattr(self, '_cached_inv_vocab'):
            self._cached_inv_vocab = {v: k for k, v in self.vocab.items()}
        return self._cached_inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eos

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def cls(self):
        candidate = self.tokenizer.cls_token_id
        return self._check_token_candidate(candidate)

    @property
    def sep(self):
        candidate = self.tokenizer.sep_token_id
        return self._check_token_candidate(candidate)

    @property
    def pad(self):
        candidate = self.tokenizer.pad_token_id

        # just use eos_token_id if pad_token_id is not available, it is reasonable
        # maybe add a new token, and resize embedding layer is better
        if candidate is None:
            candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def mask(self):
        candidate = self.tokenizer.mask_token_id
        return self._check_token_candidate(candidate)

    @property
    def bos(self):
        raise NotImplementedError("Missing <bos>")

    @property
    def eos(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def additional_special_tokens_ids(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self.tokenizer.additional_special_tokens_ids

    @staticmethod
    def _check_token_candidate(candidate):
        if candidate is None:
            raise AttributeError("Token doesn't exist")
        return candidate