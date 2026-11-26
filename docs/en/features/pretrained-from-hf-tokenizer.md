# PretrainedFromHF Tokenizer

Loads a local Hugging Face tokenizer through the Megatron-Core tokenizer builder entry point.

## Usage

```shell
--tokenizer-type PretrainedFromHF
--tokenizer-name-or-path /path/to/local/tokenizer
```

`--tokenizer-name-or-path` must point to an existing local directory or file path. Loading uses `local_files_only=True` and `trust_remote_code=False`. By default, a fast tokenizer is used; add `--tokenizer-not-use-fast` to switch to the non-fast version.

When the pad token is missing, EOS is used. The tokenizer supports `vocab_extra_ids` handling and computes `padded_vocab_size` when it has not been set yet.
