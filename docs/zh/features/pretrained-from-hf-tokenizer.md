# PretrainedFromHF 分词器

通过 Megatron-Core 的分词器构建入口加载本地 Hugging Face 分词器。

## 使用方法

```shell
--tokenizer-type PretrainedFromHF
--tokenizer-name-or-path /path/to/local/tokenizer
```

`--tokenizer-name-or-path` 必须指向已存在的本地目录或文件路径。加载使用 `local_files_only=True`、`trust_remote_code=False`。默认使用 fast tokenizer，添加 `--tokenizer-not-use-fast` 切换为非 fast 版本。

缺少 pad token 时使用 EOS。分词器支持 `vocab_extra_ids` 处理，并在尚未设置时计算 `padded_vocab_size`。
