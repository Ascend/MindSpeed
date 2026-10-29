# get_batch Configuration Cache

Caches the `TransformerConfig` constructed inside `get_batch`, reducing the host overhead caused by repeatedly creating the configuration.

## Usage

Add `--cache-get-batch-config` to the training script. Disabled by default.

The adapter locates the training entry from the `forward_step_func` passed to `pretrain`. The entry must provide both `get_batch` and `core_transformer_config_from_args`; only configuration construction initiated directly by that `get_batch` is cached. The cache is distinguished by the args object, the configuration class, the MLA switch, and the heterogeneous layer configuration path, and does not cache data batches.
