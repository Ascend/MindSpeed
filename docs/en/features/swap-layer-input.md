# Transformer Layer Input Offloading

By offloading the Transformer layer input, the device memory occupied by activations retained during training is reduced.

## Usage

**Backend restriction:** When combined with `--moe-fb-overlap`, `--transformer-impl transformer_engine` must be set.

Add `--swap-layer-input` to the training configuration. Disabled by default.

`--swap-layer-input` and [swap-attention](swap_attention.md) are independent switches; the layer input does not need to be specified through `--swap-modules`.
