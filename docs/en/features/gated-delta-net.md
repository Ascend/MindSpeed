# GatedDeltaNet (GDN)

Based on the GatedDeltaNet in Megatron-Core, MindSpeed provides NPU implementations of `chunk_gated_delta_rule` and `l2norm`, and the causal convolution uses the basic adaptation of MA. The non-deterministic mode uses MindSpeed's Triton gated delta rule.

## Usage

**Backend restriction:** GDN/hybrid models must set `--transformer-impl transformer_engine`, and `local` is not supported.

Prepare the supporting environment according to [Software Installation](../user-guide/install_guide.md) and import `megatron_adaptor`. Set `--transformer-impl transformer_engine`. The model uses Megatron's native `--hybrid-layer-pattern` to describe the layer structure, where `G` represents a GDN layer; for example, `G*` represents one GDN layer and one Attention layer. The specific dimensions and layer arrangement follow the model configuration.

When using a hybrid model pipeline, the stages are divided by `|` in `--hybrid-layer-pattern`, and it must not be configured together with `--pipeline-model-parallel-layout`, `--num-layers-per-virtual-pipeline-stage`, or `--num-virtual-stages-per-pipeline-rank`.

This document describes the training path; GDN does not yet support the inference path.
