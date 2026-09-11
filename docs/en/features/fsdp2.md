# FSDP2

## Background and Challenges

PyTorch's Fully Sharded Data Parallel (FSDP) aims to provide a high-performance eager-mode implementation with communication bucketing and communication/computation overlap. The API represents communication buckets by flattening and concatenating a group of parameters into a FlatParameter. However, this FlatParameter design makes it difficult to apply differentiated operations (such as parameter freezing and precision conversion) to individual parameters within a bucket, compromising composability. It also complicates the internal implementation—for example, the state dict logic spans thousands of lines of code and requires additional communication.

## Solution

To address these limitations, FSDP2 removes FlatParameter and uses DTensors sharded along dimension 0 to represent sharded parameters. This enables convenient operations on individual parameters, a communication-free sharded state dict, and a simplified initialization flow.

## Usage

In MindSpeed, the entry point for FSDP2 is a configuration file. By generating the configuration file and passing command-line arguments, you can use this feature.

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=2 # Must not be set to 1
--use-torch-fsdp2 \
--fsdp2-config-path ./fsdp2_config.yaml \
--ckpt-format torch_dist \
--untie-embeddings-and-output-weights \
# Note: The distributed optimizer must not be enabled
```

The configuration items for `fsdp2_config.yaml` are as follows:

```yaml
sharding_size: int # Sharding group size, indicating the number of NPUs in each parameter sharding group
sub_modules_to_wrap: Optional[Iterable[torch.nn.Module]] = None # List of module classes to be wrapped by FSDP, must be imported via absolute path, e.g.: mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
reshard_after_forward: Union[bool, int] = True # Reshard parameters immediately after forward computation
param_dtype: bf16 # Parameter storage precision
reduce_dtype: fp32 # Gradient communication precision
cast_forward_inputs: bool = True # Automatically cast forward inputs to compute precision
ignored_modules: Optional[Iterable[torch.nn.Module]] = None # List of module classes to exclude from FSDP management, must be imported via absolute path: e.g., mindspeed_mm.models.ae.base.AEModel
offload_to_cpu: bool = False # Offload weights, gradients, and optimizer states to CPU
pin_memory: bool = True  # Takes effect only when offload_to_cpu is True
num_to_forward_prefetch: int  # Specifies the number of layers for forward prefetch, default value is 0
recompute_modules: Optional[Iterable[torch.nn.Module]] = None # List of module classes that require recomputation, must be imported via absolute path, for example: mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
```

> [!NOTE]
>
> **Recomputation Mechanism Description**: In FSDP2 mode, the `recompute_modules` configuration item in `fsdp2_config.yaml` is used for recomputation, implemented based on PyTorch's native `apply_activation_checkpointing` + `checkpoint_wrapper`. It wraps specified modules with activation checkpointing, meaning intermediate activation values are not saved during forward propagation and are recomputed during backward propagation.
> Recomputation in this mode is mutually exclusive with Megatron-style recompute (such as `--recompute-granularity`, `--recompute-method`, `--recompute-num-layers`) and cannot be enabled simultaneously.
>
> **Recomputation Performance Cost**: Recomputation trades computation for memory. Full-layer recomputation can save approximately 30–40% of memory, but may result in a 15–25% throughput drop. Partial recomputation can save approximately 15–25% of memory, with a throughput drop of about 5–10%. The actual cost depends on the model architecture, batch size, and hardware configuration.
>
> **Reasons for Throughput Drop**: Recomputation avoids saving intermediate activations during forward propagation and recomputes them during backward propagation, adding extra computational overhead. Additionally, if `reshard_after_forward=True` is configured, backward propagation also requires re-fetching sharded parameters, which may introduce additional communication overhead.
>
> **Recomputation Usage Boundary**: It is recommended to enable recompute only when activation memory is the primary bottleneck. Recommended conditions for enabling: OOM errors occur during training; activation memory usage exceeds 50% of total memory; or when a larger batch size or longer sequence length is desired. Not recommended when computation itself is the bottleneck (NPU utilization close to 100%), communication is the bottleneck (network bandwidth is saturated), or when the model is small, batch size is small, and memory is sufficient.

## Effects

For Llama-7b, FSDP2 achieves higher MFU compared to FSDP1, reduces peak memory by 7%, and maintains the same loss curve.

## Notes

1. When FSDP2 training is enabled, the distributed optimizer and its related configurations must be disabled.

2. When FSDP2 training is enabled, the model weight save format `ckpt-format` only supports `torch_dist` or `torch_dcp`.

    - When configured as `torch_dist`, the model must implement the `sharded_state_dict()` method by inheriting from `MegatronModule` or through custom implementation; additionally, the 0-dimension size of all weights in the model must be greater than or equal to sharding_size.

    - When configured as `torch_dcp`, the model must implement the `state_dict_for_save_checkpoint()` method by inheriting from `MegatronModule` or through custom implementation, and the weight dictionary returned by this method must be consistent with the return value of `model.state_dict()`.

3. When FSDP2 training is enabled, recomputation-related configurations must be disabled, including `--recompute-granularity`, `--recompute-method`, and `--recompute-num-layers`.
