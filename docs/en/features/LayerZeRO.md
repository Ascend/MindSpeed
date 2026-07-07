# LayerZeRO

## Background and Challenges

Compared to single-type LLMs, multimodal models have more complex architectures. In multimodal scenarios, the TP overhead is much larger than in LLM scenarios, and sharing with DSP introduces additional overhead. Among these, the excessive communication domain of ZeRO3 causes particularly severe extra overhead.

## Solution

To address the problem of excessive overhead caused by the large ZeRO3 communication domain in multimodal scenarios, LayerZeRO is adopted. It applies ZeRO1 sharding to optimizer states and uses ZeRO3 parameters for parameter reconstruction within micro-batches. At runtime, parameters are views of the reconstructed parameters, and the full parameters are destroyed after use. Before each optimizer update, gradients are synchronized to ZeRO1, and after the update, ZeRO1 parameters are synchronized to ZeRO3. This achieves memory savings comparable to TP+ZeRO1 while providing effective communication overlap.

### Approach

#### Communication Groups and Parameter Partitioning

ZeRO1 communication group: The communication group for all data parallel (DP) processes.
ZeRO3 communication group: A subset of the ZeRO1 communication group, consisting of some DP processes.
The LayerZeRO on each device consists of both ZeRO1 and ZeRO3 components.

#### Parameter Partitioning Management

Devices with the same local rank within different ZeRO3 communication groups should store identical parameter shards.
All ZeRO3 shards are partitioned within the DP domain.
ZeRO1 parameter partitioning flattens all managed trainable parameters into a one-dimensional tensor, then evenly distributes it across the devices in the ZeRO1 communication group.
Parameter and gradient synchronization occurs within a single ZeRO1 communication group.

#### Parameter Synchronization and Gradient Synchronization

Register hook functions at various stages of runtime to correctly implement the execution logic:
Reconstruction and destruction of parameter prefetching;
Gradient memory allocation and destruction;
Gradient synchronization;
Optimizer gradient synchronization.

## Application Scenario

In multimodal scenarios where long-sequence activations are much larger than model parameters, enabling LayerZeRO and increasing the CP parallelism degree achieves memory savings similar to TP+ZeRO1 and accelerates long-sequence training.

## Usage

In MindSpeed, the entry point for LayerZero is a configuration file. By generating a configuration file and passing command-line arguments, you can use this feature.

```bash
--layerzero \
--layerzero-config config.yml \
```

Configurable items in config.yml:

```bash
    zero3_size: int  # Size of the ZeRO3 communication group, a positive integer. ZeRO3 is typically performed within a node
    transformer_layers:  Optional[Iterable[torch.nn.Module]] = None   # Class hierarchy name of the wrapped layer: module.submodule.class
    param_dtype: Optional[Literal["fp16", "bf16", "fp32"]] = "fp16"   # Mixed precision related: runtime parameter precision
    reduce_dtype:  Optional[Literal["fp16", "bf16", "fp32"]] = "fp16" # Mixed precision related: runtime gradient precision
    ignored_modules:  Optional[Iterable[str]] = None                  # Parts of the model that do not need to be managed by LayerZeRO. If training is required, users need to customize gradient and parameter synchronization for these parts. For parts of the model that do not need training and whose parameters should not be sharded, this option must be configured. Defaults to None in invalid cases
    offload_grads: bool=False    # Whether to offload full gradients during gradient accumulation
    ckpt_load_path: str=None     # Absolute path for saving checkpoints under the same LayerZeRO configuration, used for resuming training from a checkpoint
    autocast_input: bool = True  # Whether to automatically cast input to mixed precision
    autocast_output: bool = True # Whether to cast the output to fp32
```

NOTE

1. Supports TP, CP, PP (1F1B) parallelism, gradient accumulation, and gradient offloading.
2. The model wrapper is replaced from MegatronModule to LayerZeRO. Parts of the model development that depend on MegatronModule may become invalid.
3. For OpenSoraPlan1.3, you need to additionally set `ignored_modules` to ignore some modules (of type `nn.Module`) to avoid parameter reconstruction for the VAE and `text_encoder` parts. This configuration option extracts the corresponding modules from the model, and you only need to provide the attribute name.
4. CheckPoint serialization uses the pickle component. Unauthorized users must not have write permissions to the UnderFS storage directory and its parent directories. Otherwise, it may lead to CheckPoint tampering, which could cause pickle deserialization injection risks.

## Application Effects

In the MM-OpenSoraPlan1.3 model training scenario, the parallel configuration (CP=8 + LayerZeRO) has memory usage that is basically the same as the baseline (TP=8 + ZeRO1). The experimental group's parallel configuration (CP=8 + LayerZeRO) achieves a 9.7% end-to-end performance improvement compared to the baseline (TP=8 + ZeRO1).
