# Communication Over Computation (CoC)

## Background and Challenges

In large model training, the forward and backward propagation of its ColumnParallelLinear and RowParallelLinear components both involve adjacent, sequentially dependent computation-communication combinations. The computation is Matmul, while the communication is AllReduce (when sequence parallelism is not enabled) or AllGather and ReduceScatter (when sequence parallelism is enabled). These computation-communication combinations are often executed serially due to sequential dependencies (i.e., the input of the latter is the output of the former), but at this point, both the computation and communication streams experience some idle waiting time, and the execution efficiency of this process is not maximized.

## Solution

By splitting computation and communication tasks into finer-grained subtasks to achieve mutual pipeline overlap.

### Approach

#### Python script side implementation

Further split the tensor (into 2/4/8 parts), and use Python scripts to achieve parallelism between computation and communication for each sub-tensor, thereby increasing the utilization of computation and communication streams.

#### Fused operator implementation

Based on the MTE remote memory access capability, a fused large kernel approach is used to split computation and communication tasks into finer-grained sub-tasks within the operator implementation, achieving mutual pipeline overlap.

## Use Scenario

This feature is currently used primarily in training scenarios, and is applicable when the Attention module and MLP module execute serially and there is a sequential dependency and positional adjacency between computation and communication.

When using the Python script-side implementation, there is a certain requirement for the m-axis of the Matmul left matrix: it must be a multiple of the split number (2/4/8), and it is not suitable for cases where the time consumption of computation and communication segments differs significantly. Note that the script-side implementation is prone to host-bound issues when the number of matrix splits is large, thus failing to achieve the expected performance gain. It supports three communication scenarios: ALL_REDUCE, ALL_GATHER, and REDUCE_SCATTER, and allows flexible configuration of whether communication or computation occurs first.

For computation-communication fused operators, the following are currently supported:

1. `MATMUL_ALL_REDUCE` fused operator (computation first, then communication) and its deterministic computation
2. `MATMUL_REDUCE_SCATTER` fused operator (computation first, then communication) and its deterministic computation
3. `ALL_GATHER_MATMUL`, `ALL_GATHER_MATMUL_V2` fused operators (communication first, then computation) (the V2 interface supports retrieving intermediate `ALL_GATHER` results)
4. Quantization scenario: The `MATMUL_ALL_REDUCE` fused operator supports w8A16 pseudo-quantization in fp16 format, with granularities of per tensor, per channel, and per group

## Usage

There are currently two implementation methods for computation-communication parallelism: script enabling and fused operator enabling. You only need to choose one of them. Both methods require replacing the forward functions of the `ColumnParallelLinear` and `RowParallelLinear` classes in the original Megatron framework. The replacement scripts have been coded and adapted according to the Megatron version specified by MindSpeed, and are located in the `mindspeed/core/tensor_parallel/lcal_coc/` directory.

Select one of the following two scenarios as needed.

Set `--use-ascend-coc` to enable the computation-communication parallelism feature. The usage is configured through the following variables:

### 1. Using the CoC feature enabled through the Python script

```shell
--use-ascend-coc
--coc-parallel-num 2 # or 4, or 8r 8
```

### 2. Using the CoC feature enabled by fused operators

Note: The CoC fused operator requires ATB to be installed before use.

ATB installation method:

- Binary package installation: After installing the CANN-NNAL package, `source /usr/local/Ascend/nnal/atb/set_env.sh`

```shell
--use-ascend-coc
--coc-fused-kernel # Note: Currently only supports TP=8 scenarios!only supports TP=8 scenarios!
```

The environment variable for the fused operator has a higher priority, meaning that when coc-parallel-num > 1 and `coc-fused-kernel` is enabled, the former will not take effect.

## CFG Customization Method

You can customize the `coc_cfgs` dictionary in `mindspeed/core/tensor_parallel/lcal_coc/user_config.py` to customize part of the CoC configuration.

Only applicable to the CoC enabled through Python scripts
'matmul_soc_friendly': Whether to perform transpose/padding operations on the input matmul tensors so that they enter the Matmul operator with an NPU-friendly shape, thereby achieving a certain performance gain. Defaults to True;
'customized_coc': Customizes the number of COC splits for matmul of a specified shape. Defaults to `{}`. If you need to set the number of CoC splits for a matmul of a specified shape to 1 (disable COC) or a value different from `coc-parallel-num`, you can follow this example:
'customized_coc': {"[16384, 5120, 1920]": 8, "[16384, 1920, 5120]": 1}

Applicable only to the CoC enabled through fused operators
'enable_coc_in_column_backward': Whether to use COC in the backward pass of `ColumnParallelLinear` (the backward pass of `ColumnParallelLinear` inherently contains non-interdependent CoC), defaults to `False`;

[Applicable to both the script implementation and the fused operator implementation]
'recompute_all_gather': Whether to recompute all-gather in the backward pass of `ColumnParallelLinear`, defaults to True. If set to `False`, the all-gather result from the forward pass will be saved for the backward pass, which reduces backward computation time but increases peak memory usage during training;

## Application Effect of CoC Fused Operators

An end-to-end performance gain of approximately 3.20% is achieved on the BLOOM 7B model, approximately 5.47% on the BLOOM 176B model, and approximately 7.85% on the LLAMA2 70B model. The relative precision error is controlled within a range of 2%.

## Notes

Currently incompatible with the `--use-ascend-mc2` feature. MoE models are not yet supported.
