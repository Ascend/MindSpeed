# TFLOPS Calculation

## Background and Challenges

MFU for large models is currently calculated by dividing the framework-printed TFLOPS value by effective computing power. However, this theoretical formula only applies to standard model architectures. Any structural modification invalidates it. HFU, meanwhile, still requires manual computation.

## Solution

We provide the API that counts the floating-point operations (FLOPs) of all MatMul-related operators. It also supports counting total FLOPs for forward/backward passes and recomputation during training.

### Approach

The proposed FLOP-counting API covers the following MatMul-related operators: `MatMul`, `BatchMatMul`, `FlashAttention`, as well as fused operators for MC2, CoC, GEMM, and matmul_add_fp32.

## Usage

To enable this feature, set `--op-cal-tflops` to invoke it.

## Application Effects

By printing the values `actual throughput per NPU (TFLOP/s/NPU)` and `actual throughput per NPU with recompute (TFLOP/s/NPU)`, you can conveniently calculate MFU and HFU.

## Notes

1. Since this feature collects TFLOPS information for each card, the computation load varies across cards in CP/EP/PP scenarios. Therefore, the information from all cards must be aggregated and averaged at the end, which adds an extra all_reduce communication.

2. Using this feature may affect performance because it adds an extra communication and calculates the floating-point operation count for each operator.

3. In the Ring Attention long-sequence parallelism scheme under causal scenarios, some computations are reduced due to algorithm optimization. This causes a discrepancy between the theoretical value and the actual collected value. The theoretical reduction in FA computation is `(CP-1)/2CP`.

4. This feature does not currently support MLA scenarios.
