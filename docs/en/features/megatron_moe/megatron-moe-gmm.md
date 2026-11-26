# Megatron MoE Grouped GEMM (GMM)

## Background and Challenges

For multi-expert computation on a single card in MoE, there are fragmented expert computation operations and communications. The Grouped GEMM (Grouped General Matrix Multiplication) operator merges multi-expert computations to improve the training performance of multi-expert on a single card in MoE.

## Solution

By calling the gmm fused operator, multiple expert computations are fused to achieve acceleration.

## Usage

**Backend limitation:** `--transformer-impl transformer_engine` must be set; `local` is not supported.

Set `--transformer-impl transformer_engine --moe-grouped-gemm` to enable Grouped GEMM using TEGroupedMLP; supports `allgather` and `alltoall` dispatchers; does not support the `local` backend or `flex`, which has not yet completed NPU adaptation.

`--te-gmm-mode` defaults to `compatible`; when `performance` is selected, packed grouped weight is used.

`--gmm-gradient-accumulation-fusion` (alias `--gemm-gradient-accumulation-fusion`) independently controls expert gradient accumulation fusion.

## Effects

Typical scenarios:

- Scenarios where a smaller EP leads to more experts per device, and DeepSeek MoE has a large number of experts.
- Single small DeepSeek MoE fine-grained expert, FFN scale not large, and larger TP leading to smaller per-card shard computation.

1. As the FFN scale increases, computation is no longer fragmented, single-expert computation efficiency improves, and the benefit of Grouped GEMM diminishes.

    Table 1: Grok model FFN size and performance acceleration comparison

    |ffn_hidden_size| 32768 | 16384| 8192| 4096|
    |--|--|--|--|--|
    |baseline|2280|1780|1537|1446|
    |GEMM|2416|1719|1448|1331|
    |Performance improvement|-5.30%|3.53%|6.12%|8.60%|

2. The larger the TP and the smaller the EP, the greater the benefit.
   
    Table 2: Performance gains for different Mixtral 8x7B model configurations

    | Configuration | tp4 ep2 16expert | tp4 ep2 8expert | tp2 ep4 16expert | tp2 ep4 8expert |
    |--|--|--|--|--|
    | baseline | 27969 | 20127 | 11976 | 13981 |
    | GEMM | 19415 | 17361 | 11049 | 14290 |
    | Performance gain | 44.06% | 17.93% | 8.39% | -2.19% |

## Notes

1. TEGroupedMLP supports `--bf16`; gradient accumulation fusion requires BF16 or FP16 and cannot be used together with FP8/FP4.
2. When Grouped GEMM computation is enabled via the `--moe-grouped-gemm` parameter, the npu_gmm fused operator is invoked.

The operator input and output formats are as follows:

```python
y = npu_gmm(x, weight, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None)
```

Input:

- x: Required input; a tensor; data type: float16, bfloat16, float32
- weight: Required input; a tensor; data type: float16, bfloat16, float32
- bias: Optional input; a tensor; data type: float16, float32; default value: None. In training scenarios, only bias=None is supported.
- group_list: optional input; data type: list[int64], tensor; default value: None.
- group_type: optional input; data type: int64; representing the axis to be grouped. For example, if matrix multiplication is C[m,n]=A[m,k]xB[k,n], then group_type values are -1: no grouping, 0: m-axis grouping, 1: n-axis grouping, 2: k-axis grouping. Default value: 0.
- gemm_fusion: optional input; data type: bool; value: True, False. Used to enable the GMM+ADD fused operator when accumulating gradients in the backward pass. Default value: False.
- original_weight: optional input; a tensor; data type: float16, bfloat16, float32. Used to obtain the main_grad of the weight before view for gradient accumulation in GMM+ADD; Default value: None.

Output:

- y: required output; data type: float16, bfloat16, float32

In non-quantized scenarios, this operator only supports the following combinations of input and output types. Specifying unsupported types for parameters may cause operator errors and affect training efficiency:

| x        | weight   | bias    | group_list         | group_type | gemm_fusion | original_weight | y                                 |
|----------|----------|---------|--------------------|------------|-------------|-----------------|-----------------------------------|
| float16  | float16  | float16 | list[int64] or tensor | int64      | bool        | float16         | float16                           |
| bfloat16 | bfloat16 | float32 | list[int64] or tensor | int64      | bool        | bfloat16        | bfloat16                          |
| float32  | float32  | float32 | list[int64] or tensor | int64      | bool        | float32         | float32 (only supported when x, weight, and y are all single tensors) |
