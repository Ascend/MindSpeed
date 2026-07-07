# FFN External API (Forward Only)

npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, Tensor? expert_tokens=None,
        Tensor? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None,
        Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None,
        Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None, Tensor? antiquant_offset2=None,
        int? inner_precise=None, ScalarType? output_dtype=None) -> Tensor

Computation Logic:

  - **Non-quantized scenario:**

    $$
    y=activation(x * W1 + b1) * W2 + b2
    $$

  - **Quantized scenario:**

    $$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
    $$

  - **Fake-quantized scenario:**

    $$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
    $$

**NOTE**
  When the activation layer is geglu/swiglu/reglu, performance enablement must meet threshold requirements. That is, the FFN fusion operator can only be attempted for use cases where the vector time of the small operators corresponding to the FFN structure in the entire network is 30us and accounts for more than 10%; or, when the performance of the small operators is unknown, try enabling FFN, and if performance degrades, do not enable FFN.

## Non-quantized scenario

Input:

- x: required input, the input x in the formula, data type int8, float16, bfloat16, supports input with a minimum of 2 dimensions [M, K1] and a maximum of 8 dimensions
- weight1: required input, the expert weight data, W1 in the formula, data type int4, int8, float16, bfloat16, input when there is an expert and when no expert is [E, K1, N1]/[K1, N1] respectively
- weight2: required input, the expert weight data, W2 in the formula, data type int4, int8, float16, bfloat16, input when there is an expert and when no expert is [E, K2, N2]/[K2, N2] respectively

    **NOTE**
    M represents the number of tokens, corresponding to BS in transform (B (Batch) represents the batch size of input samples, S (Seq-Length) represents the sequence length of input samples); K1 represents the input channel of the first group of matmul, corresponding to H (Head-Size) in transform, which represents the hidden size; N1 represents the output channel of the first group of matmul; K2 represents the input channel of the second group of matmul; N2 represents the output channel of the second group of matmul, corresponding to H in transform; E represents the number of experts in the scenario with expert.

- activation: required input, represents the activation function used, activation in the formula, currently supported: fastgelu/gelu/relu/silu and geglu/swiglu/reglu
- expert_tokens: optional input, data type int64
- expert_tokens_index: optional input, data type int64

    **NOTE**
    expert_tokens and expert_tokens_index cannot be simultaneously input.
    If not empty, expert_tokens and expert_tokens_index support a max length of 256.

- bias1: optional input, weight data correction value, b1 in the formula, data type int32, float16, float32, input when there is an expert/scenario with expert is [E, N1]/[N1] respectively
- bias2: optional input, weight data correction value, b2 in the formula, data type int32, float16, float32, input when there is an expert/scenario with expert is [E, N2]/[N2] respectively
- inner_precise: optional input, indicates high-precision or high-performance selection, supported data type int64, this parameter only takes effect for float16, bfloat16 and int8 have no distinction between high precision and high performance.
    - when inner_precise is 0, it indicates enabling high-precision mode, and the operator internal uses float32 data type for computation
    - when inner_precise is 1, it indicates high performance mode

Output:

- y: required output, data type float16, bfloat16

## Quantized scenario

Input:

- x: required input, input x in the formula, data type int8, float16, bfloat16, supports input with a minimum of 2 dimensions [M, K1] and a maximum of 8 dimensions
- weight1: required input, expert weight data, W1 in the formula, data type int4, int8, float16, bfloat16, input when there is/no expert is [E, K1, N1]/[K1, N1] respectively
- weight2: required input, expert weight data, W2 in the formula, data type int4, int8, float16, bfloat16, input when there is/no expert is [E, K2, N2]/[K2, N2] respectively

    **NOTE**
    M represents the number of tokens, corresponding to BS in transform (B (Batch) represents the batch size of input samples, S (Seq-Length) represents the sequence length of input samples); K1 represents the input channel of the first group of matmul, corresponding to H (Head-Size) in transform, which represents the hidden size; N1 represents the output channel of the first group of matmul; K2 represents the input channel of the second group of matmul; N2 represents the output channel of the second group of matmul, corresponding to H in transform; E represents the number of experts in the scenario with expert.

- activation: required input, represents the activation function used, activation in the formula, currently supported are fastgelu/gelu/relu/silu and geglu/swiglu/reglu
- expert_tokens: optional input, data type int64
- expert_tokens_index: optional input, data type int64

    **NOTE**
    expert_tokens and expert_tokens_index cannot be simultaneously input.
    If not empty, expert_tokens and expert_tokens_index support a max length of 256.

- bias1: optional input, weight data correction value, b1 in the formula, data type int32, float16, float32, input when there is an expert / when no expert is [E, N1] / [N1] respectively
- bias2: optional input, weight data correction value, b2 in the formula, data type int32, float16, float32, input when there is an expert / when no expert is [E, N2] / [N2] respectively
- scale: optional input, quantization parameter, quantization scaling coefficient, data type float32, under per-tensor the input is a one-dimensional vector in both scenarios with expert and when no expert, and the number of input elements when there is an expert / when no expert is [E] / [1] respectively; under per-channel the input is a two-dimensional vector / one-dimensional vector in scenarios with expert / when no expert, and the number of input elements when there is an expert / when no expert is [E, N1] / [N1] respectively
- offset: optional input, quantization parameter, quantization offset, data type float32, one-dimensional vector, number of input elements when no expert/scenario with expert is [1]/[E] respectively
- deq_scale1: optional input, quantization parameter, dequantization scaling factor for the first group of matmul, data types uint64, int64, float32, bfloat16, input when no expert/scenario with expert is [N1]/[E, N1] respectively
- deq_scale2: optional input, quantization parameter, dequantization scaling factor for the second group of matmul, data types uint64, int64, float32, bfloat16, input when no expert/scenario with expert is [N2]/[E, N2] respectively
- inner_precise: optional input, represents high-precision or high-performance selection, supported data type int64, this parameter only takes effect for float16, bfloat16 and int8 have no distinction between high precision and high performance.
    - when inner_precise is 0, it indicates enabling high-precision mode, and the operator internal uses float32 data type for computation
    - When inner_precise is 1, it indicates high performance
- output_dtype: optional input, represents the data type of output y. When empty, the data type of output y is float16; if not empty, supports float16, bfloat16

Output:

- y: required output, data type float16, bfloat16

## Fake-quantized scenario

Input:

- x: required input, input x in the formula, data type int8, float16, bfloat16, supports input with a minimum of 2 dimensions [M, K1] and a maximum of 8 dimensions
- weight1: required input, expert weight data, W1 in the formula, data type int4, int8, float16, bfloat16, input when no expert/scenario with expert is [E, K1, N1]/[K1, N1] respectively
- weight2: required input, expert weight data, W2 in the formula, data type int4, int8, float16, bfloat16, input when no expert/scenario with expert is [E, K2, N2]/[K2, N2] respectively

    **Note:**
    M represents the number of tokens, corresponding to BS in transform (B (Batch) represents the batch size of input samples, S (Seq-Length) represents the sequence length of input samples); K1 represents the input channel of the first matmul, corresponding to H (Head-Size) in transform, which represents the hidden size; N1 represents the output channel of the first matmul; K2 represents the input channel of the second matmul; N2 represents the output channel of the second matmul, corresponding to H in transform; E represents the number of experts in the scenario with expert.

- activation: Required input, represents the activation function used, activation in the formula, currently supported: fastgelu/gelu/relu/silu and geglu/swiglu/reglu
- expert_tokens: Optional input, represents the number of tokens for each expert, data type int64
- expert_tokens_index: Optional input, represents the number of tokens for each expert, data type int64

    **NOTE**
    expert_tokens and expert_tokens_index cannot be simultaneously input.
    If not empty, expert_tokens and expert_tokens_index support a max length of 256.

- bias1: Optional input, weight data correction value, b1 in the formula, data type int32, float16, float32, input when there is no expert/scenario with expert is [N1]/[E, N1] respectively
- bias2: optional input, weight data correction value, b2 in the formula, data type int32, float16, float32, input when there is/no expert is [E, N2]/[N2] respectively
- antiquant_scale1: optional input, fake-quantization parameter, scaling factor of the first group of matmul, data type float16, bfloat16, input when there is/no expert under per-channel is [E, N1]/[N1] respectively, input when there is/no expert under per-in-group is [E, G, N1]/[G, N1] respectively
- antiquant_scale2: optional input, fake-quantization parameter, scaling factor of the second group of matmul, data type float16, bfloat16, input when there is/no expert under per-channel is [E, N2]/[N2] respectively, input when there is/no expert under per-in-group is [E, G, N2]/[G, N2] respectively
- antiquant_offset1: optional input, fake-quantization parameter, offset of the first group of matmul, data type float16, bfloat16, input when there is/no expert under per-channel is [E, N1]/[N1] respectively, input when there is/no expert under per-in-group is [E, G, N1]/[G, N1] respectively
- antiquant_offset2: optional input, fake-quantization parameter, offset of the second group of matmul, data type float16, bfloat16, input when there is/no expert under per-channel is [E, N2]/[N2] respectively, input when there is/no expert under per-in-group is [E, G, N2]/[G, N2] respectively

    **NOTE**
    G represents the number of groups for antiquantOffsetOptional and antiquantScaleOptional in the fake-quantized per-in-group scenario.

- inner_precise: optional input, represents high-precision or high-performance selection, data type supports int64, this parameter only takes effect for float16; bfloat16 and int8 have no distinction between high precision and high performance.
    - When inner_precise is 0, it indicates enabling high-precision mode, and the operator internal uses float32 data type for computation
    - When inner_precise is 1, it indicates high performance mode

Output:

- y: required output, data type float16, bfloat16

## Constraints

- When there is an expert, the total number of expert data must be consistent with M of x.
- When the activation layer is geglu/swiglu/reglu, only the float16 high-performance scenario without expert grouping is supported (the float16 scenario refers to the scenario where the data types of all required parameters of type aclTensor are float16), and N1=2\*K2.
- When the activation layer is gelu/fastgelu/relu/silu, the float16 high-precision and high-performance scenarios, bfloat16 scenario, quantized scenario, and fake-quantized scenario with or without expert grouping are supported, and N1=K2.
- In the non-quantized scenario, quantization parameters and fake-quantization parameters cannot be input; in the quantized scenario, fake-quantization parameters cannot be input; in the fake-quantized scenario, quantization parameters cannot be input.
- Parameter types in the quantized scenario: x is int8, weight is int8, bias is int32, scale is float32, offset is float32, and the remaining parameter types fall into two cases depending on y:
    - When y is float16, deqScale supports data types: uint64, int64, float32.
    - When y is bfloat16, deqScale supports data types: bfloat16.
    - The data types of deqScale1 and deqScale2 must be consistent.
- The quantized scenario supports the per-channel mode parameter types for scale: x is int8, weight is int8, bias is int32, scale is float32, offset is float32, and the remaining parameter types fall into two cases depending on y:
    - When y is float16, deqScale supports data types: uint64, int64.
    - When y is bfloat16, deqScale supports data types: bfloat16.
    - The data types of deqScale1 and deqScale2 must be consistent.
- The fake-quantized scenario supports two different parameter types:
    - y is float16, x is float16, bias is float16, antiquantScale is float16, antiquantOffset is float16, and weight supports data types int8 and int4.
    - y is bfloat16, x is bfloat16, bias is float32, antiquantScale is bfloat16, antiquantOffset is bfloat16, and weight supports data types int8 and int4.
- When the data type of weight1/weight2 is int4, the last dimension of its shape must be an even number.
- In the fake-quantized scenario, under per-in-group, K1 in antiquantScale1 and antiquantOffset1 must be divisible by the number of groups G, and K2 in antiquantScale2 and antiquantOffset2 must be divisible by the number of groups G.
- In the fake-quantized scenario, under per-in-group, only scenarios where weight is of int4 data type are currently supported.
- In the bfloat16 non-quantized scenario, the inner_precise parameter can only be configured as 0; in the float16 non-quantized scenario, it can be configured as 0 or 1; in quantized or fake-quantized scenarios, both 0 and 1 can be configured, but the configuration does not take effect.

## Example Call of npu_ffn

```python
import os
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from mindspeed.ops import ffn

os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

ACTIVATION_TYPE = 'relu'

EXPERT_TOKENS_MODE_NONE = 0
EXPERT_TOKENS_MODE_TOKENS = 1
EXPERT_TOKENS_MODE_TOKENS_INDEX = 2


class FFNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight1, weight2, activation, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=0):
        return ffn.npu_ffn(x, weight1, weight2, activation,
            expert_tokens=expert_tokens, expert_tokens_index=expert_tokens_index,
            bias1=bias1, bias2=bias2, inner_precise=inner_precise)


def test_ffn(tokens_mode, is_graph_mode=True):
    M = 512
    K1 = 256
    N1 = 1024
    K2 = N1
    N2 = K1

    dtype = torch.float16
    bias_dtype = torch.float16 if dtype == torch.float16 else torch.float32

    expert_tokens = None
    expert_tokens_index = None

    if tokens_mode == EXPERT_TOKENS_MODE_NONE:
        x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
        weight1 = torch.empty(K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
        weight2 = torch.empty(K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
        bias1 = torch.empty(N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
        bias2 = torch.empty(N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
    elif tokens_mode == EXPERT_TOKENS_MODE_TOKENS:
        E = 8
        x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
        weight1 = torch.empty(E, K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
        weight2 = torch.empty(E, K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
        bias1 = torch.empty(E, N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
        bias2 = torch.empty(E, N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
        expert_tokens = [64, 64, 64, 64, 64, 64, 64, 64]
        expert_tokens = torch.tensor(expert_tokens, dtype=torch.int64)
    elif tokens_mode == EXPERT_TOKENS_MODE_TOKENS_INDEX:
        E = 8
        x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
        weight1 = torch.empty(E, K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
        weight2 = torch.empty(E, K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
        bias1 = torch.empty(E, N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
        bias2 = torch.empty(E, N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
        expert_tokens_index = [64, 128, 192, 256, 320, 384, 448, 512]
        expert_tokens_index = torch.tensor(expert_tokens_index, dtype=torch.int64)

    x = x.npu()
    weight1 = weight1.npu()
    weight2 = weight2.npu()
    bias1 = bias1.npu()
    bias2 = bias2.npu()

    if expert_tokens != None:
        expert_tokens = expert_tokens.npu()
    if expert_tokens_index != None:
        expert_tokens_index = expert_tokens_index.npu()

    if is_graph_mode:
        model = FFNModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        y = model(x, weight1, weight2, ACTIVATION_TYPE, expert_tokens=expert_tokens,
            expert_tokens_index=expert_tokens_index, bias1=bias1, bias2=bias2)
    else:
        y = ffn.npu_ffn(x, weight1, weight2, ACTIVATION_TYPE, expert_tokens=expert_tokens,
                expert_tokens_index=expert_tokens_index, bias1=bias1, bias2=bias2)
    print('y.shape:', y.shape)


if __name__ == '__main__':
    test_ffn(EXPERT_TOKENS_MODE_NONE, True)
    test_ffn(EXPERT_TOKENS_MODE_TOKENS, True)
    test_ffn(EXPERT_TOKENS_MODE_TOKENS_INDEX, True)
    test_ffn(EXPERT_TOKENS_MODE_NONE, False)
    test_ffn(EXPERT_TOKENS_MODE_TOKENS, False)
    test_ffn(EXPERT_TOKENS_MODE_TOKENS_INDEX, False)
```
