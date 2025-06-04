# ffn对外接口（只支持前向）

npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, Tensor? expert_tokens=None,
        Tensor? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None,
        Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None,
        Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None, Tensor? antiquant_offset2=None,
        int? inner_precise=None, ScalarType? output_dtype=None) -> Tensor

计算逻辑：
  - **非量化场景：**

    $$
    y=activation(x * W1 + b1) * W2 + b2
    $$

  - **量化场景：**

    $$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
    $$

  - **伪量化场景：**

    $$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
    $$

**说明：**
  激活层为geglu/swiglu/reglu时，性能使能需要满足门槛要求，即整网中FFN结构所对应的小算子中vector耗时30us且占比10%以上的用例方可尝试FFN融合算子；或在不知道小算子性能的情况下，尝试使能FFN，若性能劣化则不使能FFN。

## 非量化场景：
输入：
- x：必选输入，公式中的输入x，数据类型int8, float16, bfloat16，支持输入的维度最少是2维[M, K1]，最多是8维
- weight1: 必选输入，专家的权重数据，公式中的W1，数据类型int4, int8, float16, bfloat16，输入在有/无专家时分别为[E, K1, N1]/[K1, N1]
- weight2: 必选输入，专家的权重数据，公式中的W2，数据类型int4, int8, float16, bfloat16，输入在有/无专家时分别为[E, K2, N2]/[K2, N2]
    **说明：**
    M表示token个数，对应transform中的BS(B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度)；K1表示第一组matmul的输入通道数，对应transform中的H(Head-Size)表示隐藏层的大小；N1表示第一组matmul的输出通道数；K2表示第二组matmul的输入通道数；N2表示第二组matmul的输出通道数，对应transform中的H；E表示有专家场景的专家数。
- activation: 必选输入，代表使用的激活函数，公式中的activation，当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu
- expert_tokens: 可选输入，数据类型int64
- expert_tokens_index：可选输入，数据类型int64
    **说明：**
    不能同时输入expert_tokens和expert_tokens_index
    expert_tokens，expert_tokens_index，若不为空时可支持的最大长度为256个
- bias1: 可选输入，权重数据修正值，公式中的b1，数据类型int32, float16, float32，输入在有/无专家时分别为[E, N1]/[N1]
- bias2: 可选输入，权重数据修正值，公式中的b2，数据类型int32, float16, float32，输入在有/无专家时分别为[E, N2]/[N2]
- inner_precise：可选输入，表示高精度或者高性能选择，数据类型支持int64, 该参数仅对float16生效，bfloat16和int8不区分高精度和高性能。
    -   innerPrecise为0时，代表开启高精度模式，算子内部采用float32数据类型计算
    -   innerPrecise为1时，代表高性能模式

输出：
- y：必选输出，数据类型float16, bfloat16

## 全量化场景：
输入：
- x：必选输入，公式中的输入x，数据类型int8, float16, bfloat16，支持输入的维度最少是2维[M, K1]，最多是8维
- weight1: 必选输入，专家的权重数据，公式中的W1，数据类型int4, int8, float16, bfloat16，输入在有/无专家时分别为[E, K1, N1]/[K1, N1]
- weight2: 必选输入，专家的权重数据，公式中的W2，数据类型int4, int8, float16, bfloat16，输入在有/无专家时分别为[E, K2, N2]/[K2, N2]
    **说明：**
    M表示token个数，对应transform中的BS(B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度)；K1表示第一组matmul的输入通道数，对应transform中的H(Head-Size)表示隐藏层的大小；N1表示第一组matmul的输出通道数；K2表示第二组matmul的输入通道数；N2表示第二组matmul的输出通道数，对应transform中的H；E表示有专家场景的专家数。
- activation: 必选输入，代表使用的激活函数，公式中的activation，当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu
- expert_tokens: 可选输入，数据类型int64
- expert_tokens_index：可选输入，数据类型int64
    **说明：**
    不能同时输入expert_tokens和expert_tokens_index
    expert_tokens，expert_tokens_index，若不为空时可支持的最大长度为256个
- bias1: 可选输入，权重数据修正值，公式中的b1，数据类型int32, float16, float32，输入在有/无专家时分别为[E, N1]/[N1]
- bias2: 可选输入，权重数据修正值，公式中的b2，数据类型int32, float16, float32，输入在有/无专家时分别为[E, N2]/[N2]
- scale: 可选输入，量化参数，量化缩放系数，数据类型float32，per-tensor下输入在有/无专家时均为一维向量，输入元素个数在有/无专家时分别为[E]/[1]；per-channel下输入在有/无专家时为二维向量/一维向量，输入元素个数在有/无专家时分别为[E, N1]/[N1]
- offset: 可选输入，量化参数，量化偏移量，数据类型float32，一维向量，输入元素个数在有/无专家时分别为[E]/[1]
- deq_scale1：可选输入，量化参数，第一组matmul的反量化缩放系数，数据类型uint64, int64, float32, bfloat16，输入在有/无专家时分别为[E, N1]/[N1]
- deq_scale2：可选输入，量化参数，第二组matmul的反量化缩放系数，数据类型uint64, int64, float32, bfloat16，输入在有/无专家时分别为[E, N2]/[N2]
- inner_precise：可选输入，表示高精度或者高性能选择，数据类型支持int64, 该参数仅对float16生效，bfloat16和int8不区分高精度和高性能。
    -   innerPrecise为0时，代表开启高精度模式，算子内部采用float32数据类型计算
    -   innerPrecise为1时，代表高性能模式
- output_dtype：可选输入，表示输出y的数据类型，为空时输出y的数据类型为float16，不为空时支持float16, bfloat16

输出：
- y：必选输出，数据类型float16, bfloat16

## 伪量化场景：
输入：
- x：必选输入，公式中的输入x，数据类型int8, float16, bfloat16，支持输入的维度最少是2维[M, K1]，最多是8维
- weight1: 必选输入，专家的权重数据，公式中的W1，数据类型int4, int8, float16, bfloat16，输入在有/无专家时分别为[E, K1, N1]/[K1, N1]
- weight2: 必选输入，专家的权重数据，公式中的W2，数据类型int4, int8, float16, bfloat16，输入在有/无专家时分别为[E, K2, N2]/[K2, N2]
    **说明：**
    M表示token个数，对应transform中的BS(B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度)；K1表示第一组matmul的输入通道数，对应transform中的H(Head-Size)表示隐藏层的大小；N1表示第一组matmul的输出通道数；K2表示第二组matmul的输入通道数；N2表示第二组matmul的输出通道数，对应transform中的H；E表示有专家场景的专家数。
- activation: 必选输入，代表使用的激活函数，公式中的activation，当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu
- expert_tokens: 可选输入，代表各专家的token数，数据类型int64
- expert_tokens_index：可选输入，代表各专家的token数，数据类型int64
    **说明：**
    不能同时输入expert_tokens和expert_tokens_index
    expert_tokens，expert_tokens_index，若不为空时可支持的最大长度为256个
- bias1: 可选输入，权重数据修正值，公式中的b1，数据类型int32, float16, float32，输入在有/无专家时分别为[E, N1]/[N1]
- bias2: 可选输入，权重数据修正值，公式中的b2，数据类型int32, float16, float32，输入在有/无专家时分别为[E, N2]/[N2]
- antiquant_scale1: 可选输入，伪量化参数，第一组matmul的缩放系数，数据类型float16, bfloat16，per-channel下输入在有/无专家时分别为[E, N1]/[N1]，per-in-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]
- antiquant_scale2: 可选输入，伪量化参数，第二组matmul的缩放系数，数据类型float16, bfloat16，per-channel下输入在有/无专家时分别为[E, N2]/[N2]，per-in-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]
- antiquant_offset1: 可选输入，伪量化参数，第一组matmul的偏移量，数据类型float16, bfloat16，per-channel下输入在有/无专家时分别为[E, N1]/[N1]，per-in-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]
- antiquant_offset2: 可选输入，伪量化参数，第二组matmul的偏移量，数据类型float16, bfloat16，per-channel下输入在有/无专家时分别为[E, N2]/[N2]，per-in-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]
    **说明：**
    G表示伪量化per-in-group场景下，antiquantOffsetOptional、antiquantScaleOptional的组数。
- inner_precise：可选输入，表示高精度或者高性能选择，数据类型支持int64, 该参数仅对float16生效，bfloat16和int8不区分高精度和高性能。
    -   innerPrecise为0时，代表开启高精度模式，算子内部采用float32数据类型计算
    -   innerPrecise为1时，代表高性能模式

输出：
- y：必选输出，数据类型float16, bfloat16

## 约束与限制

-   有专家时，专家数据的总数需要与x的M保持一致。
-   激活层为geglu/swiglu/reglu时，仅支持无专家分组时的float16高性能场景（float16场景指类型为aclTensor的必选参数数据类型都为float16的场景），且N1=2\*K2。
-   激活层为gelu/fastgelu/relu/silu时，支持有专家或无专家分组的float16高精度及高性能场景，bfloat16场景，量化场景及伪量化场景，且N1=K2。
-   非量化场景不能输入量化参数和伪量化参数，量化场景不能输入伪量化参数，伪量化场景不能输入量化参数。
-   量化场景参数类型：x为int8、weight为int8、bias为int32、scale为float32、offset为float32，其余参数类型根据y不同分两种情况：
    -   y为float16，deqScale支持数据类型：uint64、int64、float32。
    -   y为bfloat16，deqScale支持数据类型：bfloat16。
    -   要求deqScale1与deqScale2的数据类型保持一致。
-   量化场景支持scale的per-channel模式参数类型：x为int8、weight为int8、bias为int32、scale为float32、offset为float32，其余参数类型根据y不同分两种情况：
    -   y为float16，deqScale支持数据类型：uint64、int64。
    -   y为bfloat16，deqScale支持数据类型：bfloat16。
    -   要求deqScale1与deqScale2的数据类型保持一致。
-   伪量化场景支持两种不同参数类型：
    -   y为float16、x为float16、bias为float16，antiquantScale为float16、antiquantOffset为float16，weight支持数据类型int8和int4。
    -   y为bfloat16、x为bfloat16、bias为float32，antiquantScale为bfloat16、antiquantOffset为bfloat16，weight支持数据类型int8和int4。
-   当weight1/weight2的数据类型为int4时，其shape最后一维必须为偶数。
-   伪量化场景，per-in-group下，antiquantScale1和antiquantOffset1中的K1需要能整除组数G，antiquantScale2和antiquantOffset2中的K2需要能整除组数G。
-   伪量化场景，per-in-group下目前只支持weight是int4数据类型的场景。
-   innerPrecise参数在bfloat16非量化场景，只能配置为0；float16非量化场景，可以配置为0或者1；量化或者伪量化场景，0和1都可配置，但是配置后不生效。

## npu_ffn 接口的调用方式
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