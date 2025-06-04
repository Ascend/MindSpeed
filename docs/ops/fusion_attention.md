# fusion attention 对外接口

### 注意当前若要使用v2版本接口，需要开启`--use-fusion-attn-v2`特性

npu_fusion_attention(
                    query, key, value, head_num,
                    input_layout, *, pse=None,
                    padding_mask=None, atten_mask=None,
                    scale=1., keep_prob=1., pre_tokens=2147483647,
                    next_tokens=2147483647, inner_precise=0, prefix=None,
                    actual_seq_qlen=None, actual_seq_kvlen=None,
                    sparse_mode=0, gen_mask_parallel=True,
                    sync=False, pse_type=1, q_start_idx=None,
                    kv_start_idx=None)

- 计算公式：

   注意力的正向计算公式如下：

   - pse_type=1时，公式如下：

    $$
    attention\\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\\_mask)), keep\\_prob)*value
    $$

   - pse_type=其他取值时，公式如下：

    $$
    attention\\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\\_mask),keep\\_prob)*value
    $$

## 前向接口：
输入：
- query：必选输入，Device侧的Tensor，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND。
- key：必选输入，Device侧的Tensor，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND。
- value：必选输入，Device侧的Tensor，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND。
- atten_mask：可选输入，数据类型bool，缺省none。在softmax之前drop的mask。
- pse：可选输入，Device侧的Tensor，可选参数，表示位置编码。数据类型支持FLOAT16、BFLOAT16，数据格式支持ND。非varlen场景支持四维输入，包含BNSS格式、BN1Skv格式、1NSS格式。如果非varlen场景Sq大于1024或varlen场景、每个batch的Sq与Skv等长且是sparse_mode为0、2、3的下三角掩码场景，可使能alibi位置编码压缩，此时只需要输入原始PSE最后1024行进行内存优化，即alibi_compress = ori_pse[:, :, -1024:, :]，参数每个batch不相同时，输入BNHSkv(H=1024)，每个batch相同时，输入1NHSkv(H=1024)。如果pse_type为2或3的话，需传入数据类型为float32的slope数据，slope数据支持BN或N两种shape。
- padding_mask：可选输入，Device侧的Tensor，暂不支持该参数。
- atten_mask：Device侧的Tensor，可选参数，取值为1代表该位不参与计算（不生效），为0代表该位参与计算，数据类型支持BOOL、UINT8，数据格式支持ND格式，输入shape类型支持BNSS格式、B1SS格式、11SS格式、SS格式。varlen场景只支持SS格式，SS分别是maxSq和maxSkv。
- prefix：Host侧的int array，可选参数，代表prefix稀疏计算场景每个Batch的N值。数据类型支持INT64，数据格式支持ND。
- actual_seq_qlen：Host侧的int array，可选参数，varlen场景时需要传入此参数。表示query每个S的累加和长度，数据类型支持INT64，数据格式支持ND。
  比如真正的S长度列表为：2 2 2 2 2 则actual_seq_qlen传：2 4 6 8 10。
- actual_seq_kvlen：Host侧的int array，可选参数，varlen场景时需要传入此参数。表示key/value每个S的累加和长度。数据类型支持INT64，数据格式支持ND。
  比如真正的S长度列表为：2 2 2 2 2 则actual_seq_kvlen传：2 4 6 8 10。
- sparse_mode：Host侧的int，表示sparse的模式，可选参数。数据类型支持：INT64，默认值为0，支持配置值为0、1、2、3、4、5、6、7、8。当整网的atten_mask都相同且shape小于2048*2048时，建议使用defaultMask模式，来减少内存使用,
  具体可参考昇腾社区说明https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000448.html。
- q_start_idx：Host侧的int array，可选参数，长度为1的int类型数组。pse_type配置为2或3时，表示内部生成alibi编码在Sq方向偏移的格数，正数表示0对角线向上移动。缺省值为0，表示不进行偏移。
- kv_start_idx：Host侧的int array，可选参数，长度为1的int类型数组。pse_type配置为2或3时，表示内部生成alibi编码在Skv方向偏移的格数，正数表示0对角线向左移动。缺省值为0，表示不进行偏移。

输出：
(Tensor, Tensor, Tensor, Tensor, int, int, int)

- 第1个输出为Tensor，计算公式的最终输出y，数据类型支持：FLOAT16、BFLOAT16。
- 第2个输出为Tensor，Softmax 计算的Max中间结果，用于反向计算，数据类型支持：FLOAT。
- 第3个输出为Tensor，Softmax计算的Sum中间结果，用于反向计算，数据类型支持：FLOAT。
- 第4个输出为Tensor，保留参数，暂未使用。
- 第5个输出为int，DSA生成dropoutmask中，Philox算法的seed。
- 第6个输出为int，DSA生成dropoutmask中，Philox算法的offset。
- 第7个输出为int，DSA生成dropoutmask的长度。

属性：
- scale：可选属性，Host侧的double，可选参数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE，默认值为1。
- pse_type：可选属性，Host侧的int，数据类型支持INT64，默认值为1。支持范围0-3。
- pse_type配置为0的时候，pse由外部传入，计算流程是先mul scale再add pse。
- pse_type配置为1的时候，pse由外部传入，计算流程是先add pse再mul scale。
- pse_type配置为2的时候，pse由内部生成，生成标准alibi位置信息。内部生成的alibi矩阵0线与Q@K^T的左上角对齐。
- pse_type配置为3的时候，pse由内部生成，生成的alibi位置信息为标准的基础上再做sqrt开平方。内部生成的alibi矩阵0线与Q@K^T的左上角对齐。
- head_num：必选属性，Host侧的int，代表head个数，数据类型支持INT64。
- input_layout：必选属性，Host侧的string，代表输入query、key、value的数据排布格式，支持BSH、SBH、BSND、BNSD、TND(actual_seq_qlen/actual_seq_kvlen需传值)；后续章节如无特殊说明，S表示query或key、value的sequence length，Sq表示query的sequence length，Skv表示key、value的sequence length，SS表示Sq*Skv
- keep_prob：可选属性，数据类型float，默认值为1.0。在 softmax 后的保留比例。
- pre_tokens：可选属性，Host侧的int，用于稀疏计算的参数，可选参数，数据类型支持INT64，默认值为2147483647。
- next_tokens：可选属性，Host侧的int，用于稀疏计算的参数，可选参数，数据类型支持INT64，默认值为2147483647。
- inner_precise：可选属性，Host侧的int，用于提升精度，数据类型支持INT64，默认值为0。
- gen_mask_parallel：debug参数，DSA生成dropout随机数向量mask的控制开关，默认值为True：同AICORE计算并行，False：同AICORE计算串行
- sync：debug参数，DSA生成dropout随机数向量mask的控制开关，默认值为False：dropout mask异步生成，True：dropout mask同步生成

## 反向接口
输入：
- grad：必选输入，数据类型float16, bfloat16，正向attention_out的梯度输入

输出：
- grad_query：必选输出，数据类型float16, bfloat16
- grad_key：必选输出，数据类型float16, bfloat16	
- grad_value：必选输出，数据类型float16, bfloat16


## 输入限制
- 输入query、key、value的B：batch_size必须相等，取值范围1~2M。非varlen prefix场景B最大支持2K，varlen prefix场景B最大支持1K。
- 输入query、key、value、pse的数据类型必须一致。pse_type=2或3的时候例外，此时pse需要传fp32的slope
- 输入query、key、value的input_layout必须一致。
- 输入query的N和key/value的N 必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。当Nq/Nkv > 1时，即为GQA，当Nkv=1时，即为MQA。
- 输入key/value的shape必须一致。
- 输入query、key、value的S：sequence length，取值范围1~1M。
- 输入query、key、value的D：head dim，取值范围1~512。
- sparse_mode为1、2、3、4、5、6、7、8时，应传入对应正确的atten_mask，否则将导致计算结果错误。当atten_mask输入为None时，sparse_mode，pre_tokens，next_tokens参数不生效，固定为全计算。
- sparse_mode配置为1、2、3、5、6时，用户配置的pre_tokens、next_tokens不会生效。
- sparse_mode配置为0、4时，须保证atten_mask与pre_tokens、next_tokens的范围一致。
- keep_prob的取值范围为(0, 1]。
- varlen场景不支持非压缩prefix，即不支持sparse_mode=5；当Sq>Skv时，prefix的N值取值范围[0, Skv]，当Sq<=Skv时，prefix的N值取值范围[Skv-Sq, Skv]。
- sparse_mode=7或者8时，不支持可选输入pse。
- varlen场景：
  atten_mask输入不支持补pad，即atten_mask中不能存在某一行全1的场景。
  pse仅支持alibi场景，即shape为BNHSkv(H=1024)、1NHSkv(H=1024)两种场景。
- 支持actual_seq_qlen中某个Batch上的S长度为0；如果存在S为0的情况，不支持pse输入， 假设真实的S长度为[2,2,0,2,2]，则传入的actual_seq_qlen为[2,4,4,6,8]；不支持某个batch中Sq不为0，但是Skv为0的场景。

## flash_attention_v2 类的调用方式

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention

batch_size, head_num, seq_size, head_dim = 1, 32, 2048, 128
input_layout = "BNSD"
keep_prob = 1.0
pre_tokens = 65536
next_tokens = 1
scale_value = 0.088
q_scale = 1.0

input_shape = (batch_size, head_num, seq_size, head_dim)
softmax_shape = (batch_size, head_num, seq_size)
mask_shape = (1, 1, seq_size, seq_size)
dtype = torch.float16
dx = (torch.rand(input_shape).to(dtype) - 0.5)
query = (torch.rand(input_shape).to(dtype) - 0.5) * 5
key = (torch.rand(input_shape).to(dtype) - 0.5) * 5
value = (torch.rand(input_shape).to(dtype) - 0.5) * 5
atten_mask = np.tri(seq_size, k=-next_tokens) + np.tri(seq_size, k=-pre_tokens).transpose()
alibi_mask = torch.zeros((batch_size, head_num, seq_size, seq_size)).to(dtype)
atten_mask = torch.from_numpy(atten_mask).to(torch.bool)

query.requires_grad = True
key.requires_grad = True
value.requires_grad = True

#正向接口案例
result = npu_fusion_attention(query.npu(), key.npu(), value.npu(), head_num, input_layout, atten_mask=atten_mask.npu(), scale=scale_value, keep_prob=keep_prob, pre_tokens=pre_tokens, next_tokens=next_tokens, pse_type=1)

#反向接口案例
result[0].backward(dx.npu())
```
