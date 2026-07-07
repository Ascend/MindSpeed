# Fusion Attention External API

## Enabling the `--use-fusion-attn-v2` Feature for the V2 Version

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

- Formula:

   The forward calculation formula for attention is as follows:

   - When pse_type=1, the formula is as follows:

    $$
    attention\\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\\_mask)), keep\\_prob)*value
    $$

   - When pse_type is set to other values, the formula is as follows:

    $$
    attention\\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\\_mask),keep\\_prob)*value
    $$

## Forward API

Inputs:

- query: Required Input, a tensor on the Device side. Data Type supports FLOAT16 and BFLOAT16. Data Format supports ND.
- key: Required Input, a tensor on the Device side. Data Type supports FLOAT16 and BFLOAT16. Data Format supports ND.
- value: Required Input, a tensor on the Device side. Data Type supports FLOAT16 and BFLOAT16. Data Format supports ND.
- atten_mask: Optional Input, Data Type bool, default None. The mask for dropout before softmax.
- pse: Optional input, a tensor on the Device side, optional parameter, representing positional encoding. Data types supported: FLOAT16, BFLOAT16. Data format supported: ND. In non-varlen scenarios, four-dimensional input is supported, including BNSS format, BN1Skv format, and 1NSS format. If Sq is greater than 1024 in non-varlen scenarios, or in varlen scenarios where Sq and Skv of each batch are equal and it is a lower triangular mask scenario with sparse_mode 0, 2, or 3, alibi positional encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input for memory optimization, i.e., alibi_compress = ori_pse[:, :, -1024:, :]. When the parameter differs for each batch, input BNHSkv (H=1024); when it is the same for each batch, input 1NHSkv (H=1024). If pse_type is 2 or 3, slope data of data type float32 must be passed in, and the slope data supports two shapes: BN or N.
- padding_mask: Optional input, a tensor on the Device side. This parameter is not currently supported.
- atten_mask: A tensor on the Device side, optional parameter. A value of 1 indicates that the position does not participate in the calculation (not effective), and a value of 0 indicates that the position participates in the calculation. Data types supported: BOOL, UINT8. Data format supported: ND format. Input shape types supported: BNSS format, B1SS format, 11SS format, SS format. In varlen scenarios, only the SS format is supported, where SS represents maxSq and maxSkv respectively.
- prefix: An int array on the Host side, optional parameter, representing the N value for each batch in the prefix sparse calculation scenario. Data type supported: INT64. Data format supported: ND.
- actual_seq_qlen: An int array on the Host side, optional parameter. This parameter must be passed in varlen scenarios. It represents the cumulative sum length of each S in the query. Data type supported: INT64. Data format supported: ND.
For example, if the actual S length list is: 2 2 2 2 2, then actual_seq_qlen is passed as: 2 4 6 8 10.
- actual_seq_kvlen: An int array on the Host side, optional parameter, required in varlen scenarios. It represents the cumulative sum length of each S for key/value. The supported data type is INT64, and the supported data format is ND.
For example, if the actual S length list is: 2 2 2 2 2, then actual_seq_kvlen is passed as: 2 4 6 8 10.
- sparse_mode: An int on the Host side, indicating the sparse mode, optional parameter. Supported data type: INT64. Default value: 0. Supported configuration values are 0, 1, 2, 3, 4, 5, 6, 7, 8. When the atten_mask is the same across the entire network and its shape is smaller than 2048*2048, it is recommended to use the defaultMask mode to reduce memory usage.
For details, refer to the Ascend Community documentation at <https://www.hiascend.com/document/detail/zh/Pytorch/26.0.0/apiref/apilist/ptaoplist_000448.html.>
- q_start_idx: An int array on the host side, an optional parameter, which is an int array of length 1. When pse_type is configured as 2 or 3, it indicates the number of cells to shift the internally generated alibi encoding in the Sq direction, where a positive number means the zero diagonal moves upward. The default value is 0, indicating no shift.
- kv_start_idx: An int array on the host side, an optional parameter, which is an int array of length 1. When pse_type is configured as 2 or 3, it indicates the number of cells to shift the internally generated alibi encoding in the Skv direction, where a positive number means the zero diagonal moves leftward. The default value is 0, indicating no shift.

Output:
(Tensor, Tensor, Tensor, Tensor, int, int, int)

- The 1st output is a Tensor, the final output y of the calculation formula. Supported data types: FLOAT16, BFLOAT16.
- The 2nd output is a Tensor, the Max intermediate result of the Softmax calculation, used for backward computation. Supported data type: FLOAT.
- The 3rd output is a Tensor, the intermediate sum result of the Softmax calculation, used for backward computation. Supported data type: FLOAT.
- The 4th output is a Tensor, a reserved parameter, currently unused.
- The 5th output is an int, the seed of the Philox algorithm for generating the dropout mask in DSA.
- The 6th output is an int, the offset of the Philox algorithm for generating the dropout mask in DSA.
- The 7th output is an int, the length of the dropout mask generated in DSA.

Attributes:

- scale: Optional attribute, a double on the Host side, representing the scaling factor used as the scalar value for the Muls operation in the computation flow. The data type supports DOUBLE. Default value: 1.
- pse_type: Optional attribute, an int on the Host side. The data type supports INT64. Default value: 1. The supported range is 0-3.
- When pse_type is configured as 0, pse is passed in externally, and the computation flow first performs mul scale and then add pse.
- When pse_type is configured as 1, pse is passed in externally, and the computation flow first performs add pse and then mul scale.
- When pse_type is configured as 2, pse is generated internally, producing standard alibi position information. The 0th line of the internally generated alibi matrix aligns with the top-left corner of Q@K^T.
- When pse_type is configured as 3, pse is generated internally, producing alibi position information that is the square root of the standard version. The 0th line of the internally generated alibi matrix aligns with the top-left corner of Q@K^T.
- head_num: Required attribute, an int on the Host side, representing the number of heads. Data type supports INT64.
- input_layout: Required attribute, a string on the Host side, representing the data layout format of the input query, key, and value. Supports BSH, SBH, BSND, BNSD, TND (actual_seq_qlen/actual_seq_kvlen must be passed); in subsequent chapters, unless otherwise specified, S represents the sequence length of query or key/value, Sq represents the sequence length of query, Skv represents the sequence length of key/value, and SS represents Sq*Skv.
- keep_prob: Optional attribute, data type float, default value is 1.0. The retention ratio after softmax.
- pre_tokens: Optional attribute, int on the Host side, parameter for sparse computation, optional parameter, data type supports INT64, default value is 2147483647.
- next_tokens: Optional attribute, int on the Host side, parameter for sparse computation, optional parameter, data type supports INT64, default value is 2147483647.
- inner_precise: Optional attribute, int on the Host side, used to improve precision, data type supports INT64, default value is 0.
- gen_mask_parallel: Debug parameter, control switch for DSA generating the dropout random number vector mask, default value is True: parallel with AICORE computation, False: serial with AICORE computation
- sync: Debug parameter, control switch for DSA generating the dropout random number vector mask, default value is False: dropout mask generated asynchronously, True: dropout mask generated synchronously

## Backward API

Input:

- grad: Required Input, Data Type float16, bfloat16, gradient input of the forward attention_out

Output:

- grad_query: Required Output, Data Type float16, bfloat16
- grad_key: Required Output, data type float16, bfloat16
- grad_value: Required Output, data type float16, bfloat16

## Input Constraints

- The B (batch_size) of the inputs query, key, and value must be equal, with a value range of 1 to 2M. In non-varlen prefix scenarios, B supports a maximum of 2K; in varlen prefix scenarios, B supports a maximum of 1K.
- The data types of the inputs query, key, value, and pse must be consistent. An exception is when pse_type is 2 or 3, in which case pse must pass an fp32 slope.
- The input_layout of query, key, and value must be consistent.
- The N of query and the N of key/value must be in a proportional relationship, meaning Nq/Nkv must be a non-zero integer, with Nq in the value range of 1 to 256. When Nq/Nkv > 1, it is GQA; when Nkv=1, it is MQA.
- The shapes of key and value must be consistent.
- S (sequence length) of query, key, and value: value range is 1 to 1M.
- D (head dim) of query, key, and value: value range is 1 to 512.
- When sparse_mode is 1, 2, 3, 4, 5, 6, 7, or 8, the correct corresponding atten_mask must be passed in; otherwise, the calculation result will be incorrect. When the atten_mask input is None, the sparse_mode, pre_tokens, and next_tokens parameters do not take effect, and full computation is fixed.
- When sparse_mode is configured as 1, 2, 3, 5, or 6, the user-configured pre_tokens and next_tokens do not take effect.
- When sparse_mode is configured as 0 or 4, ensure that the ranges of atten_mask, pre_tokens, and next_tokens are consistent.
- The value range of keep_prob is (0, 1].
- The varlen scenario does not support uncompressed prefix, meaning sparse_mode=5 is not supported. When Sq > Skv, the N value range of prefix is [0, Skv]; when Sq <= Skv, the N value range of prefix is [Skv - Sq, Skv].
- When sparse_mode=7 or 8, the optional input pse is not supported.
- Varlen scenario:
  The atten_mask input does not support padding, meaning that no row in atten_mask can be all ones.
  pse only supports the alibi scenario, i.e., shapes BNHSkv (H=1024) and 1NHSkv (H=1024).
- Supports the case where the S length of a batch in actual_seq_qlen is 0. If any S is 0, pse input is not supported. Assuming the actual S lengths are [2,2,0,2,2], the passed actual_seq_qlen would be [2,4,4,6,8]. The scenario where Sq is not 0 but Skv is 0 for a batch is not supported.

## Example Call of the flash_attention_v2 Class

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

# Forward call example
result = npu_fusion_attention(query.npu(), key.npu(), value.npu(), head_num, input_layout, atten_mask=atten_mask.npu(), scale=scale_value, keep_prob=keep_prob, pre_tokens=pre_tokens, next_tokens=next_tokens, pse_type=1)

# Backward call example
result[0].backward(dx.npu())
```
