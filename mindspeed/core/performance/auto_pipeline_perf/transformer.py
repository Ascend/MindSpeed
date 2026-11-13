# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from megatron.training import get_args
from mindspeed.model.transformer import generate_attention_mask
import mindspeed.model.transformer


def get_attention_mask_wrapper(get_attention_mask):
    @wraps(get_attention_mask)
    def wrapper(*args, **kwargs):
        argument = get_args()
        automated_pipeline_perf = argument.automated_pipeline_perf and argument.optimized_mbs_list
        if automated_pipeline_perf:
            generate_attention_mask()
        else:
            get_attention_mask(*args, **kwargs)
        return mindspeed.model.transformer._GLOBAL_ATTN_MASK
    return wrapper