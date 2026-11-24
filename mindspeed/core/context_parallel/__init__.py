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

from megatron.core import mpu
from megatron.core.parallel_state import get_context_parallel_global_ranks
from megatron.core.parallel_state import get_nccl_options
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.utils import divide
from megatron.core.transformer.dot_product_attention import DotProductAttention
from mindspeed.args_utils import get_full_args as get_args