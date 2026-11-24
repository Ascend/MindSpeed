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

from .modules.layers import linear_backward_wgrad_detach
from .transformer_layer import transformer_layer_forward_backward_overlaping, dualpipev_fb_overlap_mtp_layer_forward
from .transformer_block import transformer_block_fb_overlap_init_wrapper, mtp_block_fb_overlap_forward_wrapper
from .vpp_schedules import forward_backward_pipelining_with_interleaving