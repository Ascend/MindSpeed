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

_COC_CFGS = {
    'recompute_all_gather': True,
    'matmul_soc_friendly': True,
    'print_tensor_value_open': False,
    'customized_coc': {},
    'enable_coc_in_column_backward': False,
    'k_min': 1024,
    'k_max': 4096,
}


def get_value_from_cfg(attr_name):
    return _COC_CFGS[attr_name]
