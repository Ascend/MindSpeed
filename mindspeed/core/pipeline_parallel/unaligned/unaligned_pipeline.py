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

from megatron.core import parallel_state


def get_num_layers_to_build_unaligned(config):
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    pipline_num_transformer_layers = config.pipeline_num_transformer_layers
    if vpp_rank is None:
        vpp_rank = 0
    return pipline_num_transformer_layers[pp_rank][vpp_rank]


def get_layer_offset_unaligned(config):
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    pipline_num_transformer_layers = config.pipeline_num_transformer_layers
    if vpp_rank is None:
        vpp_rank = 0
    offsets = get_layer_offset_pp_vp_unaligned(pipline_num_transformer_layers)
    return offsets[pp_rank][vpp_rank]


def get_layer_offset_pp_vp_unaligned(pipline_num_transformer_layers):
    row = len(pipline_num_transformer_layers)
    col = len(pipline_num_transformer_layers[0])
    offsets = []
    for j in range(col):
        for i in range(row):
            offsets.append(pipline_num_transformer_layers[i][j])

    prefix_sum = [0] * (len(offsets) + 1)
    for index, num_layers in enumerate(offsets):
        prefix_sum[index + 1] = prefix_sum[index] + num_layers
    prefix_sum = [[prefix_sum[j * row + i] for j in range(col)] for i in range(row)]
    return prefix_sum