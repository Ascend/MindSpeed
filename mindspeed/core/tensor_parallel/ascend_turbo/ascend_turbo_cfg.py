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

class AscendConfig:
    def __init__(self):
        self.ColumnParallelLinear = None
        self.RowParallelLinear = None
        self.group_func = None
        self.world_size_func = None

        self.sequence_parallel_enabled = True
        self.all_gather_recomputation = True

    def set_sequence_parallel(self, sequence_parallel):
        self.sequence_parallel = sequence_parallel

    def set_all_gather_recomputation(self, all_gather_recomputation):
        self.all_gather_recomputation = all_gather_recomputation

    def set_group(self, group_func):
        self.group_func = group_func

    def get_group(self):
        return self.group_func()

    def set_world_size(self, world_size_func):
        self.world_size_func = world_size_func

    def get_world_size(self):
        return self.world_size_func()

    def set_column_parallel_linear(self, column_parallel_linear):
        self.ColumnParallelLinear = column_parallel_linear

    def set_row_parallel_linear(self, row_parallel_linear):
        self.RowParallelLinear = row_parallel_linear

    def parallel_linear_plugin(self, column_parallel_forward, row_parallel_forward):
        self.ColumnParallelLinear.forward = column_parallel_forward
        self.RowParallelLinear.forward = row_parallel_forward


ascend_turbo_cfg = AscendConfig()
