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

import torch


class PipelineCheckpointManager:
    instance = None

    def __init__(self, num_of_chunks=2):
        self.open_ri_pipe = False
        self.do_pre_recompute = False
        self.checkpoint_list = []
        self.chunk_list = [[] for i in range(num_of_chunks)]
        self.chunk_do_recompute = True

    def batch_fin(self, chunk_idx):
        self.chunk_list[chunk_idx].append(self.checkpoint_list)
        self.checkpoint_list = []

    def iter_fin(self):
        if len(self.checkpoint_list) != 0:
            raise RuntimeError("recompute list is not empty")

        for batch_list_for_chunk in self.chunk_list:
            for layer_list_for_batch in batch_list_for_chunk:
                if len(layer_list_for_batch) != 0:
                    raise RuntimeError(
                        f"{torch.distributed.get_rank()} batch recompute list is not empty, got {layer_list_for_batch[0]} in index 0")
            batch_list_for_chunk.clear()
        self.open_ri_pipe = False
        self.do_pre_recompute = False
        self.chunk_do_recompute = True

    def add_recompute(self, recompute_func):
        self.checkpoint_list.append(recompute_func)

    def recompute_next_force(self, chunk_idx):
        layer_list_for_head_batch = []
        while len(layer_list_for_head_batch) == 0:
            if len(self.chunk_list[chunk_idx]) == 0:
                print(f"rank-{torch.distributed.get_rank()} warning, poping empty list")
                break
            layer_list_for_head_batch = self.chunk_list[chunk_idx].pop(0)

        for layer_func in layer_list_for_head_batch:
            layer_func()

    def recompute_next(self, chunk_idx):
        layer_list_for_head_batch = self.chunk_list[chunk_idx].pop(0)
        if self.do_pre_recompute:
            for layer_func in layer_list_for_head_batch:
                layer_func()

    def disable_recompute(self):
        self.chunk_do_recompute = False

    def enable_recompute(self):
        self.chunk_do_recompute = True


def get_pipeline_checkpoint_manager(num_of_chunks=2):
    if PipelineCheckpointManager.instance is None:
        PipelineCheckpointManager.instance = PipelineCheckpointManager(num_of_chunks=num_of_chunks)
    return PipelineCheckpointManager.instance
