# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd.
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
from megatron.core.transformer.cuda_graphs import is_graph_capturing


def distributed_data_parallel_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        if self.ddp_config.reset_bucket_group_order:
            self.bucket_group_index_list = []
        return res
    return wrapper



def _make_forward_pre_hook(self):
    """
    Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
    when a module uses a parameter in a bucket with a still incomplete all-gather).
    """

    def hook(module, *unused):
        assert (
            self.use_forward_hook
        ), "Should use pre-hook only when overlap_param_gather is True"

        if is_graph_capturing():
            return

        # Make sure all parameters in this module have been all-gathered as necessary.
        for param in module.parameters(recurse=False):
            # Skip parameters without an associated buffer (such parameters have a
            # .requires_grad field equal to False).
            if param not in self.param_to_bucket_group:
                continue
            assert param.requires_grad

            if self.ddp_config.reset_bucket_group_order and self.param_to_bucket_group[
                param] not in self.bucket_group_index_list:
                self.bucket_group_index_list.append(self.param_to_bucket_group[param])
                if len(self.bucket_group_index_list) == len(self.bucket_groups):
                    for i in range(len(self.bucket_group_index_list) - 1):
                        self.bucket_group_index_list[i].next_param_gather_bucket_group = self.bucket_group_index_list[
                            i + 1]

            # If aligning param all-gather across pipeline stages, all-gather is dispatched
            # by start_param_sync calls in core/pipeline_parallelism/schedules.py.
            # If overlapping param all-gather with optimizer step, then all-gather has
            # already been dispatched in optimizer step.
            skip_next_bucket_dispatch = (
                    self.ddp_config.align_param_gather
                    or self.overlap_param_gather_with_optimizer_step
            )
            self.param_to_bucket_group[param].finish_param_sync(
                skip_next_bucket_dispatch=skip_next_bucket_dispatch
            )

    return hook