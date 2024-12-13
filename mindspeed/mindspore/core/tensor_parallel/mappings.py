# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import mindtorch.torch as torch
from mindspeed.mindspore.mindspore_adaptor import bprop_commn


def mapping_reduce(input_):
    """All-reduce the input tensor across model parallel group."""
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
    input_ = input_.detach()
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def reduce_from_model_parallel_region_bprop(self, input_, output, grad_output):
    return bprop_commn(self, grad_output)


def scatter_to_sequence_parallel_region_bprop(self, input_, output, grad_output):
    return bprop_commn(self, grad_output)


def reduce_scatter_to_sequence_parallel_region_bprop(self, input_, output, grad_output):
    return bprop_commn(self, grad_output)


def gather_from_model_parallel_region_bprop(self, input, output, grad_output):
    return bprop_commn(self, grad_output)


def copy_to_model_parallel_region_bprop(self, input, output, grad_output):
    return bprop_commn(self, grad_output)


def scatter_to_model_parallel_region_bprop(self, input, output, grad_output):
    return bprop_commn(self, grad_output)


def gather_from_sequence_parallel_region_bprop(self, input, tensor_parallel_output_grad, output, grad_output):
    return bprop_commn(self, grad_output)
