# Copyright 2025 Huawei Technologies Co., Ltd.
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

from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32


def step(self):
    """
    This patch for solve count_zeros automatically used in ChainedOptimizer.
    ChainedOptimizer will step all optimizers one by one.
    """
    found_inf_flag = self.prepare_grads()
    if found_inf_flag:
        return False, None, None

    grad_norm = self.get_grad_norm()

    # Clip gradients.
    for optimizer in self.chained_optimizers:
        if hasattr(optimizer, 'is_stub_optimizer') and optimizer.is_stub_optimizer:
            continue
        if optimizer.config.clip_grad > 0.0:
            clip_grad_by_total_norm_fp32(
                optimizer.get_parameters(),
                max_norm=optimizer.config.clip_grad,
                total_norm=grad_norm,
                use_decoupled_grad=optimizer.config.use_precision_aware_optimizer,
            )

    # Count the zeros in the grads.
    num_zeros_in_grad = 0
    for optimizer in self.chained_optimizers:
        num_zeros_in_grad += (
            optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
        )

    update_successful = self.step_with_ready_grads()

    return update_successful, grad_norm, num_zeros_in_grad
