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

def restore_expert_input(linear, original, recomputed):
    """Restore a released activation without replacing a saved quantized operand.

    TENPU's dense packed path accepts an early recomputed wgrad input. Its
    quantized path owns a GroupedTensor with forward-time scales; replacing it
    with BF16 would either fail or use the wrong representation. Restore the
    original storage for that path (also covers high-precision backward
    overrides and the compatible, per-expert weight layout).
    """
    if tuple(original.shape) != tuple(recomputed.shape):
        raise ValueError("Recomputed expert input must have the original shape.")
    if original.dtype != recomputed.dtype or original.device != recomputed.device:
        raise ValueError("Recomputed expert input must have the original dtype and device.")
    dense_packed = (
        getattr(linear, 'single_grouped_weight', False)
        and not getattr(linear, 'fp8', False)
        and not getattr(linear, 'debug', False)
    )
    if dense_packed:
        linear.set_recomputed_input_for_delayed_wgrad(recomputed)
    elif original.untyped_storage().data_ptr() != recomputed.untyped_storage().data_ptr():
        original.untyped_storage().resize_(recomputed.untyped_storage().size())
        original.untyped_storage().copy_(recomputed.untyped_storage())
