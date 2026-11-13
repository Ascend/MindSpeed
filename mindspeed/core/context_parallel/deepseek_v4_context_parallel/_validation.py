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


def _resolve_deepseek_v4_seq_dim(layout, seq_dim, name):
    expected = 1 if layout == "BSND" else 0
    if seq_dim is None:
        return expected
    seq_dim = int(seq_dim)
    if seq_dim != expected:
        raise ValueError(f"{name} must be {expected} for layout {layout}, got {seq_dim}.")
    return seq_dim


def _get_deepseek_v4_batch_size(q, layout_q):
    if layout_q == "BSND":
        return int(q.shape[0])
    return 1
