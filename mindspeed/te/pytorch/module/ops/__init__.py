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

from mindspeed.te.pytorch.module.ops.default_ops import DefaultOps
from mindspeed.te.pytorch.module.ops.mc2_ops import Mc2Ops


def get_ops():
    from mindspeed.args_utils import get_full_args as get_args
    args = get_args()
    if hasattr(args, 'use_ascend_mc2') and args.use_ascend_mc2:
        return Mc2Ops
    else:
        return DefaultOps


class DummyHandle:
    @classmethod
    def wait(cls, *args, **kwargs):
        pass
