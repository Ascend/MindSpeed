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

AUTO_CAST_INPUT = True
AUTO_CAST_OUTPUT = True


def set_auto_cast_input(state: bool):
    global AUTO_CAST_INPUT
    if not isinstance(state, bool):
        raise AssertionError("state must be a boolean")
    AUTO_CAST_INPUT = state


def set_auto_cast_output(state: bool):
    global AUTO_CAST_OUTPUT
    if not isinstance(state, bool):
        raise AssertionError("state must be a boolean")
    AUTO_CAST_OUTPUT = state
