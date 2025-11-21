# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any


def dict_torch_dtype_to_str(pretrained_config, d: dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("torch_dtype", None) is not None:
        if isinstance(d["torch_dtype"], dict):
            d["torch_dtype"] = {k: str(v).split(".")[-1] for k, v in d["torch_dtype"].items()}
        elif not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).lower()
    for value in d.values():
        if isinstance(value, dict):
            pretrained_config.dict_torch_dtype_to_str(value)


def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("dtype") is not None:
        if isinstance(d["dtype"], dict):
            d["dtype"] = {k: str(v).split(".")[-1] for k, v in d["dtype"].items()}
        # models like Emu3 can have "dtype" as token in config's vocabulary map,
        # so we also exclude int type here to avoid error in this special case.
        elif not isinstance(d["dtype"], (str, int)):
            d["dtype"] = str(d["dtype"]).lower()
    for value in d.values():
        if isinstance(value, dict):
            self.dict_dtype_to_str(value)