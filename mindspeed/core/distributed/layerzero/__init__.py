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

from .zero3 import LayerZeRO3
from .config import layerzero_setup_model_and_optimizer, layerzero_initialize_model_parallel_wrapper
from .megatron_adaptor import mga_finalize_model_grads
from .state.mga_checkpoint import save_checkpoint
