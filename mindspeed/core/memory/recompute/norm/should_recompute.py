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
from mindspeed.core.memory.recompute.recompute_common import should_recompute
from mindspeed.core.pipeline_parallel.ripipe_schedules import get_pipeline_checkpoint_manager


def should_recompute_norm(layer_number, config):
    if not config.recompute_norm or layer_number is None:
        return False

    if config.recompute_in_bubble or config.recompute_in_advance:
        pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(config.virtual_pipeline_model_parallel_size)
        if pipeline_checkpoint_manager.chunk_do_recompute:
            return False
        elif config.recompute_in_bubble:
            return True
    
    return should_recompute(config, layer_number, config.recompute_norm_num_layers)
