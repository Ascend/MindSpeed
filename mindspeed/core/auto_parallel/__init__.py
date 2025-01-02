# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from .utils.profiler import AutoProfiler
from .auto_parallel_apply import search_optimal_configuration
from .patch.language_module import compute_language_model_loss_wrapper