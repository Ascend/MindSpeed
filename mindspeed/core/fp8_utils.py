# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings


def quantize_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
    """Cast shard fp32 main params to fp8 model params."""

    warnings.warn("Currently, it is not supported to Cast shard fp32 main params to fp8 model params")
