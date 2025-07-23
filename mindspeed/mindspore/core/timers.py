# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.


def _get_global_min_max_time(self, names, reset, barrier, normalizer):
    """Report only min and max times across all ranks."""

    rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
    rank_name_to_time = rank_name_to_time.numpy()
    name_to_min_max_time = {}
    for i, name in enumerate(names):
        rank_to_time = rank_name_to_time[:, i]
        # filter out the ones we did not have any timings for
        rank_to_time = rank_to_time[rank_to_time > 0.0]
        # If the timer exists:
        # if rank_to_time.numel() > 0:
        if rank_to_time.size > 0:
            name_to_min_max_time[name] = (
                rank_to_time.min() / normalizer,
                rank_to_time.max() / normalizer,
            )
    return name_to_min_max_time
