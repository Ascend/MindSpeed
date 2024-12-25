# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import List

import numpy as np

from .swap_policy_config import swap_policy_config
from .swap_utils import print_with_rank, PrintLevel
from .swap_cpp_adaptor import ProfilerDataOneStep, SwapPolicyCandidate, SwapStage


class TensorArrangerBase:
    def __init__(self, profiler_op_step: ProfilerDataOneStep, output_file_path, duration_time):
        self.op_list = profiler_op_step.op_list
        self.profiler_op_step = profiler_op_step
        self.duration_time = duration_time
        self.stage_data = []
        self.stage_map = {}
        self.stage_index_map = {}
        self.stage_start_time = dict()
        self.stage_end_time = dict()
        self.set_data()
        self.candidate_index = 0
        self.output_file_path = output_file_path

        self.D2H_bandwidth = swap_policy_config.D2H_bandwidth
        self.H2D_bandwidth = swap_policy_config.H2D_bandwidth
        self.color_map = {
            "SwapStageType.INIT": "yellow",
            "SwapStageType.FWD": "red",
            "SwapStageType.BWD": "blue",
            "SwapStageType.OPTIM": "purple",
            "Delay": "green",
        }

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="SwapEngine", print_level=print_level)

    def set_data(self):
        time_line = list(np.linspace(0, self.duration_time, len(self.op_list) + 1))[1:]
        for index, op in enumerate(self.op_list):
            if not self.stage_data or op.stage != self.stage_data[-1]["stage"]:
                self.stage_data.append(
                    {
                        "op_id": op.op_id,
                        "stage": op.stage,
                        "stage_type": str(op.stage.stage_type),
                        "start_time": time_line[index],
                        "type": "op_stream",
                        "candidate_index": -1,
                    }
                )
            self.stage_data[-1]["end_time"] = time_line[index]
        for index, row in enumerate(self.stage_data):
            if row["stage"] in self.stage_map:
                raise ValueError("Find duplicate stage ...")
            self.stage_index_map[index] = row["stage"]
            self.stage_map[row["stage"]] = {
                "index": index,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "layer_time": row["end_time"] - row["start_time"],
                "time_left": row["end_time"] - row["start_time"],
                "candidate_list": [],
            }
            self.stage_end_time[index] = row["end_time"]
            self.stage_start_time[index] = row["start_time"]

    def get_swap_time(self, size):
        swap_out_time = size / 1024 / 1024 / self.D2H_bandwidth
        swap_in_time = size / 1024 / 1024 / self.H2D_bandwidth
        return swap_out_time, swap_in_time

    def reset_simulation(self):
        self.candidate_index = 0
        for stage in self.stage_map:
            self.stage_map[stage]["candidate_list"] = []
            self.stage_map[stage]["time_left"] = self.stage_map[stage]["layer_time"]
        self.stage_time_left = dict()

    def set_swapin_free_stage_to_candidate(self, cur_time, candidate):
        swap_in_free_stage_index = self.stage_map[candidate.swap_in_stage_actual]["index"]
        # 首先swap_in后实际释放的时机设置为swap_in_stage_actual的后一个
        # 由于在排布时为了减少实际执行中计算流等待swap流swap in的情况，
        # 所有candidate的swap_in_stage_actual设置都至少比理论上计算流需要的stage提前了一个stage
        # 因此这里将所有candidate实际swap in释放的stage设置为swap_in_stage_actual的后一个stage，一定不会超出所有stage的边界
        candidate.swap_in_free_stage = self.stage_index_map[swap_in_free_stage_index + 1]
        for index, stage in self.stage_index_map.items():
            # 如果当前candidate在排布中实际swap in结束时间所在stage，
            # 加上延迟free的stage数后没有超过总stage数边界，
            # 则将实际swap in 释放stage设置为排布获得的swap in结束时间所在stage再往后延swap_in_free_stage_delay个stage
            if (
                index < len(self.stage_index_map) - swap_policy_config.swap_in_free_stage_delay
                and cur_time < self.stage_end_time[index]
            ):
                candidate.swap_in_free_stage = self.stage_index_map[index + swap_policy_config.swap_in_free_stage_delay]
                return

    def set_free_stage_to_candidate(self, cur_time, candidate):
        candidate.free_stage = candidate.swap_in_stage_actual
        for index, _ in self.stage_index_map.items():
            if (
                index < len(self.stage_index_map) - swap_policy_config.free_stage_delay
                and cur_time < self.stage_end_time[index]
            ):
                candidate.free_stage = self.stage_index_map[index + swap_policy_config.free_stage_delay]
                return

    def set_free_stage(self):
        for index, stage in self.stage_index_map.items():
            value = self.stage_map[stage]
            time_left = self.stage_time_left[value["index"]]
            start_time = self.stage_start_time[index] - time_left + value["time_left"]
            cur_time = start_time

            # Initialize an empty list to store swap information for each candidate
            swap_list_out_opid = []

            # Iterate through each item in the candidate list
            for swap_stage, swap_time, stream_type, candidate_index, candidate in value["candidate_list"]:
                # Determine operation ID based on candidate type
                if candidate.is_optimizer_or_weight:
                    op_id = self.profiler_op_step.layer_start_opid[candidate.swap_out_stage]
                else:
                    op_id = candidate.swap_out_op.op_id

                # Append a tuple with the relevant information to the list
                swap_list_out_opid.append((swap_stage, swap_time, stream_type, candidate_index, candidate, op_id))

            swap_list_out_opid = sorted(swap_list_out_opid, key=lambda item: (item[-1], -item[-2].tensor.info.size))
            value["candidate_list"] = [
                (swap_stage, swap_time, stream_type, candidate_index, candidate)
                for swap_stage, swap_time, stream_type, candidate_index, candidate, _ in swap_list_out_opid
            ]

            for swap_stage, swap_time, stream_type, candidate_index, candidate in value["candidate_list"]:
                cur_time += swap_time
                if stream_type == "swap_out_stream":
                    self.set_free_stage_to_candidate(cur_time, candidate)
                elif stream_type == "swap_in_stream":
                    self.set_swapin_free_stage_to_candidate(cur_time, candidate)


class TensorArranger(TensorArrangerBase):
    def __init__(self, profiler_op_step: ProfilerDataOneStep, output_file_path, duration_time):
        super(TensorArranger, self).__init__(profiler_op_step, output_file_path, duration_time)
        self.profiler_op_step = profiler_op_step
        self.stage_time_left = dict()

    def calculate_time_left(self, find_index):
        time_left = 0
        for index in range(find_index + 1):
            time_left = min(0, time_left)
            time_left += self.stage_map[self.stage_index_map[index]]["time_left"]
        return time_left

    def save_stage_time_left(self):
        time_left = 0
        for index, stage in self.stage_index_map.items():
            time_left = min(0, time_left)
            time_left += self.stage_map[stage]["time_left"]
            self.stage_time_left[index] = time_left

    def get_layer_time_excess(self, layer: SwapStage, swap_time):
        return self.stage_map[layer]["time_left"] - swap_time

    def cause_delay(self, candidate: SwapPolicyCandidate):
        swap_out_time, swap_in_time = self.get_swap_time(candidate.tensor.info.size)
        swap_out_affected = self.get_layer_time_excess(candidate.swap_out_stage, swap_out_time)
        swap_in_stage_index = self.stage_map[candidate.swap_in_stage]["index"]
        swap_in_stage_index = swap_in_stage_index - 1
        swap_in_stage = self.stage_index_map[swap_in_stage_index]
        swap_in_affected = self.get_layer_time_excess(swap_in_stage, swap_in_time)
        return swap_out_affected < 0 or swap_in_affected < 0

    def run(self, candidates: List[SwapPolicyCandidate], _: List[SwapPolicyCandidate], delay=False):
        """
        delay: if False, then items in candidates would not cause delay in current simulation
        """
        for cand in candidates:
            swap_out_stage = cand.swap_out_stage
            swap_in_stage = cand.swap_in_stage
            swap_out_stage_index = self.stage_map[swap_out_stage]["index"]
            swap_in_stage_index = self.stage_map[swap_in_stage]["index"]
            swap_out_time, swap_in_time = self.get_swap_time(cand.tensor.info.size)
            swap_in_stage_index = swap_in_stage_index - 1
            swap_in_stage = self.stage_index_map[swap_in_stage_index]
            self.stage_map[swap_out_stage]["candidate_list"].append(
                (swap_out_stage, swap_out_time, "swap_out_stream", self.candidate_index, cand)
            )
            self.stage_map[swap_out_stage]["time_left"] -= swap_out_time
            if delay:
                find_flag = False
                for find_swap_in_index in range(swap_in_stage_index, swap_out_stage_index, -1):
                    time_left = self.calculate_time_left(find_swap_in_index)
                    if time_left > swap_in_time:
                        swap_in_stage = self.stage_index_map[find_swap_in_index]
                        find_flag = True
                        break
                if not find_flag:
                    swap_in_stage = self.stage_index_map[swap_in_stage_index]
            cand.swap_in_stage_actual = swap_in_stage
            self.stage_map[swap_in_stage]["candidate_list"].append(
                (swap_in_stage, swap_in_time, "swap_in_stream", self.candidate_index, cand)
            )
            self.stage_map[swap_in_stage]["time_left"] -= swap_in_time
