# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from typing import Dict, List

import numpy as np

from .swap_policy_config import swap_policy_config
from .swap_utils import print_with_rank, PrintLevel, timer
from .swap_cpp_adaptor import (
    ProfilerDataOneStep,
    SwapPolicyCandidate,
    TensorInfoDetail,
    UniqueSwapPtr,
    MemoryReductionInfo,
    MemoryPeakInfo,
    SwapStage,
    SwapStageType,
    SwapTensorType,
)
from .swap_arranger import TensorArranger


class PolicyGenerator:
    def __init__(self, profiler_op_step: ProfilerDataOneStep):
        self.size_coverage_weight = swap_policy_config.size_coverage_weight

        self.profiler_op_step = profiler_op_step
        self.tensor_info_dict: Dict[UniqueSwapPtr, TensorInfoDetail] = {}
        self.policy_candidate_list: List[SwapPolicyCandidate] = []
        self.intersect_candidates: List[SwapPolicyCandidate] = []
        self.swap_list: List[SwapPolicyCandidate] = []
        self.peak_list: List[MemoryReductionInfo] = []

        self.candidate_selected: Dict[SwapPolicyCandidate, bool] = {}
        self.memory_reduction_list = profiler_op_step.memory_reduction_list
        # new data structure
        self.mri_opid2idx = self.profiler_op_step.mri_opid2idx
        self.memory_peaks = self.profiler_op_step.memory_peaks
        self.swap_arranger = TensorArranger(
            self.profiler_op_step,
            os.path.join(swap_policy_config.output_root_path, f"Simulation_{swap_policy_config.rank}.html"),
            swap_policy_config.duration_time,
        )

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="Policy", print_level=print_level)

    def reduction_target_satisfied(self):
        for memory_reduction in self.memory_reduction_list:
            if not memory_reduction.cleared():
                return False
        self.print_with_rank("Successfully reach reduction target ...", print_level=PrintLevel.INFO)
        return True

    def get_covered_reductions(self, candidate_list=None):
        if not self.memory_reduction_list:
            return
        flag = 0
        if candidate_list is None:
            flag = 1
            candidate_list = self.policy_candidate_list
            for memory_info in self.memory_reduction_list:
                memory_info.intersect_candidate_list.clear()
        for candidate in candidate_list:
            candidate.num_covered_reductions = 0
            swap_out_stage = self.profiler_op_step.layer_info.get_next_layer(candidate.swap_out_stage_actual)
            swap_in_stage = candidate.swap_in_stage_actual
            start_op_id = self.profiler_op_step.layer_info.layer_start_opid[swap_out_stage]
            end_op_id = self.profiler_op_step.layer_info.layer_start_opid[swap_in_stage]
            if start_op_id >= self.memory_reduction_list[-1].op_id or end_op_id <= self.memory_reduction_list[0].op_id:
                candidate.start_mri_opid = -1
                candidate.end_mri_opid = -1
                candidate.num_covered_reductions = 0
            else:
                # 二分法查找
                # find the mri with smallest opid that has opid >= start_op_id
                start_mri_opid = self.get_closest_mri(start_op_id, cmp="ge")
                # find the mri with largest opid that has opid < end_op_id
                end_mri_opid = self.get_closest_mri(end_op_id, cmp="lt")
                if end_mri_opid == end_op_id:
                    end_mri_opid = self.memory_reduction_list[self.mri_opid2idx[end_mri_opid] - 1].op_id
                if start_mri_opid < start_op_id:
                    self.print_with_rank(
                        f"start_op_id={start_op_id}, end_op_id={end_op_id}, \
                        start_mri_opid={start_mri_opid}, end_mri_opid={end_mri_opid}",
                        print_level=PrintLevel.INFO,
                    )
                    if start_mri_opid < start_op_id:
                        raise ValueError("candidate.start_mri_opid should be >= than start_op_id")
                if end_mri_opid > end_op_id:
                    self.print_with_rank(
                        f"start_op_id={start_op_id}, end_op_id={end_op_id}, \
                        start_mri_opid={start_mri_opid}, end_mri_opid={end_mri_opid}",
                        print_level=PrintLevel.INFO,
                    )
                    if end_mri_opid > end_op_id:
                        raise ValueError("candidate.end_mri_opid should be <= end_op_id")
                # candidate增加属性：start_mri_opid, end_mri_opid, num_covered_reductions
                if end_mri_opid < start_mri_opid:
                    candidate.start_mri_opid = -1
                    candidate.end_mri_opid = -1
                    candidate.num_covered_reductions = 0
                else:
                    candidate.start_mri_opid = start_mri_opid
                    candidate.end_mri_opid = end_mri_opid
                    # 计算candidate能cover的mri的个数，通过mri_opid2idx的map算start_mri_opid和end_mri_opid之间的mri的个数
                    candidate.num_covered_reductions = (
                        self.mri_opid2idx[end_mri_opid] - self.mri_opid2idx[start_mri_opid] + 1
                    )
            if flag:
                if candidate.start_mri_opid != -1 and candidate.end_mri_opid != -1:
                    for mri_idx in range(self.mri_opid2idx[start_mri_opid], self.mri_opid2idx[end_mri_opid] + 1):
                        self.memory_reduction_list[mri_idx].intersect_candidate_list.append(candidate)

    def get_closest_mri(self, target_opid, cmp="ge"):
        """
        Binary search for the opid closest to target_opid.
        cmp:
            'ge': result opid greater than or equal to target_opid;
            'lt': result opid less than target_opid;
        """
        p1 = 0
        p2 = len(self.memory_reduction_list) - 1
        if cmp not in ["ge", "lt"]:
            raise ValueError("For now only support cmp='ge' or cmp='lt' ")
        while p1 < p2 - 1:
            mid = (p1 + p2) // 2
            mid_opid = self.memory_reduction_list[mid].op_id
            if mid_opid == target_opid:
                return mid_opid
            elif mid_opid < target_opid:
                p1 = mid
            elif mid_opid > target_opid:
                p2 = mid
        if cmp == "ge":
            if self.memory_reduction_list[p1].op_id >= target_opid:
                return self.memory_reduction_list[p1].op_id
            else:
                return self.memory_reduction_list[p2].op_id
        elif cmp == "lt":
            if self.memory_reduction_list[p2].op_id < target_opid:
                return self.memory_reduction_list[p2].op_id
            else:
                return self.memory_reduction_list[p1].op_id

    def update_memory_reduction(self, candidate_list: List[SwapPolicyCandidate]):
        self.get_covered_reductions(candidate_list)
        for candidate in candidate_list:
            if candidate.start_mri_opid != -1 and candidate.end_mri_opid != -1:
                for mri_idx in range(
                    self.mri_opid2idx[candidate.start_mri_opid], self.mri_opid2idx[candidate.end_mri_opid] + 1
                ):
                    mri = self.memory_reduction_list[mri_idx]
                    mri.update_memory_reduction_need(-candidate.tensor.info.size)

    @timer
    def select_candidate(self):
        self.tensor_info_dict.clear()
        for op in self.profiler_op_step.op_list:
            for tensor in op.tensor_list:
                tensor_info = self.tensor_info_dict.setdefault(tensor.ptr, TensorInfoDetail(tensor))
                tensor_info.update_op(op)

        for detail_tensor in self.tensor_info_dict.values():
            detail_tensor.policy_candidate_list.clear()
            if (
                not detail_tensor.is_used_multiple_times()
                or detail_tensor.info.tensor_type == SwapTensorType.SHARED_MEMORY
                or detail_tensor.info.size < swap_policy_config.tensor_size_filter
            ):
                continue
            if detail_tensor.info.tensor_type == SwapTensorType.OPTIM:
                self.select_optim_tensor(detail_tensor)
            elif detail_tensor.info.tensor_type in (SwapTensorType.MODEL, SwapTensorType.OTHERS):
                self.select_model_tensor(detail_tensor)

        self.policy_candidate_list = list(
            set().union(*[i.policy_candidate_list for i in self.tensor_info_dict.values()])
        )
        self.candidate_selected = dict([(candidate, False) for candidate in self.policy_candidate_list])
        self.get_covered_reductions()

    def select_optim_tensor(self, detail_tensor: TensorInfoDetail):
        first_op = detail_tensor.used_op_list[0]
        if first_op.stage.stage_type != SwapStageType.OPTIM:
            return
        swap_out_stage = SwapStage(stage_type=SwapStageType.FWD, micro_batch_index=1, layer_index=1)
        swap_in_stage = SwapStage(stage_type=SwapStageType.OPTIM, micro_batch_index=0, layer_index=0)
        swap_policy_candidate = SwapPolicyCandidate(
            detail_tensor, is_optimizer_or_weight=True, swap_out_stage=swap_out_stage, swap_in_stage=swap_in_stage
        )
        detail_tensor.policy_candidate_list.append(swap_policy_candidate)
        return

    # 找到FWD最后一次使用和BWD第一次使用
    def select_model_tensor(self, detail_tensor: TensorInfoDetail):
        if any(op.stage.stage_type == SwapStageType.OPTIM for op in detail_tensor.used_op_list):
            return
        fwd_last_op = None
        bwd_first_op = None
        for op in detail_tensor.used_op_list:
            if op.stage.stage_type == SwapStageType.FWD and (fwd_last_op is None or fwd_last_op.op_id < op.op_id):
                fwd_last_op = op
            if op.stage.stage_type == SwapStageType.BWD and (bwd_first_op is None or bwd_first_op.op_id > op.op_id):
                bwd_first_op = op
        if fwd_last_op and bwd_first_op:
            swap_policy_candidate = SwapPolicyCandidate(
                detail_tensor, is_optimizer_or_weight=False, swap_out_op=fwd_last_op, swap_in_op=bwd_first_op
            )
            detail_tensor.policy_candidate_list.append(swap_policy_candidate)
        return

    def compute_score(self):
        if not self.policy_candidate_list:
            return
        tensor_info_sizes = [i.tensor.info.size for i in self.policy_candidate_list]
        max_size = max(tensor_info_sizes)
        min_size = min(tensor_info_sizes)
        max_size = max_size ** (1 / 3)
        min_size = min_size ** (1 / 3)
        size_range = max(0.001, max_size - min_size)

        coverages = [i.num_covered_reductions for i in self.policy_candidate_list]
        max_coverage = max(coverages)
        min_coverage = min(coverages)
        coverage_range = max(0.001, max_coverage - min_coverage)

        for candidate in self.policy_candidate_list:
            normalized_coverage = (candidate.num_covered_reductions - min_coverage) / coverage_range
            normalized_size = (candidate.tensor.info.size ** (1 / 3) - min_size) / size_range
            candidate.score = normalized_coverage + self.size_coverage_weight * normalized_size

    def get_peak_list(self):
        # Select the maximum mri value from the top mri of each MemoryPeakInfo (self.memory_peaks)
        # so each iteration only one peak is selected.
        self.peak_list.clear()

        def get_max_for_each_mp(mp: MemoryPeakInfo):
            """
            找到每个MemoryPeak区间内对应的MemoryReductionInfo当前的最大memory_reduction_need
            """
            if mp.mp_mri_start_opid == -1 or mp.mp_mri_end_opid == -1:
                return None
            start_idx = self.mri_opid2idx[mp.mp_mri_start_opid]
            end_idx = self.mri_opid2idx[mp.mp_mri_end_opid] + 1
            mri_list = self.memory_reduction_list[start_idx:end_idx]
            mrn = [mri.memory_reduction_need for mri in mri_list]
            max_idx = np.argmax(mrn)
            self.print_with_rank(
                f"current top mri in MemoryPeakInfo is {mri_list[max_idx]}", print_level=PrintLevel.INFO
            )
            return mri_list[max_idx]

        mp_max = [(i, get_max_for_each_mp(mp)) for i, mp in enumerate(self.memory_peaks)]
        for mp in mp_max:
            self.print_with_rank(f"top mri from each MemoryPeakInfo {mp[1]}", print_level=PrintLevel.INFO)
        mp_max_list = np.array([0 if not item[1] else item[1].memory_reduction_need for item in mp_max])
        self.print_with_rank(f"top mri from each MemoryPeakInfo {[mp_max_list]}", print_level=PrintLevel.INFO)
        selected_peak_idx = np.argmax(mp_max_list)
        self.peak_list = [mp_max[selected_peak_idx][1]]

    def get_intersect_candidates(self):
        self.get_peak_list()
        self.intersect_candidates.clear()
        self.print_with_rank(f"len of peak list is {len(self.peak_list)}", print_level=PrintLevel.INFO)
        peak = self.peak_list[0]
        if not peak:
            return
        self.intersect_candidates = [
            cand for cand in peak.intersect_candidate_list if not self.candidate_selected[cand]
        ]
        self.intersect_candidates.sort(key=lambda x: (-x.score, x.start_mri_opid))
        self.print_with_rank(
            f"len of self.intersect_candidates after {len(self.intersect_candidates)}", print_level=PrintLevel.INFO
        )

    def simulation_select(self):
        reduction_need = self.peak_list[0].memory_reduction_need
        selected_candidates = []
        for cand in self.intersect_candidates:
            if not self.swap_arranger.cause_delay(cand):
                selected_candidates.append(cand)
                reduction_need -= cand.tensor.info.size
                if reduction_need <= 0:
                    return selected_candidates, False
        if not selected_candidates:
            return [self.intersect_candidates[0]], True
        return selected_candidates, False

    def simulation(self, use_custom_policy=False):
        if use_custom_policy:
            selected_candidates = self.policy_candidate_list
            cause_delay = False
        else:
            selected_candidates, cause_delay = self.simulation_select()
        self.print_with_rank(f"selected_candidates have {len(selected_candidates)} cands", print_level=PrintLevel.DEBUG)
        self.swap_list.extend(selected_candidates)
        self.swap_arranger.run(selected_candidates, self.swap_list, delay=cause_delay)
        self.update_memory_reduction(selected_candidates)
        for cand in selected_candidates:
            self.candidate_selected[cand] = True

    def get_sorted_swap_list(self):
        """
        Sort swap_list by: primary key: swap_out time; secondary key: tensor size reverse
        """
        swap_list_out_opid = [
            (
                candidate,
                (
                    self.profiler_op_step.layer_info.layer_start_opid[candidate.swap_out_stage]
                    if candidate.is_optimizer_or_weight
                    else candidate.swap_out_op.op_id
                ),
            )
            for candidate in self.swap_list
        ]
        swap_list_out_opid = sorted(swap_list_out_opid, key=lambda item: (item[1], -item[0].tensor.info.size))
        swap_list = [candidate for (candidate, out_opid) in swap_list_out_opid]
        return swap_list
