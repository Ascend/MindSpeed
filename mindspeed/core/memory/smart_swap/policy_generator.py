# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from typing import Dict, List
from collections import defaultdict

import numpy as np

from .swap_policy_config import swap_policy_config, SwapPolicyPref
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


class PolicyGeneratorV2(PolicyGenerator):
    def __select_tensor_used_in_optim(self):
        op_list = self.profiler_op_step.op_list
        all_optim_tensors = []
        result = {}
        for op in op_list:
            for tensor in op.tensor_list:
                if tensor.tensor_type == SwapTensorType.OPTIM:
                    all_optim_tensors.append(tensor)

        for tensor in all_optim_tensors:
            result[tensor] = []

        GB = 1024 * 1024 * 1024
        self.print_with_rank(
            f"selected optim size accumulated: {sum([tensor.size for tensor in result]) / GB:.2f} GB",
            print_level=PrintLevel.INFO
        )

        return result

    def __select_tensor_used_in_fwd_bwd(self):
        op_list = self.profiler_op_step.op_list
        tensor_stages = defaultdict(set)
        for op in op_list:
            # Parse the 'stage' string into a dictionary
            stage_info = op.stage
            stage_type = stage_info.stage_type
            # Parse each tensor in the tensor_list
            for tensor in op.tensor_list:
                tensor_stages[tensor].add(stage_type)

        def check_valid_tensors(tensor, stages):
            fwd_bwd_check = SwapStageType.FWD in stages and SwapStageType.BWD in stages
            return (
                fwd_bwd_check
                and tensor.tensor_type != SwapTensorType.SHARED_MEMORY
                and tensor.size > swap_policy_config.tensor_size_filter
            )

        valid_tensors = {tensor for tensor, stages in tensor_stages.items() if check_valid_tensors(tensor, stages)}

        # Step 1: Map each valid tensor to the list of ops that contain it in their tensor_list
        tensor_to_ops = defaultdict(list)
        for op in op_list:
            for tensor in op.tensor_list:
                if tensor in valid_tensors:
                    tensor_to_ops[tensor].append(op)

        # Step 2: Process each ptr to filter the ops according to the rules
        result = {}
        for tensor in valid_tensors:
            ops = tensor_to_ops.get(tensor, [])
            # Group the ops by their stage (using equality)
            stage_to_ops_pair_list = []
            for op in ops:
                current_stage = op.stage
                found = False
                # Check existing groups for a stage equal to current_stage
                for stage, group_ops_list in stage_to_ops_pair_list:
                    if stage == current_stage:
                        group_ops_list.append(op)
                        found = True
                        break
                if not found:
                    stage_to_ops_pair_list.append((current_stage, [op]))

            # For each group, select the appropriate op
            selected_ops = []
            for stage, group_ops_list in stage_to_ops_pair_list:
                if stage.stage_type == SwapStageType.FWD:
                    selected_op = max(group_ops_list, key=lambda x: x.op_id)
                    selected_ops.append(selected_op)
                elif stage.stage_type == SwapStageType.BWD:
                    selected_op = min(group_ops_list, key=lambda x: x.op_id)
                    selected_ops.append(selected_op)

            # Only keep tensors used by just two ops to keep policy simple (one from FWD and the other from BWD)
            if len(selected_ops) != 2:
                self.print_with_rank(
                    f"more than 2 ops use the same tensor {tensor}, skip...", print_level=PrintLevel.DEBUG
                )
                for op in selected_ops:
                    self.print_with_rank(f"    op: {op}", print_level=PrintLevel.DEBUG)
                continue

            result[tensor] = selected_ops

        return result

    def __filter_tensor_within_layers(self, tensor_info):
        tensor_info_by_fwd = defaultdict(list)
        GB = 1024 * 1024 * 1024
        tick_icon = "\u2705"
        cross_icon = "\u274c"

        # Pick all tensors in forward stage.
        for tensor, used_by_ops in tensor_info.items():
            first_op_stage = used_by_ops[0].stage
            second_op_stage = used_by_ops[1].stage
            stage_fwd = first_op_stage if first_op_stage.stage_type == SwapStageType.FWD else second_op_stage
            tensor_info_by_fwd[stage_fwd].append(tensor)

        for stage in tensor_info_by_fwd:
            tensor_info_by_fwd[stage].sort(key=lambda tensor: (min(op.op_id for op in tensor_info[tensor])))

        swap_bucket_size = swap_policy_config.swap_bucket_size / GB
        filtered_tensors = {}
        for stage, tensor_list in tensor_info_by_fwd.items():
            mb_index = stage.micro_batch_index
            total_size = sum([(tensor.size / GB) for tensor in tensor_list])
            if swap_bucket_size == 0:
                selected_size = 0
                selection_done = True
            else:
                selected_size = total_size
                selection_done = False
            temp_size = 0
            if mb_index == 1:
                self.print_with_rank(f"Stage: {str(stage)}", print_level=PrintLevel.INFO)
            for tensor in tensor_list:
                select_status = cross_icon

                if not selection_done:
                    temp_size += tensor.size / GB
                    if swap_bucket_size >= 0 and temp_size > swap_bucket_size:  # overlimit, do not pick and stop
                        selection_done = True
                        selected_size = temp_size - (tensor.size / GB)
                    else:
                        select_status = tick_icon
                        filtered_tensors[tensor] = tensor_info[tensor]

                if mb_index == 1:
                    self.print_with_rank(f"    {select_status} Tensor: {str(tensor)}", print_level=PrintLevel.INFO)
            if mb_index == 1:
                self.print_with_rank(
                    f"Total tensor size per stage: {total_size} GB, selected: {selected_size} GB",
                    print_level=PrintLevel.INFO,
                )

        return filtered_tensors

    def __select_swappable_tensors(self):
        tensor_info_optim = self.__select_tensor_used_in_optim()
        tensor_used_in_fwd_bwd = self.__select_tensor_used_in_fwd_bwd()
        tensor_filtered = self.__filter_tensor_within_layers(tensor_used_in_fwd_bwd)

        optim_tensors = []
        activ_tensors = []
        for key, value in tensor_info_optim.items():
            self.print_with_rank(
                f"tensor: {key}, used_by_ops: {[(item.op_name, item.op_id, str(item.stage)) for item in value]}",
                print_level=PrintLevel.DEBUG,
            )
            optim_tensors.append((key, value))

        for key, value in tensor_filtered.items():
            self.print_with_rank(
                f"tensor: {key}, used_by_ops: {[(item.op_name, item.op_id, str(item.stage)) for item in value]}",
                print_level=PrintLevel.DEBUG,
            )
            activ_tensors.append((key, value))

        return optim_tensors, activ_tensors

    def __get_neighbor_stages(self, from_stage, offset):
        result_stage = from_stage
        for _ in range(abs(offset)):
            if offset >= 0:
                result_stage = self.profiler_op_step.get_next_stage(result_stage)
                if result_stage is None:
                    result_stage = SwapStage(stage_type=SwapStageType.OPTIM, micro_batch_index=0, layer_index=0)
            else:
                result_stage = self.profiler_op_step.get_prev_stage(result_stage)
        return result_stage

    def __create_activation_candidate(self, tensor, used_by_ops_list):
        tensor_info_detail = TensorInfoDetail(tensor)

        candidate = None
        if tensor.tensor_type == SwapTensorType.OTHERS:
            fwd_op = used_by_ops_list[0]
            bwd_op = used_by_ops_list[1]
            d2h_stage = self.__get_neighbor_stages(fwd_op.stage, 0)
            h2d_stage = self.__get_neighbor_stages(bwd_op.stage, -1)
            d2h_free_stage = self.__get_neighbor_stages(d2h_stage, 2)
            h2d_free_stage = self.__get_neighbor_stages(h2d_stage, 2)
            candidate = SwapPolicyCandidate(
                tensor_info_detail,
                is_optimizer_or_weight=False,
                swap_out_op=fwd_op,
                swap_in_op=bwd_op,
                free_stage=d2h_free_stage,
                swap_in_free_stage=h2d_free_stage,
            )
            candidate.set_device_to_host_stage(d2h_stage)
            candidate.set_host_to_device_stage(h2d_stage)

        return candidate

    def __create_optim_candidates(self, optim_tensors):
        candidates = []
        if swap_policy_config.policy_pref not in [SwapPolicyPref.BETTER_MEMORY_SAVING]:
            return candidates

        optim_tensors_sorted = sorted(optim_tensors, key=lambda item: item[0].size, reverse=True)
        for _, tensor_info in enumerate(optim_tensors_sorted):
            tensor_info_detail = TensorInfoDetail(tensor_info[0])
            d2h_stage = SwapStage(stage_type=SwapStageType.FWD, micro_batch_index=1, layer_index=1)
            h2d_stage = SwapStage(stage_type=SwapStageType.OPTIM, micro_batch_index=0, layer_index=0)
            d2h_free_stage = d2h_stage
            h2d_free_stage = h2d_stage

            candidate = SwapPolicyCandidate(
                tensor_info_detail,
                is_optimizer_or_weight=True,
                swap_out_stage=d2h_stage,
                swap_in_stage=h2d_stage,
                free_stage=d2h_free_stage,
                swap_in_free_stage=h2d_free_stage,
            )
            candidates.append(candidate)

        return candidates

    def __lint(self, candidates):
        """
        Policies under the following situations will be removed:
        1. Swap out stage and swap in stage are adjacent.
        """
        result = []
        for can in candidates:
            in_stage = can.swap_in_stage_actual
            out_stage = can.swap_out_stage_actual
            if (
                self.profiler_op_step.get_next_stage(out_stage) == in_stage
                or self.profiler_op_step.get_prev_stage(out_stage) == in_stage
                or out_stage == in_stage
            ):
                self.print_with_rank(f">>>>>>>>> FOUND CLOSE STAGES!! remove from policy", print_level=PrintLevel.DEBUG)
            else:
                result.append(can)
        return result

    def select_candidate(self):
        candidates = []

        optim_tensors, activ_tensors = self.__select_swappable_tensors()

        # Create candidates for optimizer tensors
        optim_candidates = self.__create_optim_candidates(optim_tensors)
        if len(optim_candidates) != 0:
            candidates.extend(optim_candidates)

        # Create candidates for activation tensors
        for tensor_item in activ_tensors:
            tensor = tensor_item[0]
            used_by_ops_list = tensor_item[1]

            candidate = self.__create_activation_candidate(tensor, used_by_ops_list)
            if candidate is not None:
                candidates.append(candidate)

        self.swap_list = self.__lint(candidates)
        self.print_with_rank(f"num of selected candidates: {len(self.swap_list)}", print_level=PrintLevel.INFO)
