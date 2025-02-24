# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import stat
import time
import pickle
from typing import Dict

import pandas

from .policy_generator import PolicyGenerator
from .swap_policy_config import swap_policy_config
from .swap_utils import print_with_rank, PrintLevel, timer
from .swap_cpp_adaptor import (
    SwapConfig,
    ProfilerDataOneStep,
    PolicyResult,
    SwapTensorType,
    SwapPolicyCandidate,
    UniqueSwapPtr,
    TensorInfoDetail,
    record_tensor_ptr_with_types,
    SwapPolicy,
)


class SwapEngine:
    def __init__(self, models, optimizer, get_optimizer_tensors_fcn, config: SwapConfig, custom_policy_fcn):
        # 相关模块
        self.models = models
        self.optimizer = optimizer
        self.get_optimizer_tensors_fcn = get_optimizer_tensors_fcn
        self.custom_policy_fcn = custom_policy_fcn

        # 控制参数
        self.config = config
        self.rank = swap_policy_config.rank
        self.output_root_path = swap_policy_config.output_root_path
        if swap_policy_config.save_policy or swap_policy_config.save_profiler_data:
            if not os.path.exists(self.output_root_path) and self.rank == 0:
                os.makedirs(self.output_root_path)
        self.duration_time = None
        self.step_parameters = {}
        self.all_step_duration = {}

        # profiling 数据
        self.profiler_op_step: ProfilerDataOneStep = None
        self.profiler_all_step: Dict[int, ProfilerDataOneStep] = dict()  # 目前为止所有step的profiler数据

        # 处理后的数据，用于生成策略
        self.tensor_info_dict: Dict[UniqueSwapPtr, TensorInfoDetail] = dict()

        # 当前生成的最新policy
        self.newest_policy_result: PolicyResult = PolicyResult()
        self.map_unique_ptr2tensor_type = dict()

        # 用户policy策略函数
        if self.custom_policy_fcn is None:
            print_with_rank("User policy is missing, skip user policy.", print_level=PrintLevel.INFO)
            self.use_custom_policy = False
        else:
            print_with_rank("Found user policy.", print_level=PrintLevel.INFO)
            self.use_custom_policy = True

    @property
    def step(self):
        return self.config.step

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="SwapEngine", print_level=print_level)

    def clear_policy(self):
        self.newest_policy_result.clear()

    def append_profiler_data(self, profiler_op_step: ProfilerDataOneStep):
        self.profiler_all_step[profiler_op_step.step] = profiler_op_step
        self.forced_swap_list = [i for i in profiler_op_step.swap_list if i.is_oom]
        if swap_policy_config.save_profiler_data:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            profiler_all_step_file = os.path.join(self.output_root_path, f"profiler_all_step_{self.rank}.pkl")
            with os.fdopen(os.open(profiler_all_step_file, flags, mode=mode), "wb") as file:
                pickle.dump(self.profiler_all_step, file)

    def save_policy_list(self, swap_list):
        swap_list_pd = pandas.DataFrame([i.tensor.info.get_dict() for i in swap_list])
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = stat.S_IWUSR | stat.S_IRUSR
        policy_file = os.path.join(self.output_root_path, f"Policy_{self.rank}.csv")
        with os.fdopen(os.open(policy_file, flags, mode=mode), "wb") as file:
            swap_list_pd.to_csv(file)

    def record_tensor_types(self):
        self.map_unique_ptr2tensor_type.clear()
        # 针对优化器状态的特殊类tensor，将其记录在C++侧的map映射中，方便其执行匹配
        if self.optimizer and self.get_optimizer_tensors_fcn:
            tensors = self.get_optimizer_tensors_fcn(self.optimizer)
            unique_ptrs = record_tensor_ptr_with_types(tensors, SwapTensorType.OPTIM, 1, False)
            for unique_ptr in unique_ptrs:
                self.map_unique_ptr2tensor_type[UniqueSwapPtr(unique_ptr)] = SwapTensorType.OPTIM

    def is_similar_with_policy_profiler(self, profiler_op_step: ProfilerDataOneStep):
        if self.newest_policy_result.policy_step is None:
            ret = True
            self.print_with_rank("The policy step is None, maybe initial stage ...")
        else:
            ret = self.is_equal_op_sequence(
                swap_policy_config, profiler_op_step, self.profiler_all_step[self.newest_policy_result.policy_step]
            )
            self.print_with_rank(
                (
                    f"now: {len(profiler_op_step.op_list)}, "
                    f"last: {self.profiler_all_step[self.newest_policy_result.policy_step].length}, "
                    f"ret: {ret}"
                )
            )
        if ret:
            self.print_with_rank("The sequence is similar with the policy one ...")
        return ret

    @timer
    def process_profiler_data(self):
        self.print_with_rank("Processing data ... ", print_level=PrintLevel.INFO)
        # 获取特殊类tensor的unique_ptr信息
        self.record_tensor_types()
        # 将profiler_op_step中的UniquePtr全部映射为最新的ptr
        self.profiler_op_step.map_unique_ptr_as_latest()
        # 刷新tensor type
        self.profiler_op_step.update_tensor_types(self.map_unique_ptr2tensor_type)
        self.print_with_rank(str(self.profiler_op_step))

        self.newest_policy_result.policy_step = self.step
        self.newest_policy_result.op_names_frequency_list = self.profiler_op_step.get_sorted_op_names()

    def run(self, profiler_op_step: ProfilerDataOneStep, is_new_op_sequence) -> PolicyResult:
        self.current_profiler_step = profiler_op_step
        self.profiler_op_step = (
            profiler_op_step if is_new_op_sequence else self.profiler_all_step[self.newest_policy_result.policy_step]
        )

        # 汇总参数 上一步的参数，运行时间，policy结果
        # 自适应迭代 fun，分优先级
        # 更新参数
        if is_new_op_sequence:
            self.process_profiler_data()

        policy_candidates, tensor_size_thresh = self.make_policy()
        self.newest_policy_result.tensor_size_thresh = tensor_size_thresh
        self.newest_policy_result.policy_list = policy_candidates
        self.newest_policy_result.fwd_op_layer_info = self.profiler_op_step.layer_info.fwd_op_layer_info
        self.newest_policy_result.bwd_op_layer_info = self.profiler_op_step.layer_info.bwd_op_layer_info

        return self.newest_policy_result

    @staticmethod
    def is_equal_op_sequence(
        policy_config, cur_sequence: ProfilerDataOneStep, target_sequence: ProfilerDataOneStep = None
    ) -> bool:
        """
        Compare how different cur_sequence is from target_sequence, and return a ratio.
        暂时先只比较长度
        """
        if target_sequence is None:
            return False
        target_len = cur_sequence.length
        cur_len = target_sequence.length
        return abs(target_len - cur_len) / cur_len < policy_config.op_diff_thresh

    def record_parameters(self):
        self.step_parameters[self.step] = {
            "duration_time": swap_policy_config.duration_time,
            "size_coverage_weight": swap_policy_config.size_coverage_weight,
            "redundant_memory": swap_policy_config.redundant_memory,
        }

    def set_parameters(self):
        swap_step = list(self.step_parameters.keys())
        min_duration = min(self.all_step_duration[i] for i in swap_step)
        best_step = [key for key, value in self.all_step_duration.items() if value == min_duration][0]
        swap_policy_config.duration_time = self.step_parameters[best_step]["duration_time"]
        swap_policy_config.size_coverage_weight = self.step_parameters[best_step]["size_coverage_weight"]
        swap_policy_config.redundant_memory = self.step_parameters[best_step]["redundant_memory"]

    def adjust_parameters(self):
        setattr(
            swap_policy_config,
            "duration_time",
            min(
                getattr(swap_policy_config, "duration_time", float("inf")),
                self.current_profiler_step.duration_time * swap_policy_config.adjust_step_duration,
            ),
        )

        if self.forced_swap_list:
            swap_policy_config.redundant_memory += swap_policy_config.adjust_memory
            self.profiler_op_step.init_memory_reduction_list()
            self.record_parameters()
            return

        swap_policy_config.size_coverage_weight += swap_policy_config.adjust_size_coverage_weight
        self.record_parameters()

    def check_policy_valid(self, candidate: SwapPolicyCandidate):
        # swap out free stage: (swap out op, swap in stage actual)
        # swap in stage actual: (swap out free stage, swap in op)
        # swap in free stage: (swap in op, )
        if not candidate.is_optimizer_or_weight:
            free_stage_opid = self.profiler_op_step.layer_start_opid[candidate.free_stage]
            swap_in_stage_actual_opid = self.profiler_op_step.layer_start_opid[candidate.swap_in_stage_actual]
            swap_in_free_stage_opid = self.profiler_op_step.layer_start_opid[candidate.swap_in_free_stage]
            swap_out_opid = (
                candidate.swap_out_op.op_id
                if not candidate.is_optimizer_or_weight
                else self.profiler_op_step.layer_start_opid[candidate.swap_out_stage]
            )
            swap_in_opid = (
                candidate.swap_in_op.op_id
                if not candidate.is_optimizer_or_weight
                else self.profiler_op_step.layer_start_opid[candidate.swap_in_stage]
            )
            if not (free_stage_opid > swap_out_opid and free_stage_opid < swap_in_stage_actual_opid):
                print(
                    f"Error! swap_out_free_stage_opid [{free_stage_opid}] should be > swap_out_opid [{swap_out_opid}] and < swap_in_stage_actual_opid [{swap_in_stage_actual_opid}]"
                )
                return False
            if not (swap_in_stage_actual_opid < swap_in_opid):
                print(
                    f"Error! swap_in_stage_actual_opid [{swap_in_stage_actual_opid}] should be < swap_in_opid [{swap_in_opid}]"
                )
                return False
            if not (swap_in_free_stage_opid > swap_in_stage_actual_opid):
                print(
                    f"Error! swap_in_free_stage_opid [{swap_in_free_stage_opid}] should be > swap_in_stage_actual_opid [{swap_in_stage_actual_opid}]"
                )
                return False
        return True

    @timer
    def make_policy(self):
        self.print_with_rank("Making policy ...", print_level=PrintLevel.INFO)
        self.adjust_parameters()
        self.profiler_op_step.reset_memory_reduction_list()
        policy_generator = PolicyGenerator(self.profiler_op_step)
        policy_generator.select_candidate()

        start_time = time.time()
        if self.use_custom_policy:
            policy_generator.simulation(use_custom_policy=True)
        else:
            policy_generator.compute_score()
            while not policy_generator.reduction_target_satisfied():
                # 寻找能降内存的policy
                policy_generator.get_intersect_candidates()
                # 选不出来就退出
                if not policy_generator.intersect_candidates:
                    self.print_with_rank(f"Fail to reach reduction target ...", print_level=PrintLevel.INFO)
                    break
                policy_generator.simulation()
        end_time = time.time()
        self.print_with_rank(f"policy generate takes {end_time - start_time} seconds.", print_level=PrintLevel.INFO)

        policy_generator.swap_arranger.save_stage_time_left()
        policy_generator.swap_arranger.set_free_stage()

        if self.use_custom_policy:
            # create SwapPolicy by providing existing swap list and profiler info
            curr_swap_policy = SwapPolicy(policy_generator.swap_list, self.profiler_op_step)
            self.custom_policy_fcn(curr_swap_policy)
            policy_generator.swap_list = curr_swap_policy.get_candidates()
        swap_list = policy_generator.get_sorted_swap_list()
        tensor_size_thresh = (
            min([candidate.tensor.info.size for candidate in swap_list])
            if swap_list
            else swap_policy_config.tensor_size_thresh
        )

        self.print_with_rank(
            (
                f"\n\tCurrent Step: {self.current_profiler_step.step}, "
                f"Policy Step: {self.profiler_op_step.step}, "
                f"Max Memory: {self.profiler_op_step.max_memory}, "
                f"Target Memory: {self.profiler_op_step.target_memory}, "
                f"Duration Time: {swap_policy_config.duration_time}, "
                f"Size Cov Weight: {swap_policy_config.size_coverage_weight}, "
                f"\n\tCandidate Num: {len(policy_generator.policy_candidate_list)}, "
                f"Policy Num: {len(swap_list)}, "
                f"Optim Num: {len([i for i in swap_list if i.tensor.info.tensor_type == SwapTensorType.OPTIM])}, "
                f"Model Num: {len([i for i in swap_list if i.tensor.info.tensor_type != SwapTensorType.OPTIM])}, "
                f"Min Tensor Size: {tensor_size_thresh}"
            ),
            print_level=PrintLevel.INFO,
        )
        if swap_policy_config.save_policy:
            self.save_policy_list(swap_list)
        return swap_list, tensor_size_thresh
