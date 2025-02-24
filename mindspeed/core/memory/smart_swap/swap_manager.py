# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import time
from enum import Enum
from collections.abc import Iterable

import torch

from .hooks import register_swap_hooks_to_modules
from .swap_policy_config import swap_policy_config
from .swap_utils import print_with_rank, PrintLevel
from .swap_cpp_adaptor import (
    SwapConfig,
    ProfilerDataOneStep,
    SwapStageType,
    SwapTensorType,
    record_tensor_ptr_with_types,
    get_smart_swap_cpp,
)
from .swap_engine import SwapEngine


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class SwapRunningStage(Enum):
    WARMUP_STAGE = 0  # Warmup阶段：opSequence不稳定
    SEARCHING_POLICY_STAGE = 1  # 迭代策略阶段：opSequence稳定，可能有OOM，策略不稳定
    STABLE_STAGE = 2  # 策略稳定阶段：opSequence稳定，策略稳定
    RESERVED = 3


@singleton
class SwapManager:
    def __init__(
        self,
        num_micro_batch_fcn,
        models,
        num_layers,
        optimizer=None,
        get_optimizer_tensors_fcn=None,
        get_shared_tensors_fcn=None,
        custom_policy_fcn=None,
    ):
        if torch.distributed.is_initialized():
            swap_policy_config.rank = torch.distributed.get_rank()

        option = {"OP_HOOK_ENABLE": "enable"}
        torch.npu.set_option(option)

        self.smart_swap_cpp = get_smart_swap_cpp()
        self.smart_swap_cpp.init_cpp_manager()
        self.smart_swap_cpp.NPUSwapManager.GetInstance().swap_enable = True
        self.smart_swap_cpp.NPUSwapManager.GetInstance().swap_oom_enable = True
        self.config = SwapConfig()
        self.num_micro_batch_fcn = num_micro_batch_fcn
        self.models = models
        self.get_shared_tensors_fcn = get_shared_tensors_fcn
        self.swap_hook_registers: list = []
        self.swap_engine = SwapEngine(models, optimizer, get_optimizer_tensors_fcn, self.config, custom_policy_fcn)
        self.start_time = time.time()
        self.cur_warmup_step = 0
        self.running_stage = SwapRunningStage.RESERVED
        self.is_new_op_sequence = True
        self.model_num_layers = num_layers
        self.global_initialize()

    def __del__(self):
        option = {"OP_HOOK_ENABLE": "disable"}
        torch.npu.set_option(option)
        self.smart_swap_cpp.deinit_cpp_manager()

    def __check_layer_param(self, model_num_layers):
        if not isinstance(model_num_layers, int):
            raise ValueError("model_num_layers must be an integer.")
        if model_num_layers != -1 and model_num_layers <= 0:
            raise ValueError("model_num_layers must be a positive integer or -1.")

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="SwapManager", print_level=print_level)

    def global_initialize(self):
        stage = self.config.stage
        stage.stage_type = SwapStageType.INIT
        self.config.stage = stage
        self.config.step = 0
        self.config.micro_batch_num = self.num_micro_batch_fcn()
        self.config.fwd_op_layer_info = []
        self.config.bwd_op_layer_info = []
        self.register_model_hooks(self.models)
        self.record_shared_memory(self.models)
        self.start_time = time.time()
        self.init_for_new_op_seq()
        self.config.enable_profiler = True
        self.config.enable_executor = False
        self.config.enable_custom_record_stream = swap_policy_config.enable_custom_record_stream
        self.__check_layer_param(self.model_num_layers)
        swap_policy_config.logical_layer_num = (
            -1 if self.model_num_layers < 0 else (10 // self.model_num_layers + 1) * self.model_num_layers
        )

    def record_model_tensor_type(self, models):
        tensors = []
        for model in models:
            # MODEL
            for name, param in model.named_parameters():
                tensors.append(param.data)

        record_tensor_ptr_with_types(tensors, SwapTensorType.MODEL, 0, False)

    def record_shared_memory(self, models):
        if models and self.get_shared_tensors_fcn:
            tensors = self.get_shared_tensors_fcn(models)
            record_tensor_ptr_with_types(tensors, SwapTensorType.SHARED_MEMORY, 0, True)

    def init_for_new_op_seq(self):
        self.print_with_rank("Call init_for_new_op_seq")
        self.running_stage = SwapRunningStage.WARMUP_STAGE
        self.swap_engine.clear_policy()
        self.is_new_op_sequence = True
        self.cur_warmup_step = 0

    def step(self):
        end_time = time.time()
        self.config.one_step_duration = end_time - self.start_time
        for swap_hook_register in self.swap_hook_registers:
            swap_hook_register.reset()
        self.config.micro_batch_num = self.num_micro_batch_fcn()
        profiler_data_one_step = ProfilerDataOneStep(
            self.config.one_step_duration, self.config.step, self.config.is_oom, self.config.enable_profiler
        )
        self.swap_engine.append_profiler_data(profiler_data_one_step)
        self.swap_engine.all_step_duration[self.swap_engine.step] = self.config.one_step_duration

        self.print_with_rank(
            (
                f"Step: {self.config.step}, Time elapsed: {self.config.one_step_duration}, "
                f"Logical layer num: {swap_policy_config.logical_layer_num}, "
                f"Op num: {len(profiler_data_one_step.op_list)}, "
                f"Current running stage: {self.running_stage.name}, OOM state: {self.config.is_oom}"
            ),
            print_level=PrintLevel.INFO,
        )
        self.print_with_rank(
            ("OOM swap: \n" + "\n".join(str(i) for i in profiler_data_one_step.swap_list if i.is_oom)),
            print_level=PrintLevel.INFO,
        )
        self.print_with_rank(f"{str(profiler_data_one_step)}")

        if self.running_stage == SwapRunningStage.WARMUP_STAGE:
            if self.swap_engine.is_similar_with_policy_profiler(profiler_data_one_step):
                self.cur_warmup_step += 1
            if self.cur_warmup_step == swap_policy_config.warmup_step:
                self.running_stage = SwapRunningStage.SEARCHING_POLICY_STAGE
        elif self.running_stage == SwapRunningStage.SEARCHING_POLICY_STAGE:
            self.cur_warmup_step += 1
            if not self.swap_engine.is_similar_with_policy_profiler(profiler_data_one_step):
                self.init_for_new_op_seq()
            elif self.cur_warmup_step == swap_policy_config.stable_step:
                self.running_stage = SwapRunningStage.STABLE_STAGE
        elif self.running_stage == SwapRunningStage.STABLE_STAGE:
            if self.swap_engine.forced_swap_list:
                self.init_for_new_op_seq()
        else:
            raise RuntimeError(f"Get incorrect running_stage: {self.running_stage.name}")

        self.print_with_rank(f"Change running stage to: {self.running_stage.name}", print_level=PrintLevel.INFO)
        if self.running_stage == SwapRunningStage.WARMUP_STAGE:
            self.config.enable_profiler = True
            self.config.enable_executor = False
        elif self.running_stage == SwapRunningStage.SEARCHING_POLICY_STAGE:
            self.config.enable_profiler = True
            self.config.enable_executor = True
            policy_result = self.swap_engine.run(profiler_data_one_step, self.is_new_op_sequence)
            policy_result.set_py_swap_policy_to_cpp(self.config)
            self.smart_swap_cpp.updateStep()
            self.is_new_op_sequence = False
            self.print_with_rank(f"Policy result:\n{policy_result}", print_level=PrintLevel.DEBUG)
        elif self.running_stage == SwapRunningStage.STABLE_STAGE:
            self.config.enable_profiler = False
            self.config.enable_executor = True
            self.smart_swap_cpp.updateStep()
        else:
            raise RuntimeError(f"Get incorrect running_stage: {self.running_stage.name}")

        self.print_with_rank(
            (
                f"All step duration: "
                f"{[(step, time) for step, time in self.swap_engine.all_step_duration.items()]}\n\n"
            ),
            print_level=PrintLevel.INFO,
        )

        self.config.step += 1
        self._update_config_for_step_hook(SwapStageType.INIT, 0, 0, 0)
        self.start_time = time.time()

    def _update_config_for_step_hook(
        self, stage_type: SwapStageType, layer_index, micro_batch_index, current_stage_op_id
    ):
        stage = self.config.stage
        stage.stage_type = stage_type
        stage.layer_index = layer_index
        stage.micro_batch_index = micro_batch_index

        self.config.stage = stage
        self.config.current_stage_op_id = current_stage_op_id

    def fwd_pre_hook_custom_func(self, _, fwd_idx):
        self._update_config_for_step_hook(SwapStageType.FWD, 1, fwd_idx, 0)

    def bwd_pre_hook_custom_func(self, _, bwd_idx):
        self._update_config_for_step_hook(SwapStageType.BWD, 1, bwd_idx, 0)

    def bwd_post_hook_custom_func(self, _, bwd_idx):
        if bwd_idx == self.num_micro_batch_fcn():
            self._update_config_for_step_hook(SwapStageType.OPTIM, 0, 0, 0)

    def register_model_hooks(self, models):
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            swap_hook_register = register_swap_hooks_to_modules(model)
            swap_hook_register.register_custom_func(
                self.fwd_pre_hook_custom_func, None, self.bwd_pre_hook_custom_func, self.bwd_post_hook_custom_func
            )
            self.swap_hook_registers.append(swap_hook_register)
        self.print_with_rank("Register model swap hooks completed.")
