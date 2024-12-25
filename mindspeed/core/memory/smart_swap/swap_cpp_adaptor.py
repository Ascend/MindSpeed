# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from collections import Counter
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass
from bisect import bisect_right, bisect_left

import torch

from .swap_policy_config import swap_policy_config
from .swap_utils import print_with_rank, PrintLevel
from .swap_adaptor import load_smart_swap_module


def get_smart_swap_cpp():
    return load_smart_swap_module()


class SwapTensorType(Enum):
    MODEL = get_smart_swap_cpp().SwapTensorType.MODEL
    OPTIM = get_smart_swap_cpp().SwapTensorType.OPTIM
    SHARED_MEMORY = get_smart_swap_cpp().SwapTensorType.SHARED_MEMORY
    OTHERS = get_smart_swap_cpp().SwapTensorType.OTHERS
    RESERVED = get_smart_swap_cpp().SwapTensorType.RESERVED


class SwapStageType(Enum):
    INIT = get_smart_swap_cpp().SwapStageType.INIT
    FWD = get_smart_swap_cpp().SwapStageType.FWD
    BWD = get_smart_swap_cpp().SwapStageType.BWD
    OPTIM = get_smart_swap_cpp().SwapStageType.OPTIM
    RESERVED = get_smart_swap_cpp().SwapStageType.RESERVED


def record_tensor_ptr_with_types(
    tensors: List[torch.Tensor], tensor_type: SwapTensorType, update_weak_ptr_map=0, is_update_blacklist=False
):
    # 调用下面的函数时，当前在c++侧会自动clear其维护的map
    return get_smart_swap_cpp().recordTensorPtrWithTypes(tensors, tensor_type.value, update_weak_ptr_map, is_update_blacklist)


class SwapStage:
    def __init__(self, cpp_instance=None, stage_type=None, micro_batch_index=None, layer_index=None):
        self.stage_type: SwapStageType = None
        self.micro_batch_index = None
        self.layer_index = None

        if cpp_instance:
            self.from_cpp(cpp_instance)
        if stage_type is not None:
            self.stage_type = stage_type
        if micro_batch_index is not None:
            self.micro_batch_index = micro_batch_index
        if layer_index is not None:
            self.layer_index = layer_index

    def __eq__(self, other):
        if not isinstance(other, SwapStage):
            return NotImplemented
        return (
            self.stage_type == other.stage_type
            and self.micro_batch_index == other.micro_batch_index
            and self.layer_index == other.layer_index
        )

    def __ne__(self, other):
        if not isinstance(other, SwapStage):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.stage_type, self.micro_batch_index, self.layer_index))

    def copy(self):
        # return a python SwapStage copy
        instance = SwapStage()
        instance.stage_type = self.stage_type
        instance.micro_batch_index = self.micro_batch_index
        instance.layer_index = self.layer_index
        return instance

    def from_cpp(self, instance):
        self.stage_type = SwapStageType(instance.stageType)
        self.micro_batch_index = instance.microBatchIndex
        self.layer_index = instance.layerIndex

    def to_cpp(self, instance):
        instance.stageType = self.stage_type.value
        instance.microBatchIndex = self.micro_batch_index
        instance.layerIndex = self.layer_index

    def __str__(self):
        ret = dict(stage_type=self.stage_type.name, mbi=self.micro_batch_index, li=self.layer_index)
        return str(ret)

    def calculate_layer_index(self, stage_op_idx, fwd_layer_info, bwd_layer_info):
        # stage_op_idx: op_idx starting from the current stage
        # op_layer_info: fwd_op_layer_info, or bwd_op_layer_info
        self.layer_index = 0
        if self.stage_type == SwapStageType.FWD:
            op_layer_info = fwd_layer_info
        elif self.stage_type == SwapStageType.BWD:
            op_layer_info = bwd_layer_info
        elif self.stage_type == SwapStageType.OPTIM or self.stage_type == SwapStageType.INIT:
            self.layer_index = 0
            return self.layer_index
        else:
            raise RuntimeError(f"calculate_layer_index error, stage_type={self.stage_type}")

        for i, op_layer_info_value in enumerate(op_layer_info):
            if stage_op_idx <= op_layer_info_value:
                self.layer_index = i + 1  # layerIndex 从1开始
                break
        if self.layer_index == 0:
            self.layer_index = len(op_layer_info) + 1
        return self.layer_index


class SwapConfig:
    def __init__(self):
        self.cpp_config = get_smart_swap_cpp().NPUSwapManager.GetInstance().config

    def dict(self):
        return dict(
            micro_batch_num=self.micro_batch_num,
            layer_num=self.layer_num,
            is_oom=self.is_oom,
            stage=str(self.stage),
            step=self.step,
            one_step_duration=self.one_step_duration,
            policy_step=self.policy_step,
            current_stage_op_id=self.current_stage_op_id,
            enable_profiler=self.enable_profiler,
            enable_executor=self.enable_executor,
            fwd_op_layer_info=self.fwd_op_layer_info,
            bwd_op_layer_info=self.bwd_op_layer_info,
            enable_custom_record_stream=self.enable_custom_record_stream,
        )

    @property
    def micro_batch_num(self):
        return self.cpp_config.microBatchNum

    @micro_batch_num.setter
    def micro_batch_num(self, value):
        self.cpp_config.microBatchNum = value

    @property
    def layer_num(self):
        return self.cpp_config.layerNum

    @layer_num.setter
    def layer_num(self, value):
        self.cpp_config.layerNum = value

    @property
    def is_oom(self):
        return self.cpp_config.isOOM

    @is_oom.setter
    def is_oom(self, value):
        self.cpp_config.isOOM = value

    @property
    def stage(self) -> SwapStage:
        stage = SwapStage()
        stage.from_cpp(self.cpp_config.stage)
        return stage

    @stage.setter
    def stage(self, value: SwapStage):
        value.to_cpp(self.cpp_config.stage)

    @property
    def step(self):
        return self.cpp_config.step

    @property
    def next_step(self):
        return self.step + 1

    @step.setter
    def step(self, value):
        self.cpp_config.step = value

    @property
    def one_step_duration(self):
        return self.cpp_config.oneStepDuration

    @one_step_duration.setter
    def one_step_duration(self, value):
        self.cpp_config.oneStepDuration = value

    @property
    def policy_step(self):
        return self.cpp_config.policyStep

    @policy_step.setter
    def policy_step(self, value):
        self.cpp_config.policyStep = value

    @property
    def current_stage_op_id(self):
        return self.cpp_config.currentStageOpId

    @current_stage_op_id.setter
    def current_stage_op_id(self, value):
        self.cpp_config.currentStageOpId = value

    @property
    def enable_profiler(self):
        return self.cpp_config.enableProfiler

    @enable_profiler.setter
    def enable_profiler(self, value):
        self.cpp_config.enableProfiler = value

    @property
    def enable_executor(self):
        return self.cpp_config.enableExecutor

    @enable_executor.setter
    def enable_executor(self, value):
        self.cpp_config.enableExecutor = value

    @property
    def enable_custom_record_stream(self):
        return self.cpp_config.enableCustomRecordStream

    @enable_custom_record_stream.setter
    def enable_custom_record_stream(self, value):
        self.cpp_config.enableCustomRecordStream = value

    @property
    def tensor_size_thresh(self):
        return self.cpp_config.tensorSizeThresh

    @tensor_size_thresh.setter
    def tensor_size_thresh(self, value):
        self.cpp_config.tensorSizeThresh = value

    @property
    def fwd_op_layer_info(self):
        return self.cpp_config.fwdOpLayerInfo

    @fwd_op_layer_info.setter
    def fwd_op_layer_info(self, value):
        self.cpp_config.fwdOpLayerInfo = value

    @property
    def bwd_op_layer_info(self):
        return self.cpp_config.bwdOpLayerInfo

    @bwd_op_layer_info.setter
    def bwd_op_layer_info(self, value):
        self.cpp_config.bwdOpLayerInfo = value


class UniqueSwapPtr:
    def __init__(self, cpp_instance=None, ptr_base=None, index=None):
        self.ptr_base = None
        self.index = None

        if cpp_instance:
            self.from_cpp(cpp_instance)
        if ptr_base:
            self.ptr_base = ptr_base
        if index:
            self.index = index

    def from_cpp(self, instance):
        self.ptr_base = instance.ptrBase
        self.index = instance.index

    def to_cpp(self, instance):
        instance.ptrBase = self.ptr_base
        instance.index = self.index

    def __str__(self):
        return f"{self.ptr_base}_{self.index}"

    def __eq__(self, other):
        if not isinstance(other, UniqueSwapPtr):
            return NotImplemented
        return self.ptr_base == other.ptr_base and self.index == other.index

    def __ne__(self, other):
        if not isinstance(other, UniqueSwapPtr):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.ptr_base, self.index))


class ProfilerTensorInfo:
    def __init__(self, tensor_dict):
        self.origi_ptr = UniqueSwapPtr(cpp_instance=tensor_dict["ptr"])
        self.ptr = UniqueSwapPtr(cpp_instance=tensor_dict["ptr"])
        self.size = tensor_dict["size"]
        self.shape = tensor_dict["shape"]
        self.dtype = tensor_dict["dtype"]
        self.tensor_type = SwapTensorType(tensor_dict["tensorType"])

    def get_dict(self):
        ret = dict(
            ptr=str(self.ptr), size=self.size, shape=self.shape, dtype=self.dtype, tensor_type=self.tensor_type.name
        )
        return ret

    def __str__(self):
        return str(self.get_dict())


class ProfilerOpInfo:
    def __init__(self, op_dict):
        self.op_name = op_dict["opName"]
        self.op_id = op_dict["opId"]
        self.stage = SwapStage(cpp_instance=op_dict["stage"])
        self.step = op_dict["step"]
        self.allocated_bytes = op_dict["allocated_bytes"]
        self.reserved_bytes = op_dict["reserved_bytes"]
        self.active_bytes = op_dict["active_bytes"]
        self.tensor_list = []

        tensor_list = op_dict["tensor"]
        for tensor in tensor_list:
            self.tensor_list.append(ProfilerTensorInfo(tensor))

    def print_dict(self):
        return dict(
            name=self.op_name,
            op_id=self.op_id,
            stage=str(self.stage),
            tensor_list=[str(tensor) for tensor in self.tensor_list],
        )

    def print_dict_brief(self):
        return dict(name=self.op_name, op_id=self.op_id, stage=str(self.stage))

    def __str__(self) -> str:
        return str(
            dict(
                name=self.op_name,
                op_id=self.op_id,
                stage=str(self.stage),
                tensor_list=[str(tensor) for tensor in self.tensor_list],
            )
        )

    def get_brief_dict(self):
        return str(dict(name=self.op_name, op_id=self.op_id, stage=str(self.stage)))

    def __eq__(self, other):
        if not isinstance(other, ProfilerOpInfo):
            return NotImplemented
        return self.op_name == other.op_name and self.op_id == other.op_id and self.stage == other.stage

    def __ne__(self, other):
        if not isinstance(other, ProfilerOpInfo):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.op_name, self.op_id, self.stage))

    def __lt__(self, other):
        if not isinstance(other, ProfilerOpInfo):
            return NotImplemented
        return self.op_id < other.op_id

    def __gt__(self, other):
        if not isinstance(other, ProfilerOpInfo):
            return NotImplemented
        return self.op_id > other.op_id


class ProfilerSwapInfo:
    def __init__(self, swap_dict):
        self.op_id = swap_dict["opId"]
        self.swap_name = swap_dict["swapName"]
        self.size = swap_dict["size"]
        self.is_oom = swap_dict["isOOM"]
        self.src_ptr = UniqueSwapPtr(swap_dict["srcPtr"])
        self.dst_ptr = UniqueSwapPtr(swap_dict["dstPtr"])

    def print_dict(self):
        return dict(
            op_id=self.op_id,
            swap_name=self.swap_name,
            size=str(self.size),
            is_oom=self.is_oom,
            src_ptr=str(self.src_ptr),
            dst_ptr=str(self.dst_ptr),
        )

    def __str__(self) -> str:
        return str(
            dict(
                op_id=self.op_id,
                swap_name=self.swap_name,
                size=str(self.size),
                is_oom=self.is_oom,
                src_ptr=str(self.src_ptr),
                dst_ptr=str(self.dst_ptr),
            )
        )


class MemoryReductionInfo:
    # 适用于1.去除OOM 2.通过策略下降xxxG峰值内存 两种情况
    def __init__(self, op, memory_reduction_total):
        self.op = op
        self.op_id = op.op_id
        self.memory_reduction_need = memory_reduction_total
        self.memory_reduction_total = memory_reduction_total
        self.intersect_candidate_list: List[SwapPolicyCandidate] = []

    def __str__(self):
        return (
            f"Reduction_need:{self.memory_reduction_need}, "
            f"Reduction_total:{self.memory_reduction_total}, "
            f"OP: {self.op.get_brief_dict()}"
        )

    def update_memory_reduction_need(self, amount):
        self.memory_reduction_need += amount

    def cleared(self):
        return self.memory_reduction_need <= 0

    def check_in_list(self, memory_reduction_list):
        # precondition: memory_reduction_list is sorted according to op_id
        if not memory_reduction_list or len(memory_reduction_list) == 0:
            return False
        if memory_reduction_list[0].op_id > self.op_id:
            return False
        if memory_reduction_list[-1].op_id < self.op_id:
            return False
        return True

    def print_dict(self):
        ret = dict(
            op_id=str(self.op_id),
            op_name=str(self.op.op_name),
            memory_reduction_need=str(self.memory_reduction_need),
            memory_reduction_total=str(self.memory_reduction_total),
        )
        return ret

    def print_dict_op(self):
        return self.op.print_dict()


@dataclass
class MemoryPeakInfo:
    """
    模型运行中根据内存曲线进行抽象得到的数据结构。
    以bwd-fwd的交替为标志,每个MemoryPeak为相邻两次bwd-fwd交替之间的op序列,
    代表内存曲线从一个local minima升至local maxima再降至local minima的区间
    例如:在非PP的1F1B场景下,每个microbatch(一次前向后一次反向)为一个MemoryPeak;
    在PP场景中,如果stage序列为fwd1->fwd2->bwd1->fwd3->bwd2->fwd4->bwd3->bwd4,
    则第一个MemoryPeak为fwd1至bwd1, 第二个MemoryPeak为fwd3->bwd2, 等等

    MemoryPeakInfo记录每个MemoryPeak的信息
    start_opid: 当前MemoryPeak开始的opid (这个MemoryPeak区间中第一个前向阶段的第一个op的opid)
    end_opid: 当前MemoryPeak结束的opid (这个MemoryPeak中最后一个反向阶段的最后一个op的opid)
    mp_mri_start_opid: 在这个MemoryPeak区间内第一处需要降内存(MemoryReductionInfo)的opid
    mp_mri_end_opid: 在这个MemoryPeak区间内最后一处需要降内存(MemoryReductionInfo)的opid
    """

    start_opid: int
    end_opid: int
    mp_mri_start_opid: int = -1
    mp_mri_end_opid: int = -1

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="MemoryPeakInfo", print_level=print_level)


class ProfilerLayerInfo:
    def __init__(self, op_list: List[ProfilerOpInfo]):
        self.op_list = op_list
        self.logical_layer_num = swap_policy_config.logical_layer_num

        self.stage_data = []
        self.fwd_op_layer_info = []
        self.bwd_op_layer_info = []
        self.layer_start_opid: Dict[SwapStage, int] = {}
        self.layer_to_index_map: Dict[SwapStage, int] = {}
        self.index_to_layer_map: Dict[int, SwapStage] = {}
        self.memory_peaks = []
        self.generate_layer_info()

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="ProfilerLayerInfo", print_level=print_level)

    def generate_layer_info(self):
        self.stage_data.clear()
        self.fwd_op_layer_info.clear()
        self.bwd_op_layer_info.clear()
        self.layer_start_opid.clear()
        self.layer_to_index_map.clear()
        self.index_to_layer_map.clear()
        self.memory_peaks.clear()
        self.logical_layer_num = swap_policy_config.logical_layer_num

        self.calculate_layer_info()
        self.set_layer_info()
        self.create_layer_mapping()
        self.get_memory_peaks()

    def calculate_layer_info(self):
        op_fwd_sequence = []
        op_bwd_sequence = []
        for op in self.op_list:
            if op.stage.micro_batch_index == 1:
                if op.stage.stage_type == SwapStageType.FWD:
                    op_fwd_sequence.append(op)
                elif op.stage.stage_type == SwapStageType.BWD:
                    op_bwd_sequence.append(op)
        if self.logical_layer_num < 0:  # use per op level layer info
            self.fwd_op_layer_info = list(range(len(op_fwd_sequence)))
            self.bwd_op_layer_info = list(range(len(op_bwd_sequence)))
        else:  # layer divided by logical layer num
            for i in range(self.logical_layer_num - 1):
                self.fwd_op_layer_info.append(len(op_fwd_sequence) // self.logical_layer_num * (i + 1))
                self.bwd_op_layer_info.append(len(op_bwd_sequence) // self.logical_layer_num * (i + 1))

    def set_layer_info(self):
        cur_stage = SwapStage()
        stage_start_idx = 0
        for op in self.op_list:
            stage = op.stage
            # 将layerindex的信息更新到stage中, 同时更新model_info.model_stage_seq
            if stage.stage_type != cur_stage.stage_type or stage.micro_batch_index != cur_stage.micro_batch_index:
                cur_stage.stage_type = stage.stage_type
                cur_stage.micro_batch_index = stage.micro_batch_index
                stage_start_idx = op.op_id
            stage_op_idx = op.op_id - stage_start_idx
            stage.calculate_layer_index(stage_op_idx, self.fwd_op_layer_info, self.bwd_op_layer_info)

    def create_layer_mapping(self):
        for index, op in enumerate(self.op_list):
            if not self.stage_data or op.stage != self.stage_data[-1]["stage"]:
                self.stage_data.append(
                    {
                        "op_id": op.op_id,
                        "stage": op.stage,
                        "stage_type": op.stage.stage_type,
                    }
                )
                self.print_with_rank(
                    (f"op_id:: {str(op.op_id)},  stage: {str(op.stage)}, stage_type: {str(op.stage.stage_type)}")
                )
        for index, row in enumerate(self.stage_data):
            if row["stage"] in self.layer_to_index_map:
                raise ValueError("Find duplicate stage ...")
            self.index_to_layer_map[index] = row["stage"]
            self.layer_to_index_map[row["stage"]] = index
            self.layer_start_opid[row["stage"]] = row["op_id"]

    def get_memory_peaks(self):
        """
        建立MemoryPeakInfo数据结构, 每次由反向阶段进入前向阶段时进入新的MemoryPeak区间
        仅将前反向进行MemoryPeakInfo的划分抽象, 不包含优化器和INIT阶段
        """
        self.memory_peaks = []
        cur_peak_start = -1
        cur_peak_end = -1
        for index, layer in self.index_to_layer_map.items():
            if cur_peak_start == -1:
                if index == 0 and layer.stage_type == SwapStageType.FWD:
                    cur_peak_start = self.layer_start_opid[layer]
                elif index > 0:
                    prev_layer = self.index_to_layer_map[index - 1]
                    if layer.stage_type == SwapStageType.FWD and prev_layer.stage_type != SwapStageType.FWD:
                        cur_peak_start = self.layer_start_opid[layer]
            if cur_peak_end == -1:
                if index == -len(self.layer_to_index_map) - 1 and layer.stage_type == SwapStageType.BWD:
                    cur_peak_end = self.layer_end_opid[layer]
                elif index < len(self.layer_to_index_map) - 1:
                    next_layer = self.index_to_layer_map[index + 1]
                    if layer.stage_type == SwapStageType.BWD and next_layer.stage_type != SwapStageType.BWD:
                        cur_peak_end = self.layer_start_opid[next_layer] - 1
                if cur_peak_start != -1 and cur_peak_end != -1:
                    cur_memory_peak = MemoryPeakInfo(cur_peak_start, cur_peak_end)
                    self.memory_peaks.append(cur_memory_peak)
                    cur_peak_start = -1
                    cur_peak_end = -1
        self.print_with_rank(
            f"current profiler step has {len(self.memory_peaks)} memory peaks", print_level=PrintLevel.INFO
        )

    def get_prev_layer(self, layer: SwapStage):
        idx = self.layer_to_index_map[layer]
        if idx - 1 not in self.index_to_layer_map:
            return None
        else:
            return self.index_to_layer_map[idx - 1]

    def get_next_layer(self, layer: SwapStage):
        idx = self.layer_to_index_map[layer]
        if idx + 1 not in self.index_to_layer_map:
            return None
        else:
            return self.index_to_layer_map[idx + 1]


class ProfilerDataOneStep:
    def __init__(self, duration_time, step, is_oom, enable_profiler=True):
        self.op_list: List[ProfilerOpInfo] = []
        self.swap_list: List[ProfilerSwapInfo] = []
        self.memory_reduction_list: List[MemoryReductionInfo] = []
        self.layer_start_opid: Dict[SwapStage, int] = dict()
        self.layer_info: ProfilerLayerInfo = None
        self.duration_time = duration_time
        self.step = step
        self.max_memory = None
        self.target_memory = None
        self.is_oom = is_oom

        if enable_profiler:
            self.acquire_data()
            self.layer_info = ProfilerLayerInfo(self.op_list)
            self.layer_start_opid = self.layer_info.layer_start_opid
            self.memory_peaks = self.layer_info.memory_peaks
            self.init_memory_reduction_list()
            self.get_memory_peak_mri()

        self.__stage_list: List[SwapStage] = []
        self.__stage_map: Dict[SwapStage, int] = {}
        self.__parse_stage_info()
        self.__op_info_cache: Dict[str, List[ProfilerOpInfo]] = {}  # {op_name, List[ProfilerOpInfo]}

    def __parse_stage_info(self):
        for op in self.op_list:
            if not self.__stage_list or op.stage != self.__stage_list[-1]:
                self.__stage_list.append(op.stage)
                self.__stage_map[op.stage] = self.__stage_list.index(op.stage)

    def __get_op_info_from_list(self, from_op: ProfilerOpInfo, op_name: str, direction: str) -> ProfilerOpInfo:
        # Determine the bisect function based on the direction
        if direction == "next":
            bisect_fn = bisect_right
            op_name_check = op_name
            idx_adjustment = 0
        elif direction == "prev":
            bisect_fn = bisect_left
            op_name_check = op_name
            idx_adjustment = -1
        else:
            raise ValueError("direction must be 'next' or 'prev'")

        if op_name_check == "":  # when search op is not specified
            begin_idx = bisect_fn(self.op_list, from_op) + idx_adjustment
            if begin_idx < 0 or begin_idx >= len(self.op_list):
                return None
            return self.op_list[begin_idx]

        # Cache logic: Fetch or cache the op_info_list
        if op_name_check not in self.__op_info_cache:
            op_info_list = [op for op in self.op_list if op.op_name == op_name_check]
            self.__op_info_cache[op_name_check] = op_info_list
        else:
            op_info_list = self.__op_info_cache[op_name_check]

        # Determine the index to start searching from
        begin_idx = bisect_fn(self.op_list, from_op) + idx_adjustment
        if begin_idx < 0 or begin_idx >= len(self.op_list):
            return None

        # Search within the cached op_info_list
        target_idx = bisect_fn(op_info_list, from_op) + idx_adjustment
        if target_idx < 0 or target_idx >= len(op_info_list):
            return None
        return op_info_list[target_idx]

    def group_op_info_by(self, op_info_list: List[ProfilerOpInfo], method="") -> List[List[ProfilerOpInfo]]:
        if not all(isinstance(item, ProfilerOpInfo) for item in op_info_list):
            raise TypeError("op_info_list can only contain elements with ProfilerOpInfo type.")
        if method == "microbatch":
            result_op_info_list = []
            mb_group = []
            curr_mb = None
            for op in op_info_list:
                if curr_mb is None:
                    curr_mb = op.stage.micro_batch_index
                    mb_group.append(op)
                else:
                    if op.stage.micro_batch_index != curr_mb:
                        curr_mb = op.stage.micro_batch_index
                        result_op_info_list.append(mb_group.copy())
                        mb_group.clear()
                    mb_group.append(op)
            return result_op_info_list
        elif method == "":
            return op_info_list
        else:
            raise NotImplementedError('group_by method other than "microbatch" is not implemented yet.')

    def get_all_op_info(self, op_names: List[str] = None) -> List[ProfilerOpInfo]:
        if op_names is None or len(op_names) == 0:
            return self.op_list
        op_info_list = []
        for op_name in op_names:
            if op_name in self.__op_info_cache:
                op_info_list.extend(self.__op_info_cache[op_name])
            else:
                op = self.get_first_op_info(op_name)
                while op is not None:
                    op_info_list.append(op)
                    op = self.get_next_op_info(op, op_name)
        return op_info_list

    def get_next_op_info(self, from_op: ProfilerOpInfo, next_op_name: str = "") -> ProfilerOpInfo:
        if from_op is None:
            return None
        return self.__get_op_info_from_list(from_op, next_op_name, "next")

    def get_prev_op_info(self, from_op: ProfilerOpInfo, prev_op_name: str = "") -> ProfilerOpInfo:
        if from_op is None:
            return None
        return self.__get_op_info_from_list(from_op, prev_op_name, "prev")

    def get_first_op_info(self, op_name: str = "") -> ProfilerOpInfo:
        if len(self.op_list) == 0:
            return None
        first_op = self.op_list[0]
        if op_name == "":
            return first_op
        return self.get_next_op_info(first_op, op_name)

    def get_last_op_info(self, op_name: str = "") -> ProfilerOpInfo:
        if len(self.op_list) == 0:
            return None
        last_op = self.op_list[-1]
        if op_name == "":
            return last_op
        return self.get_prev_op_info(last_op, op_name)

    def __get_adjacent_stage(self, stage: SwapStage, op_name: str, direction: str) -> SwapStage:
        # Determine whether we are looking for the next or previous stage
        if direction == "next":
            stage_index_adjustment = 1
            get_op_fn = self.get_first_op_info
            get_adj_op_fn = self.get_next_op_info
        elif direction == "prev":
            stage_index_adjustment = -1
            get_op_fn = self.get_last_op_info
            get_adj_op_fn = self.get_prev_op_info
        else:
            raise ValueError("direction must be 'next' or 'prev'")

        # Get the stage index from the stage map
        if stage is None:
            return None
        stage_index = self.__stage_map.get(stage, None)
        if stage_index is None:
            return None

        # If op_name is empty, handle the simple case of getting the next or previous stage
        if op_name == "":
            adjacent_stage_index = stage_index + stage_index_adjustment
            if adjacent_stage_index < 0 or adjacent_stage_index >= len(self.__stage_list):
                return None
            return self.__stage_list[adjacent_stage_index]

        # If op_name is specified, traverse the operations to find the adjacent stage
        result_stage = None
        curr_op = get_op_fn(op_name)
        while curr_op is not None:
            curr_stage_idx = self.__stage_map.get(curr_op.stage, None)
            if curr_stage_idx is None:
                break  # Avoid infinite loop if stage is not found

            is_valid_in_next_direction = direction == "next" and curr_stage_idx > stage_index
            is_valid_in_prev_direction = direction == "prev" and curr_stage_idx < stage_index

            if is_valid_in_next_direction or is_valid_in_prev_direction:
                result_stage = curr_op.stage
                break
            curr_op = get_adj_op_fn(curr_op, op_name)
        return result_stage

    def get_next_stage(self, stage: SwapStage, op_name: str = "") -> SwapStage:
        return self.__get_adjacent_stage(stage, op_name, "next")

    def get_prev_stage(self, stage: SwapStage, op_name: str = "") -> SwapStage:
        return self.__get_adjacent_stage(stage, op_name, "prev")

    def print_with_rank(self, message, print_level=PrintLevel.DEBUG):
        print_with_rank(message, prefix="ProfilerDataOneStep", print_level=print_level)

    def __str__(self):
        ret = "=" * 20 + "ProfilerDataOneStep SHOW BEGIN" + "=" * 20 + "\n"

        ret += f"The length of op_list is {len(self.op_list)}\n"
        for index, op_info in enumerate(self.op_list):
            ret += f"op_info-{index}, {str(op_info.print_dict())}\n"

        ret += f"The length of swap_list is {len(self.swap_list)}\n"
        for index, swap_info in enumerate(self.swap_list):
            ret += f"swap_info-{index}, {str(swap_info.print_dict())}\n"

        for index, memory_reduction in enumerate(self.memory_reduction_list):
            ret += f"memory_reduction-{index}, {str(memory_reduction.print_dict())}\n"

        ret += "=" * 20 + "ProfilerDataOneStep SHOW END" + "=" * 20 + "\n"
        return ret

    @property
    def length(self):
        return len(self.op_list)

    def acquire_data(self):
        op_list = get_smart_swap_cpp().getProfilerOpInfoData()
        swap_list = get_smart_swap_cpp().getProfilerSwapInfoData()
        self.op_list = [ProfilerOpInfo(i) for i in op_list]
        self.swap_list = [ProfilerSwapInfo(i) for i in swap_list]
        get_smart_swap_cpp().updateProfiler()

    def filter_swap_list(self):
        """
        修正内存建模曲线:将swap_list中有swap_out但是没有对应swap_in的tensor单独记录,
        以MemoryPeakInfo为单位记录当前MemoryPeakInfo中上述多余swap out的tensor的总size
        """
        swap_in_list = [item for item in self.swap_list if item.swap_name == "swapIn"]
        swap_in_total_size = sum([item.size for item in swap_in_list])
        self.print_with_rank(
            f"original swap in: {len(swap_in_list)} swap_in items with total size {swap_in_total_size}",
            print_level=PrintLevel.INFO,
        )
        swap_out_list = [item for item in self.swap_list if item.swap_name == "swapOut"]
        swap_out_total_size = sum([item.size for item in swap_out_list])
        self.print_with_rank(
            f"original swap out: {len(swap_out_list)} swap_out items with total size {swap_out_total_size}",
            print_level=PrintLevel.INFO,
        )
        if swap_in_total_size == swap_out_total_size:
            return None
        swap_in_dict = dict([item.src_ptr, item] for item in swap_in_list)
        extra_swap_out_dict = dict([(i, []) for i in range(len(self.memory_peaks))])
        swap_out_list.sort(key=lambda item: item.op_id)
        cur_mp_idx = 0
        for item in swap_out_list:
            if item.dst_ptr not in swap_in_dict:
                while cur_mp_idx < len(self.memory_peaks) and item.op_id > self.memory_peaks[cur_mp_idx].end_opid:
                    cur_mp_idx += 1
                if cur_mp_idx < len(self.memory_peaks):
                    extra_swap_out_dict[cur_mp_idx].append(item)
                else:
                    self.print_with_rank(
                        f"current swap out at op_id {item.op_id} happens at OPTIM stage", print_level=PrintLevel.INFO
                    )
        return extra_swap_out_dict

    def get_max_memory(self, extra_swap_out_dict=None):
        if extra_swap_out_dict:
            for i, item in extra_swap_out_dict.items():
                self.print_with_rank(f"extra_swap_out_dict has {len(item)} at {i}-th mp", print_level=PrintLevel.INFO)
        swap_list_dict = {}
        for swap_info in self.swap_list:
            swap_list_dict.setdefault(swap_info.op_id, []).append(swap_info)

        theoretical_minus_actual = 0
        cur_mp_idx = 0
        for op in self.op_list:
            swap_info_list = swap_list_dict.get(op.op_id, [])
            # 可能一个opid对应了多个swap
            swap_out_size = sum(info.size for info in swap_info_list if info.swap_name == "swapOut")
            swap_in_size = sum(info.size for info in swap_info_list if info.swap_name == "swapIn")

            # 以MemmoryPeakInfo为单位进行内存曲线校正：进入每个新的MemoryPeakInfo时都去除swap out但没swap in的tensor的总size
            if extra_swap_out_dict:
                if cur_mp_idx < len(self.memory_peaks) and op.op_id > self.memory_peaks[cur_mp_idx].end_opid:
                    extra_swap_out_size = sum([item.size for item in extra_swap_out_dict[cur_mp_idx]])
                    while cur_mp_idx < len(self.memory_peaks) and op.op_id > self.memory_peaks[cur_mp_idx].end_opid:
                        cur_mp_idx += 1
                    theoretical_minus_actual -= extra_swap_out_size

            theoretical_minus_actual = theoretical_minus_actual + swap_out_size - swap_in_size
            op.theoretical_active_bytes = op.active_bytes + theoretical_minus_actual
        return max(
            (
                op.theoretical_active_bytes
                for op in self.op_list
                if op.stage.stage_type not in [SwapStageType.INIT, SwapStageType.OPTIM]
            ),
            default=0,
        )

    def get_target_memory(self):
        self.print_with_rank(f"is current step oom? {self.is_oom}", print_level=PrintLevel.INFO)
        max_memory = max((op.active_bytes for op in self.op_list), default=0)
        if self.is_oom:
            return max_memory - swap_policy_config.redundant_memory
        elif self.swap_list:
            return max_memory

        if swap_policy_config.target_mode:
            target_memory = swap_policy_config.target_memory
        else:
            target_memory = self.max_memory - swap_policy_config.reduction_memory
        return target_memory

    def init_memory_reduction_list(self):
        self.memory_reduction_list = []
        extra_swap_out_dict = self.filter_swap_list()
        self.max_memory = self.get_max_memory(extra_swap_out_dict=extra_swap_out_dict)
        self.target_memory = self.get_target_memory()
        self.print_with_rank(
            f"max_memory={self.max_memory}, target_memory={self.target_memory}", print_level=PrintLevel.INFO
        )
        for op in self.op_list:
            if op.theoretical_active_bytes > self.target_memory:
                if op.stage.stage_type == SwapStageType.INIT:
                    self.print_with_rank("Skip init ... ")
                    continue
                if op.stage.stage_type == SwapStageType.OPTIM:
                    self.print_with_rank("Memory Bound at Optim Stage ...")
                    break
                memory_reduction_info = MemoryReductionInfo(op, op.theoretical_active_bytes - self.target_memory)
                self.memory_reduction_list.append(memory_reduction_info)
        # new data structure:build a map from index to opid of memory_reduction_info
        self.mri_opid2idx = dict(
            [(self.memory_reduction_list[i].op_id, i) for i in range(len(self.memory_reduction_list))]
        )

    def reset_memory_reduction_list(self):
        for memory_info in self.memory_reduction_list:
            memory_info.memory_reduction_need = memory_info.memory_reduction_total

    def get_memory_peak_mri(self):
        """
        建立每个MemoryPeakInfo对应的MemoryReductionInfo的开始和结束信息(mp_mri_start_opid, mp_mri_end_opid)
        """
        self.print_with_rank(
            f"current memory_reduction_list has len {len(self.memory_reduction_list)}", print_level=PrintLevel.INFO
        )
        if len(self.memory_reduction_list) == 0:
            return
        cur_mri = 0
        for idx, mp in enumerate(self.memory_peaks):
            mp.mp_mri_start_opid = -1
            mp.mp_mri_end_opid = -1
            while (
                cur_mri < len(self.memory_reduction_list)
                and self.memory_reduction_list[cur_mri].op_id >= mp.start_opid
                and self.memory_reduction_list[cur_mri].op_id <= mp.end_opid
            ):
                if mp.mp_mri_start_opid == -1:
                    mp.mp_mri_start_opid = self.memory_reduction_list[cur_mri].op_id
                cur_mri += 1
            if mp.mp_mri_start_opid > -1:
                mp.mp_mri_end_opid = self.memory_reduction_list[cur_mri - 1].op_id
            self.print_with_rank(
                f"current mp {idx} starts at opid {mp.mp_mri_start_opid} and ends at opid {mp.mp_mri_end_opid}",
                print_level=PrintLevel.INFO,
            )

    def get_sorted_op_names(self, sort_by="frequency") -> List[str]:
        op_name_sequence = [item.op_name for item in self.op_list]
        op_names_frequency_map = Counter(op_name_sequence)
        if sort_by == "frequency":
            op_names_frequency_list = sorted(
                op_names_frequency_map.keys(), key=lambda name: op_names_frequency_map[name], reverse=True
            )
        elif sort_by == "alphabetical":
            op_names_frequency_list = sorted(op_names_frequency_map.keys())
        else:
            raise NotImplementedError('sort methods other than "frequency" and "alphabetical" are not supported.')
        return op_names_frequency_list

    def map_unique_ptr_as_latest(self):
        map_old2new = {}
        for swap_row in self.swap_list:
            for key, value in map_old2new.items():
                if value == swap_row.src_ptr:
                    map_old2new[key] = swap_row.dst_ptr
            map_old2new[swap_row.src_ptr] = swap_row.dst_ptr
        for op in self.op_list:
            for tensor in op.tensor_list:
                if tensor.ptr in map_old2new:
                    tensor.ptr = map_old2new[tensor.ptr]

    def update_tensor_types(self, map_ptr2type: Dict[UniqueSwapPtr, SwapTensorType]):
        for op in self.op_list:
            for tensor in op.tensor_list:
                if tensor.ptr in map_ptr2type:
                    tensor.tensor_type = map_ptr2type[tensor.ptr]


class TensorInfoDetail:
    def __init__(self, profiler_tensor_info):
        self.info: ProfilerTensorInfo = profiler_tensor_info
        self.used_op_list: List[ProfilerOpInfo] = []
        self.policy_candidate_list: List[SwapPolicyCandidate] = []  # 一个Tensor可能被多次Swap

    def update_op(self, op: ProfilerOpInfo):
        if len(self.used_op_list) != 0 and self.used_op_list[-1].op_id == op.op_id:
            return
        self.used_op_list.append(op)

    def is_used_multiple_times(self):  # 如果Tensor只被使用了一次，不需要Swap
        return len(self.used_op_list) >= 2


class SwapPolicyCandidate:
    def __init__(
        self,
        tensor: TensorInfoDetail,
        is_optimizer_or_weight: bool = False,
        swap_out_op: ProfilerOpInfo = None,
        swap_in_op: ProfilerOpInfo = None,
        swap_out_stage: SwapStage = None,
        swap_in_stage: SwapStage = None,
        free_stage: SwapStage = None,
        swap_in_free_stage: SwapStage = None,
    ):
        self.tensor: TensorInfoDetail = tensor
        self.covered_reductions: List[MemoryReductionInfo] = []  # 可删除
        self.num_covered_reductions = 0
        self.start_mri_opid = -1  # 能覆盖的第一个mri的opid
        self.end_mri_opid = -1  # 能覆盖的最后一个mri的opid
        self.is_optimizer_or_weight = is_optimizer_or_weight
        if not is_optimizer_or_weight:
            self.swap_out_op = swap_out_op
            self.swap_in_op = swap_in_op
            self.swap_out_stage = swap_out_op.stage
            self.swap_in_stage = swap_in_op.stage
            self.swap_out_stage_actual = self.swap_out_stage
            self.swap_in_stage_actual = self.swap_in_stage
        else:
            self.swap_out_stage = swap_out_stage
            self.swap_in_stage = swap_in_stage
            self.swap_out_stage_actual = self.swap_out_stage
            self.swap_in_stage_actual = self.swap_in_stage
        self.free_stage = free_stage
        self.swap_in_free_stage = swap_in_free_stage

    def set_device_to_host_stage(self, stage: SwapStage):
        self.swap_out_stage = stage
        self.swap_out_stage_actual = stage

    def get_device_to_host_stage(self):
        return self.swap_out_stage_actual

    def set_device_to_host_free_stage(self, stage: SwapStage):
        self.free_stage = stage

    def set_host_to_device_stage(self, stage: SwapStage):
        self.swap_in_stage = stage
        self.swap_in_stage_actual = stage

    def get_host_to_device_stage(self):
        return self.swap_in_stage_actual

    def set_host_to_device_free_stage(self, stage: SwapStage):
        self.swap_in_free_stage = stage

    def to_cpp(self):
        instance = get_smart_swap_cpp().SwapPolicyInfo()
        instance.executorNeedMatch = not self.is_optimizer_or_weight
        if not self.is_optimizer_or_weight:
            self.tensor.info.origi_ptr.to_cpp(instance.ptr)
            instance.swapOutOpId = self.swap_out_op.op_id
            instance.swapInOpId = self.swap_in_op.op_id
        else:
            self.tensor.info.ptr.to_cpp(instance.ptr)
        self.swap_out_stage.to_cpp(instance.swapOutStage)
        self.swap_in_stage_actual.to_cpp(instance.swapInStage)
        self.free_stage.to_cpp(instance.freeStage)
        self.swap_in_free_stage.to_cpp(instance.swapInFreeStage)
        return instance

    def __str__(self):
        return str(
            dict(
                tensor=str(self.tensor.info),
                is_optimizer_or_weight=str(self.is_optimizer_or_weight),
                swap_out_op=self.swap_out_op.print_dict_brief() if hasattr(self, "swap_out_op") else "None",
                swap_in_op=self.swap_in_op.print_dict_brief() if hasattr(self, "swap_in_op") else "None",
                swap_out_stage=str(self.swap_out_stage),
                swap_in_stage=str(self.swap_in_stage),
                swap_out_stage_actual=str(
                    self.swap_out_stage_actual if hasattr(self, "swap_out_stage_actual") else "None"
                ),
                swap_in_stage_actual=str(
                    self.swap_in_stage_actual if hasattr(self, "swap_in_stage_actual") else "None"
                ),
                free_stage=str(self.free_stage),
                swap_in_free_stage=str(self.swap_in_free_stage),
            )
        )


class SwapPolicy:
    def __init__(self, swap_policy_candidates: List[SwapPolicyCandidate], profiler_data: ProfilerDataOneStep):
        self.__swap_policy_candidates: List[SwapPolicyCandidate] = swap_policy_candidates
        self.__profiler_data: ProfilerDataOneStep = profiler_data
        self.__stage_list: List[SwapStage] = []
        self.__stage_map: Dict[SwapStage, int] = {}
        self.__parse_stage_info()

    def __parse_stage_info(self):
        for op in self.__profiler_data.op_list:
            if not self.__stage_list or op.stage != self.__stage_list[-1]:
                self.__stage_list.append(op.stage)
                self.__stage_map[op.stage] = self.__stage_list.index(op.stage)

    def __auto_lint(self, policy: List[SwapPolicyCandidate]):
        # remove candidates with identical swap out and swap in stages.
        cand_remove_list = []
        for cand in policy:
            swap_out_stage = cand.swap_out_stage_actual
            swap_in_stage = cand.swap_in_stage_actual
            if swap_out_stage == swap_in_stage:
                cand_remove_list.append(cand)
                continue
        for cand in policy.copy():
            if cand in cand_remove_list:
                policy.remove(cand)

        # offset free stage by one if overlap.
        for cand in policy:
            swap_in_stage_actual = cand.swap_in_stage_actual
            swap_in_free_stage = cand.swap_in_free_stage
            if swap_in_stage_actual == swap_in_free_stage:
                cand.swap_in_free_stage = self.__profiler_data.get_next_stage(swap_in_free_stage)
            swap_out_stage_actual = cand.swap_out_stage_actual
            swap_out_free_stage = cand.free_stage
            if swap_out_stage_actual == swap_out_free_stage:
                cand.free_stage = self.__profiler_data.get_next_stage(swap_out_free_stage)

    def get_candidates(self) -> List[SwapPolicyCandidate]:
        return self.__swap_policy_candidates

    def set_candidates(self, candidates: List[SwapPolicyCandidate]):
        self.__auto_lint(candidates)
        self.__swap_policy_candidates = candidates

    def get_profiler_data(self) -> ProfilerDataOneStep:
        return self.__profiler_data


class PolicyResult:
    MAX_OP_NAMES_LENGTH = 64

    def __init__(self):
        self.policy_list: List[SwapPolicyCandidate] = None  # 用于SwapOut和SwapIn的Tensor信息列表
        self.policy_step = None  # 用第几个Step的Profiling结果进行匹配
        self.tensor_size_thresh = None  # 最小可能被Swap的Tensor的size大小
        self.fwd_op_layer_info = None  # 当前policy_step的Profiling对应的前向层信息
        self.bwd_op_layer_info = None  # 当前policy_step的Profiling对应的反向层信息
        self.op_names_frequency_list = None  # 当前policy_step的Profiling的OpName的频次列表，由高到低，最多有64个元素

    def clear(self):
        self.policy_list = None
        self.policy_step = None
        self.tensor_size_thresh = None
        self.fwd_op_layer_info = None
        self.bwd_op_layer_info = None
        self.op_names_frequency_list = None

    def __str__(self):
        info = dict(
            policy_step=self.policy_step,
            tensor_size_thresh=self.tensor_size_thresh,
            fwd_op_layer_info=self.fwd_op_layer_info,
            bwd_op_layer_info=self.bwd_op_layer_info,
        )
        ret = f"Basic policy is {info}\n"
        ret += f"A total number of {len(self.policy_list)} swaps are selected.\n"
        for index, item in enumerate(self.policy_list):
            ret += f"policy-{index}: \t\t{item}\n"
        return ret

    def set_py_swap_policy_to_cpp(self, config: SwapConfig):
        # 设置候选swap的tensor到c++侧
        swap_policy_info_list = []
        if self.policy_list is not None:
            for candidate in self.policy_list:
                try:
                    swap_policy_info_list.append(candidate.to_cpp())
                except Exception as e:
                    raise RuntimeError(f"candidate.to_cpp() error ! \n{candidate}") from e

        if self.fwd_op_layer_info is not None:
            config.fwd_op_layer_info = self.fwd_op_layer_info
        if self.bwd_op_layer_info is not None:
            config.bwd_op_layer_info = self.bwd_op_layer_info

        if self.policy_step:
            # 设置config相关
            config.tensorSizeThresh = self.tensor_size_thresh
            config.policy_step = self.policy_step
            # 设置op_names出现的频率
            get_smart_swap_cpp().setFrequentOpNameData(self.op_names_frequency_list[: self.MAX_OP_NAMES_LENGTH])

        else:
            config.tensorSizeThresh = swap_policy_config.tensor_size_thresh
            config.policy_step = 0
            get_smart_swap_cpp().setFrequentOpNameData([])

        get_smart_swap_cpp().setPolicyInfoData(swap_policy_info_list)
