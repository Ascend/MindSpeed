from copy import deepcopy
from typing import List
from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import NumberConstant


class ProfilingConfig:
    """
        Basic parameters of profiling
    """

    def __init__(self, search_cfg=None, args=None):
        self.search_cfg = deepcopy(search_cfg)
        self.per_micro_layer = search_cfg.num_layers // search_cfg.pp
        self.vpp = search_cfg.vpp if search_cfg.vpp else 1
        self.micro_num = search_cfg.gbs // (search_cfg.mbs * search_cfg.dp) * self.vpp
        self.stage_id = 0

        # hardware config
        if args:
            if isinstance(args, Hardware):
                self.nodes = args.num_nodes
                self.devices_per_node = args.devices_per_node
                self.node_rank = args.node_rank
            else:
                self.nodes = args.nnodes
                self.devices_per_node = args.nproc_per_node
                self.node_rank = args.node_rank
        else:
            self.nodes = 1
            self.devices_per_node = 8
            self.node_rank = 0

    def search_first_operator_idx_for_per_layer_enable_pp_last_stage(self, fw_norm_index, bw_norm_index):
        fw_layer_start = []
        bw_layer_end = []
        recompute_fw = []
        warm_micro_num = self._calculate_warm_micro_num()
        bw_idx = 0
        fw_idx = 0
        for micro in range(self.micro_num):
            i = micro // (self.vpp * self.search_cfg.pp)
            fw_layer_start.append([fw_norm_index[fw_idx]])
            fw_idx = self._calculate_fw_idx(fw_idx, i, micro)
            bw_idx = self._calculate_bw_idx(bw_idx, i, micro)
            bw_layer_end.append([bw_norm_index[bw_idx - 1]])
            if self.search_cfg.is_full_recompute:
                if warm_micro_num <= micro + 1:
                    recompute_fw.append([fw_norm_index[fw_idx]])
                    fw_idx += NumberConstant.FW_NORM_OP_NUM_ENABLE_PP_OTHER_STAGE
                if micro == self.micro_num - 1:
                    for i in range(warm_micro_num - 1):
                        fw_idx += i * NumberConstant.FW_NORM_OP_NUM_ENABLE_PP_OTHER_STAGE
                        recompute_fw.append([fw_norm_index[fw_idx]])
        if self.vpp > 1:
            fw_per_micro_opt_num = fw_layer_start[1][0] - fw_layer_start[0][0]
        else:
            fw_per_micro_opt_num = fw_norm_index[2] - fw_norm_index[0]
        bw_per_micro_opt_num = bw_norm_index[2] - bw_norm_index[0]
        return fw_layer_start, bw_layer_end, recompute_fw, fw_per_micro_opt_num, bw_per_micro_opt_num

    def search_first_operator_idx_for_per_layer_enable_pp_other_stage(self, fw_norm_index, bw_norm_index):
        fw_layer_start = []
        bw_layer_end = []
        recompute_fw = []
        fw_norm_index = [fw_norm_index[i * 2: (i + 1) * 2] for i in range(len(fw_norm_index) // 2)]
        bw_norm_index = [bw_norm_index[i * 2: (i + 1) * 2] for i in range(len(bw_norm_index) // 2)]
        warm_micro_num = self._calculate_warm_micro_num()

        for micro in range(self.micro_num):
            if micro < warm_micro_num:
                fw_layer_start.append([fw_norm_index[micro][0]])
            else:
                fw_layer_start.append([fw_norm_index[micro + micro - warm_micro_num + 1][0]])
                recompute_fw.append([fw_norm_index[micro + micro - warm_micro_num][0]])
                if micro == self.micro_num - 1:
                    recompute_fw.extend(
                        [[index[0]] for index in fw_norm_index[len(fw_norm_index) - warm_micro_num:]])
            bw_layer_end.append([bw_norm_index[micro][-1]])
        if self.search_cfg.is_full_recompute:
            if len(recompute_fw) != self.micro_num:
                for i in range(len(recompute_fw), self.micro_num):
                    recompute_fw.append([fw_norm_index[i + self.micro_num][0]])
            bw_per_micro_opt_num = bw_norm_index[0][-1] - recompute_fw[0][0]
        else:
            bw_per_micro_opt_num = bw_norm_index[1][0] - bw_norm_index[0][0]
        fw_per_micro_opt_num = fw_layer_start[1][0] - fw_layer_start[0][0]
        return fw_layer_start, bw_layer_end, recompute_fw, fw_per_micro_opt_num, bw_per_micro_opt_num

    def search_first_operator_idx_for_per_layer_enable_pp(self, fw_norm_index, bw_norm_index):
        if self.stage_id == self.search_cfg.pp - 1:
            return self.search_first_operator_idx_for_per_layer_enable_pp_last_stage(fw_norm_index, bw_norm_index)
        else:
            return self.search_first_operator_idx_for_per_layer_enable_pp_other_stage(fw_norm_index, bw_norm_index)

    def search_first_operator_idx_for_per_layer_disable_pp(self, fw_norm_index, bw_norm_index):
        fw_layer_start = []
        bw_layer_end = []
        recompute_fw = []
        if self.search_cfg.is_full_recompute:
            fw_micro_rms_num = len(fw_norm_index) // self.micro_num

            fw_norm_index = [fw_norm_index[fw_micro_rms_num * i:fw_micro_rms_num * (i + 1)]
                             for i in range(self.micro_num)]
            bw_micro_rms_num = len(bw_norm_index) // self.micro_num

            bw_norm_index = [bw_norm_index[bw_micro_rms_num * i:bw_micro_rms_num * (i + 1)]
                             for i in range(self.micro_num)]
            fw_per_micro_opt_num = fw_norm_index[0][2] - fw_norm_index[0][0]
            bw_per_micro_opt_num = bw_norm_index[0][2] - bw_norm_index[0][0]

            for micro in range(self.micro_num):
                fw_layer_start.append([fw_norm_index[micro][0]])
                bw_layer_end.append([bw_norm_index[micro][-1]])
                recompute_fw.append([fw_norm_index[micro][3]])
        else:
            fw_per_micro_opt_num = fw_norm_index[2] - fw_norm_index[0]
            bw_per_micro_opt_num = bw_norm_index[2] - bw_norm_index[0]

            for micro in range(self.micro_num):
                fw_layer_start.append([fw_norm_index[3 * micro]])
                bw_layer_end.append([bw_norm_index[3 * (micro + 1) - 1]])
        return fw_layer_start, bw_layer_end, recompute_fw, fw_per_micro_opt_num, bw_per_micro_opt_num

    def _calculate_warm_micro_num(self):
        if self.vpp != 1:
            return self.search_cfg.pp * (self.vpp - 1) + 1 + (self.search_cfg.pp - self.stage_id - 1) * 2
        else:
            return self.search_cfg.pp - self.stage_id

    def _calculate_fw_idx(self, fw_idx, i, micro):
        if i * (self.vpp * self.search_cfg.pp) <= micro < i * (
                self.vpp * self.search_cfg.pp) + self.search_cfg.pp and self.vpp > 1:
            fw_idx += NumberConstant.FW_NORM_OP_NUM_ENABLE_PP_OTHER_STAGE
        else:
            fw_idx += NumberConstant.FW_NORM_OP_NUM_ENABLE_PP_LAST_STAGE
        return fw_idx

    def _calculate_bw_idx(self, bw_idx, i, micro):
        if i * (self.vpp * self.search_cfg.pp) <= micro < i * (
                self.vpp * self.search_cfg.pp) + self.search_cfg.pp or self.vpp == 1:
            bw_idx += NumberConstant.FW_NORM_OP_NUM_ENABLE_PP_LAST_STAGE
        else:
            bw_idx += NumberConstant.FW_NORM_OP_NUM_ENABLE_PP_OTHER_STAGE
        return bw_idx


class ProfilingLayerInfo:
    def __init__(self):
        self.time = []
        self.start_memory = []
        self.peak_memory = []
        self.reserved_memory = []
        self.operator_info = []
        self.communication_info = []

    def extend_attr(self, new_layer):
        for attr_name in self.__dict__.keys():
            obj_attr = getattr(self, attr_name)
            if isinstance(obj_attr, list):
                target_attr = getattr(new_layer, attr_name, [])
                obj_attr.extend(target_attr)
                setattr(self, attr_name, obj_attr)


class ProfilingModelInfo:
    def __init__(self):
        self.embedding = ProfilingLayerInfo()
        self.forward = ProfilingLayerInfo()
        self.loss = ProfilingLayerInfo()
        self.backward = ProfilingLayerInfo()
        self.optimizer = ProfilingLayerInfo()
        self.hccl_memory = []
        self.cann_and_driver_memory = []
        self.communication_matrix = []
        self.context_parallel_comm = []
        self.pipeline_parallel_comm = []
        self.data_parallel_comm = []
        self.tensor_parallel_comm = []
        self.expert_parallel_comm = []
        self.search_cfg = None
        self.stage_id = 0
        self.mc2_total_time = []
        self.matmul_total_time = []

    def extend_stage_info(self, new_model):
        for attr_name in self.__dict__.keys():
            obj_attr = getattr(self, attr_name)
            if isinstance(obj_attr, list):
                target_attr = getattr(new_model, attr_name, [])
                obj_attr.extend(target_attr)
                setattr(self, attr_name, obj_attr)
            elif isinstance(obj_attr, ProfilingLayerInfo):
                target_attr = getattr(new_model, attr_name, None)
                obj_attr.extend_attr(target_attr)


class BaseParallelCommunication:
    """
        Basic parallel communication information.
    """

    def __init__(self):
        self.total_time_ms: float = 0.0
        self.wait_time_ms: float = 0.0
        self.overlap_time_ms: float = 0.0
        self.details: List[dict] = []


class ExpertParallelCommunication(BaseParallelCommunication):
    """
        Expert parallel communication
    """

    def __init__(self):
        super(ExpertParallelCommunication, self).__init__()
        self.min_comm_time_ms: float = 0.0


class TensorParallelCommunication(BaseParallelCommunication):
    """
        Tensor parallel communication
    """

    def __init__(self):
        super(TensorParallelCommunication, self).__init__()
        self.fixed_time_ms: float = 0.0
        self.fixed_wait_time_ms: float = 0.0


class ContextParallelCommunication(BaseParallelCommunication):
    """
        Context parallel communication
    """

    def __init__(self):
        super(ContextParallelCommunication, self).__init__()
        self.vector_time_ms: float = 0.0


class DataParallelCommunication(BaseParallelCommunication):
    """
        Data parallel communication
    """

    def __init__(self):
        super(DataParallelCommunication, self).__init__()
        self.mlp_zero_time_ms: float = 0.0
        self.mlp_ag_time_ms: float = 0.0
        self.mlp_rs_time_ms: float = 0.0
        self.other_zero_time_ms: float = 0.0
        self.other_ag_time_ms: float = 0.0
        self.other_rs_time_ms: float = 0.0


class PipelineParallelCommunication(BaseParallelCommunication):
    """
        Pipeline parallel communication
    """
