import math
import os
import re
from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_config import ProfilingConfig, \
    ProfilingModelInfo
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_meta_parse import StructureAnalyseTool
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_operator_parse import AnalyseOperatorMsg
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_communication_parse import \
    AnalyseCommunicationMsg
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_memory_parse import AnalyseMemoryMsg
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_meta_parse import FileAnalyseTool


class ProfilingParser(ProfilingConfig):
    def __init__(self, root_path, search_cfg=None, args=None):
        super(ProfilingParser, self).__init__(search_cfg, args)
        self._root_path = root_path
        self._ascend_operator_details = None
        self.stage_id = 0
        self.rank_file_path = None
        self.model = ProfilingModelInfo()
        self.logger = get_logger('profiling_parser')

    def parse_fw_bw_structure(self, fw_norm_op_idx_list, bw_norm_op_idx_list):
        if self.search_cfg.pp > 1:
            fw_layer_start_index, bw_layer_start_index, recompute_fw, fw_per_micro_opt_num, bw_per_micro_opt_num = \
                self.search_first_operator_idx_for_per_layer_enable_pp(fw_norm_op_idx_list, bw_norm_op_idx_list)
        else:
            fw_layer_start_index, bw_layer_start_index, recompute_fw, fw_per_micro_opt_num, bw_per_micro_opt_num = \
                self.search_first_operator_idx_for_per_layer_disable_pp(fw_norm_op_idx_list, bw_norm_op_idx_list)
        for micro in range(self.micro_num):
            if self.per_micro_layer != 1:
                fw_per_micro_opt_num = fw_layer_start_index[micro][-1] - fw_layer_start_index[micro][-2]
                bw_per_micro_opt_num = bw_layer_start_index[micro][-1] - bw_layer_start_index[micro][-2]
            fw_layer_start_index[micro].append(fw_layer_start_index[micro][-1] + fw_per_micro_opt_num - 1)
            bw_layer_start_index[micro].insert(0, bw_layer_start_index[micro][-1] - bw_per_micro_opt_num)
        return fw_layer_start_index, bw_layer_start_index

    def parse_model_structure(self):
        self._update_profiling_file_path()
        kernel_details = FileAnalyseTool.analyse_csv_info(self.rank_file_path, 'kernel_details.csv')
        communication_details = FileAnalyseTool.analyse_json_info(self.rank_file_path, 'communication.json')
        memory_details = FileAnalyseTool.analyse_csv_info(self.rank_file_path, 'operator_memory.csv')
        memory_record_details = FileAnalyseTool.analyse_csv_info(self.rank_file_path, 'memory_record.csv')
        structure_cls = StructureAnalyseTool(self.rank_file_path, kernel_details)
        fw_norm_op_idx_list, bw_norm_op_idx_list, matmul_total_time, mc2_total_time = structure_cls.analyse_norm_op()
        self.model.matmul_total_time = [matmul_total_time]
        self.model.mc2_total_time = [mc2_total_time]
        fw_layer_start_index, bw_layer_start_index = self.parse_fw_bw_structure(fw_norm_op_idx_list,
                                                                                bw_norm_op_idx_list)

        self._parse_operator_info(kernel_details, fw_layer_start_index, bw_layer_start_index)
        self._parse_communication_info(communication_details, kernel_details)
        self._parse_memory_info(memory_details, memory_record_details)

    def parser(self):
        """
        Parse profiling files.
        Returns:
            model: ProfilingModelInfo
        """
        self.logger.info('>>>> Profiling parse starting!')
        self._parse_each_node()
        self.logger.info('>>>> Profiling parse success!')
        return self.model

    def _validate_file_path(self, filename, attr_name):
        file_path = os.path.join(self.rank_file_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        setattr(self, attr_name, file_path)

    def _update_profiling_file_path(self):
        self._validate_file_path('kernel_details.csv', '_kernel_details_csv_path')
        self._validate_file_path('memory_record.csv', '_memory_record_csv_path')
        self._validate_file_path('operator_memory.csv', '_operator_memory_csv_path')
        self._validate_file_path('npu_module_mem.csv', '_npu_module_mem_csv_path')
        self._validate_file_path('communication.json', '_communication_json_path')
        self._validate_file_path('op_statistic.csv', '_op_statistic_csv_path')

    def _extract_rank_file_path(self):
        """
        Get all rank file path, the profiling process generates the profiler_info_{rank_id}.json file.
        Returns:
            rank_file_path: Dict[rank_id] = path
        """

        def extract_rankid_from_filename(filename):
            match = re.search(r'profiler_info_(\d+)\.json', filename)
            if match:
                return int(match.group(1))
            else:
                return None

        rank_file_path = {}
        for ascend_dir in os.listdir(self._root_path):
            profiling_path = os.path.join(self._root_path, ascend_dir)
            if os.path.isdir(profiling_path) and 'ascend' in ascend_dir:
                json_files = [f
                              for f in os.listdir(profiling_path)
                              if f.endswith('.json') and f.startswith('profiler_info_')]
                if not json_files:
                    raise ValueError(f"Args profile error, JSON is not exist in {ascend_dir}.")

                rank_id = extract_rankid_from_filename(json_files[0])
                if rank_id is not None:
                    rank_file_path[rank_id] = profiling_path
        return rank_file_path

    def _join_rank_ascend_path(self, file_name):
        rank_file_path = os.path.join(self._root_path, file_name, "ASCEND_PROFILER_OUTPUT")
        if not os.path.exists(rank_file_path):
            raise f" {rank_file_path} is not exist."
        return rank_file_path

    def _get_first_rank_and_stage_id_of_each_stage(self, node_first_rank_id, devices_each_stage, rank_file_path):
        """
        Get the rank file path based on the number of devices each stage. For example:
              devices_each_node  devices_each_stage  node  pp
        1.    8                  16                  2     1
        2.    8                  8                   2     2
        3.    8                  4                   2     4
        """
        if devices_each_stage == self.devices_per_node:
            return self._join_rank_ascend_path(rank_file_path[node_first_rank_id]), self.node_rank
        elif devices_each_stage < self.devices_per_node:
            paths_and_ids = []
            stage_num_each_node = math.ceil(len(rank_file_path) / devices_each_stage)
            for i in range(stage_num_each_node):
                cur_stage_rank = i * devices_each_stage + node_first_rank_id
                cur_stage_id = i + self.node_rank * stage_num_each_node
                paths_and_ids.append((self._join_rank_ascend_path(rank_file_path[cur_stage_rank]), cur_stage_id))
            return paths_and_ids
        else:
            return self._join_rank_ascend_path(rank_file_path[node_first_rank_id]), self.node_rank // (
                self.nodes // self.search_cfg.pp)

    def _parse_first_rank_of_each_stage(self, rank_file_path: dict):
        """Parses the first rank file of each stage."""
        node_first_rank_id = self.node_rank * self.devices_per_node
        devices_each_stage = self.nodes * self.devices_per_node // self.search_cfg.pp
        paths_and_ids = self._get_first_rank_and_stage_id_of_each_stage(node_first_rank_id, devices_each_stage,
                                                                        rank_file_path)
        if isinstance(paths_and_ids, list):
            for path, stage_id in paths_and_ids:
                self.rank_file_path = path
                self.stage_id = stage_id
                self.model.stage_id = stage_id
                self.parse_model_structure()
        else:
            self.rank_file_path, self.stage_id = paths_and_ids
            self.model.stage_id = self.stage_id
            self.parse_model_structure()

    def _parse_each_node(self):
        rank_file_path = self._extract_rank_file_path()
        self._parse_first_rank_of_each_stage(rank_file_path)

    def _parse_operator_info(self, kernel_details, fw_layer_start_index, bw_layer_start_index):
        operator = AnalyseOperatorMsg(kernel_details)
        embedding_operator = operator.analyse_embedding(0, fw_layer_start_index[0][0] - 1)
        forward_operator = operator.analyse_forward(fw_layer_start_index[0][0], fw_layer_start_index[0][-1])
        loss_operator = operator.analyse_loss(fw_layer_start_index[0][-1], bw_layer_start_index[0][0] - 1)
        backward_operator = operator.analyse_backward(bw_layer_start_index[0][0], bw_layer_start_index[0][-1])
        optimizer_operator = operator.analyse_optimizer(bw_layer_start_index[0][-1] + 1, len(kernel_details) - 1)
        self.model.embedding.operator_info.append(embedding_operator)
        self.model.forward.operator_info.append(forward_operator)
        self.model.loss.operator_info.append(loss_operator)
        self.model.backward.operator_info.append(backward_operator)
        self.model.optimizer.operator_info.append(optimizer_operator)

    def _parse_memory_info(self, memory_details, memory_record_details):
        memory_cls = AnalyseMemoryMsg(self.rank_file_path, self.search_cfg, memory_details, stage_id=self.stage_id)
        memory_cls.update_norm_indices()
        embedding_start, embedding_peak = memory_cls.analyse_embedding()
        self.model.embedding.start_memory.append(embedding_start)
        self.model.embedding.peak_memory.append(embedding_peak)
        fw_start, fw_peak = memory_cls.analyse_forward()
        self.model.forward.start_memory.append(fw_start)
        self.model.forward.peak_memory.append(fw_peak)
        loss_start, loss_peak = memory_cls.analyse_loss()
        self.model.loss.start_memory.append(loss_start)
        self.model.loss.peak_memory.append(loss_peak)
        bw_start, bw_peak = memory_cls.analyse_backward()
        self.model.backward.start_memory.append(bw_start)
        self.model.backward.peak_memory.append(bw_peak)
        optimizer_start, optimizer_peak = memory_cls.analyse_optimizer()
        self.model.optimizer.start_memory.append(optimizer_start)
        self.model.optimizer.peak_memory.append(optimizer_peak)
        self.model.cann_and_driver_memory = memory_cls.analyse_cann_and_driver(memory_record_details)

    def _parse_communication_info(self, communication_details, kernel_details):
        communication_cls = AnalyseCommunicationMsg(self.search_cfg, communication_details, kernel_details)
        communication_cls.analyse_parallel_comm()
        self.model.tensor_parallel_comm.append(communication_cls.get_tp_comm())
        self.model.pipeline_parallel_comm.append(communication_cls.get_pp_comm())
        self.model.data_parallel_comm.append(communication_cls.get_dp_comm())
        self.model.context_parallel_comm.append(communication_cls.get_cp_comm())
        self.model.expert_parallel_comm.append(communication_cls.get_ep_comm())
