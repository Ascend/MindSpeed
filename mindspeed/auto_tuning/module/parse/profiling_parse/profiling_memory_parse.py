from typing import List

from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_meta_parse import StructureAnalyseTool
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import SpecialOperatorName
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_config import ProfilingConfig
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import SpecialKeyName


class AnalyseMemoryMsg(ProfilingConfig):
    """ Analyse memory massage. """

    def __init__(self, rank_file_path, search_cfg, memory_details, stage_id=0):
        super(AnalyseMemoryMsg, self).__init__(search_cfg)
        self._rank_file_path = rank_file_path
        self._memory_details = memory_details
        self._update_norm_op()
        self.fw_memory_indices: List[List[int]]
        self.bw_memory_indices: List[List[int]]
        self.fw_memory_per_micro_opt_num: int
        self.bw_memory_per_micro_opt_num: int
        self.stage_id = stage_id

    @staticmethod
    def compare_memory(row, start_memory, peak_memory):
        """compare memory"""
        if start_memory == 0:
            start_memory = float(row[SpecialKeyName.ALLOCATED_MEMORY])
        peak_memory = max(peak_memory, float(row[SpecialKeyName.ALLOCATED_MEMORY]))
        return start_memory, peak_memory

    @staticmethod
    def analyse_cann_and_driver(memory_record_details):
        app_mem = 0
        pta_mem = None
        for row in memory_record_details:
            if row[SpecialKeyName.COMPONENT] == 'APP':
                app_mem = row[SpecialKeyName.TOTAL_RESERVED]
            elif not pta_mem and row[SpecialKeyName.COMPONENT] == 'PTA':
                pta_mem = row[SpecialKeyName.TOTAL_RESERVED]
            if app_mem and pta_mem:
                break
        return [float(app_mem) - float(pta_mem)]

    def update_norm_indices(self):
        fw_memory_indices, bw_memory_indices = self._analyse_norm_op()
        if self.search_cfg.pp > 1:
            self.fw_memory_indices, \
                self.bw_memory_indices, \
                recompute_fw, \
                self.fw_memory_per_micro_opt_num, \
                self.bw_memory_per_micro_opt_num = \
                self.search_first_operator_idx_for_per_layer_enable_pp(fw_memory_indices, bw_memory_indices)
        else:
            self.fw_memory_indices, \
                self.bw_memory_indices, \
                recompute_fw, \
                self.fw_memory_per_micro_opt_num, \
                self.bw_memory_per_micro_opt_num = \
                self.search_first_operator_idx_for_per_layer_disable_pp(fw_memory_indices, bw_memory_indices)

    def analyse_embedding(self):
        em_start_memory, em_peak_memory = 0, 0
        if self.stage_id != 0:
            return [em_start_memory], [em_peak_memory]
        embedding_start_idx = 0
        for idx, msg in enumerate(self._memory_details[1:], start=1):
            op_name = msg[SpecialKeyName.NAME]
            if self.norm_op in op_name:
                break
            if SpecialOperatorName.EMBEDDING in op_name:
                embedding_start_idx = idx
                em_start_memory, em_peak_memory = self.compare_memory(self._memory_details[idx - 1],
                                                                      em_start_memory, em_peak_memory)
            if idx > embedding_start_idx != 0:
                em_start_memory, em_peak_memory = self.compare_memory(msg, em_start_memory, em_peak_memory)

        return [em_start_memory], [em_peak_memory]

    def analyse_forward(self):
        fw_start_memory = [0.0 for _ in range(self.micro_num)]
        fw_peak_memory = [0.0 for _ in range(self.micro_num)]
        for micro in range(self.micro_num):
            self.fw_memory_indices[micro].append(
                self.fw_memory_indices[micro][-1] + self.fw_memory_per_micro_opt_num - 1)
            fw_start_memory[micro] = float(
                self._memory_details[self.fw_memory_indices[micro][0]][SpecialKeyName.ALLOCATED_MEMORY])
            for msg in self._memory_details[self.fw_memory_indices[micro][0]: self.fw_memory_indices[micro][-1]]:
                fw_start_memory[micro], fw_peak_memory[micro] = \
                    self.compare_memory(msg, fw_start_memory[micro], fw_peak_memory[micro])

        return fw_start_memory, fw_peak_memory

    def analyse_loss(self):
        ls_start_memory, ls_peak_memory = 0, 0
        if self.stage_id != self.search_cfg.pp - 1:
            return [ls_start_memory], [ls_peak_memory]
        for idx, msg in enumerate(
                self._memory_details[self.fw_memory_indices[0][-1] + 1: self.bw_memory_indices[0][0]]):
            if 'norm' in self._memory_details[idx + 1 + self.fw_memory_indices[0][-1] + 1][SpecialKeyName.NAME]:
                continue
            ls_start_memory, ls_peak_memory = self.compare_memory(msg, ls_start_memory, ls_peak_memory)
        return [ls_start_memory], [ls_peak_memory]

    def analyse_backward(self):
        bw_start_memory = [0.0 for _ in range(self.micro_num)]
        bw_peak_memory = [0.0 for _ in range(self.micro_num)]
        for micro in range(self.micro_num):
            self.bw_memory_indices[micro].insert(0,
                                                 self.bw_memory_indices[micro][-1] - self.bw_memory_per_micro_opt_num)
            bw_start_memory[micro] = float(
                self._memory_details[self.bw_memory_indices[micro][0]][SpecialKeyName.ALLOCATED_MEMORY])
            for msg in self._memory_details[self.bw_memory_indices[micro][0]: self.bw_memory_indices[micro][-1]]:
                bw_start_memory[micro], bw_peak_memory[micro] = \
                    self.compare_memory(msg, bw_start_memory[micro], bw_peak_memory[micro])

        return bw_start_memory, bw_peak_memory

    def analyse_optimizer(self):
        op_start_memory, op_peak_memory = 0, 0
        for msg in self._memory_details[self.bw_memory_indices[-1][-1] + 1:]:
            op_start_memory, op_peak_memory = self.compare_memory(msg, op_start_memory, op_peak_memory)
        return [op_start_memory], [op_peak_memory]

    def _analyse_norm_op(self):
        fw_memory_indices, bw_memory_indices = [], []
        for index, row in enumerate(self._memory_details[1:], start=1):
            if self.norm_op in self._memory_details[index - 1][SpecialKeyName.NAME]:
                continue
            if self.norm_op in row[SpecialKeyName.NAME] \
                    and SpecialOperatorName.BACKWARD not in row[SpecialKeyName.NAME]:
                fw_memory_indices.append(index)
            elif self.norm_op in row[SpecialKeyName.NAME] \
                    and SpecialOperatorName.BACKWARD in row[SpecialKeyName.NAME]:
                bw_memory_indices.append(index)

        return fw_memory_indices, bw_memory_indices

    def _update_norm_op(self):
        structure_cls = StructureAnalyseTool(self._rank_file_path, self._memory_details)
        if structure_cls.fw_norm_op == SpecialOperatorName.FW_LAYER_NORM_TYPE:
            self.norm_op = SpecialOperatorName.LAYER_NORM
        else:
            self.norm_op = SpecialOperatorName.RMS_NORM
