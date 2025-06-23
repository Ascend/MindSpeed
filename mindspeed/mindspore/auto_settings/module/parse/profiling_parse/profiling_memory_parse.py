from typing import List

from mindspeed.auto_settings.module.parse.profiling_parse.profiling_constant import SpecialKeyName



@staticmethod
def analyse_cann_and_driver(memory_record_details):
    app_mem = 0
    pta_mem = None
    for row in memory_record_details:
        if row[SpecialKeyName.COMPONENT] == 'APP':
            app_mem = row[SpecialKeyName.TOTAL_RESERVED]
        elif not pta_mem and row[SpecialKeyName.COMPONENT] == 'MindSpore':
            pta_mem = row[SpecialKeyName.TOTAL_RESERVED]
        if app_mem and pta_mem:
            break
    return [float(app_mem) - float(pta_mem)]


def analyse_loss(self):
    ls_start_memory, ls_peak_memory = 0, 0
    if self.stage_id != self.search_cfg.pp - 1:
        return [ls_start_memory], [ls_peak_memory]

    for idx, msg in enumerate(
            self._memory_details[self.fw_memory_indices[0][-1] + 1: self.bw_memory_indices[0][0]]):
        if 'Norm' in self._memory_details[idx + 1 + self.fw_memory_indices[0][-1] + 1][SpecialKeyName.NAME]:
            continue
        ls_start_memory, ls_peak_memory = self.compare_memory(msg, ls_start_memory, ls_peak_memory)
    return [ls_start_memory], [ls_peak_memory]
