from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.module.operator.operator_elemental import DictShape, ProfileList
from mindspeed.auto_tuning.module.operator.operator_shape_cal import cal_operator_flops


class DictNoteShape(DictShape):
    def __init__(self):
        super(DictNoteShape, self).__init__()
        self.tp = 0
        self.cp = 0
        self.ep = 0
        self.type = ""
        self.input_shape = ""
        self.output_shape = ""
        self.duration = 0.0
        self.num_experts = 0
        self.seq_length = 0
        self.flops = 0.0
        self.jit = 0

    def change_profile_into_dictshape(self, item, config_info):
        flops = cal_operator_flops(item.input_shapes.replace('"', ''),
                                   item.output_shapes.replace('"', ''),
                                   item.type)
        self.name = item.name
        self.type = item.type
        self.accelerator_core = item.accelerator_core
        self.index_name = ''
        self.tp = config_info.tp
        self.cp = config_info.cp
        self.ep = config_info.ep
        self.jit = config_info.jit
        self.num_experts = config_info.num_experts or 1
        self.seq_length = config_info.seq_length
        self.input_shape = item.input_shapes.replace('"', '')
        self.output_shape = item.output_shapes.replace('"', '')
        self.duration = float(item.duration_us)
        self.flops = flops


class OperatorNoteList:
    def __init__(self):
        self.operator_note_list = []
        self.operator_note_dict = {}
        self.seq_length = 0
        self._logger = get_logger('operator_note_list')

    def get_operator_note(self, block):
        self.get_operator_list(block.origin_profile_data_list)
        self.get_note_in_list(block)
        self.get_note_dict()

    def get_note_in_list(self, block):
        for (index, operator_note) in enumerate(self.operator_note_list):
            tp = block.origin_profile_data_list.data_list[index].config_info.tp
            cp = block.origin_profile_data_list.data_list[index].config_info.cp
            ep = block.origin_profile_data_list.data_list[index].config_info.ep
            num_experts = block.origin_profile_data_list.data_list[index].config_info.num_experts
            # Align the base block
            operator_note.reset_index_name(operator_note.fw, block.base_block.fw)
            operator_note.reset_index_name(operator_note.bw, block.base_block.bw)
            # Align the cp base block
            if cp > 1:
                _, cp_fw_index = operator_note.reset_index_name(operator_note.fw, block.cp_block.fw)
                _, cp_re_index = operator_note.reset_index_name(operator_note.bw, block.cp_block.re)
                _, cp_bw_index = operator_note.reset_index_name(operator_note.bw, block.cp_block.bw)
                if cp > 2:
                    operator_note.fw = block.cp_block.reset_index_diff_cp(operator_note.fw, block.cp_block.diff_list.fw,
                                                                          cp_fw_index, cp / 2)
                    operator_note.bw = block.cp_block.reset_index_diff_cp(operator_note.bw, block.cp_block.diff_list.re,
                                                                          cp_re_index, cp / 2)
                    operator_note.bw = block.cp_block.reset_index_diff_cp(operator_note.bw, block.cp_block.diff_list.bw,
                                                                          cp_bw_index, cp / 2)
            # Align the ep block
            if num_experts:
                if num_experts // ep >= 2:
                    operator_note.fw = block.ep_block.reset_index_diff_ep(operator_note.fw, block.ep_block.fw,
                                                                          (num_experts / ep) - 1)
                    operator_note.bw = block.ep_block.reset_index_diff_ep(operator_note.bw, block.ep_block.re,
                                                                          (num_experts / ep) - 1)
                    operator_note.bw = block.ep_block.reset_index_diff_ep(operator_note.bw, block.ep_block.bw,
                                                                          (num_experts / ep) - 1)

    def get_note_dict(self):
        for operator_note in self.operator_note_list:
            operator_list = operator_note.fw + operator_note.bw
            for operator in operator_list:
                dict_exist = False
                if operator.index_name in self.operator_note_dict.keys():
                    for dict_temp in self.operator_note_dict[operator.index_name]:
                        if dict_temp == operator:
                            dict_exist = True
                    if not dict_exist:
                        self.operator_note_dict[operator.index_name].append(operator)
                else:
                    self.operator_note_dict[operator.index_name] = [operator]

    def get_operator_list(self, origin_profile_data_list):
        self.seq_length = origin_profile_data_list.data_list[0].config_info.seq_length
        for (index, origin_profile_data) in enumerate(origin_profile_data_list.data_list):
            operator_note = ProfileList()
            self.change_profile_list_into_dict_shape_list(origin_profile_data.profile_list, operator_note,
                                                          origin_profile_data.config_info)
            self.operator_note_list.append(operator_note)

    @staticmethod
    def change_profile_list_into_dict_shape_list(change_profile_list, change_operator_list, config_info):
        for (index, item) in enumerate(change_profile_list.fw):
            dict_shape_fw = DictNoteShape()
            dict_shape_fw.change_profile_into_dictshape(item, config_info)
            change_operator_list.fw.append(dict_shape_fw)
        for (index, item) in enumerate(change_profile_list.bw):
            dict_shape_bw = DictNoteShape()
            dict_shape_bw.change_profile_into_dictshape(item, config_info)
            change_operator_list.bw.append(dict_shape_bw)
