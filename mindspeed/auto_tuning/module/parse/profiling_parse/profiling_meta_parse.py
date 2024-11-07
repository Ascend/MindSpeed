import csv
import json
import os
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import SpecialOperatorName
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import NumberConstant
from mindspeed.auto_tuning.utils.file_utils import check_file_size


class FileAnalyseTool:
    """
        support csv and json parse
    """

    @classmethod
    def analyse_csv_info(cls, file_path: str, csv_name: str):
        csv_path = os.path.join(file_path, csv_name)
        try:
            with open(csv_path, newline='') as csvfile:
                check_file_size(csvfile)
                reader = csv.DictReader(csvfile)
                csv_details = list(reader)

        except FileNotFoundError as e:
            raise f"'Please check file name, {e}"
        except csv.Error as e:
            raise f"An error occurred while reading the CSV file: {e}"
        return csv_details

    @classmethod
    def analyse_json_info(cls, file_path: str, json_name: str):
        json_path = os.path.join(file_path, json_name)
        json_details = {"p2p": {}, "collective": {}}
        try:
            with open(json_path, 'r') as f:
                check_file_size(f)
                details = json.load(f)
            details_value = list(details.values())[0]
            for name, info in details_value.get('p2p', {}).items():
                comm_name = name.split("@")[0]
                json_details['p2p'][comm_name] = info["Communication Time Info"]
            for name, info in details_value.get('collective', {}).items():
                comm_name = name.split("@")[0]
                json_details['collective'][comm_name] = info["Communication Time Info"]
        except KeyError as e:
            raise f"'Please check file name, {e}"
        except Exception as e:
            raise f"Read communication file error: {e}"

        return json_details


class StructureAnalyseTool:
    """
        support structure parse
    """

    def __init__(self, rank_file_path, memory_details):
        self._rank_file_path = rank_file_path
        self._memory_details = memory_details
        self.fw_norm_op = SpecialOperatorName.FW_RMS_NORM_TYPE
        self.bw_norm_op = SpecialOperatorName.BW_RMS_NORM_TYPE
        self._search_special_norm_op()

    def analyse_norm_op(self):
        """ Analyse the norm op details in kernel_details.csv. """
        fw_norm_op_idx_list = []
        bw_norm_op_idx_list = []
        matmul_total_time = 0
        mc2_total_time = 0
        for idx, row in enumerate(self._memory_details):
            if "Name" not in row or "Type" not in row:
                continue
            if row["Type"] == "MatMulCommon":
                time = float(row["Duration(us)"]) / NumberConstant.CONVERSION_TIME
                matmul_total_time += time
                mc2_total_time += time
            if row["Type"] == "AllGatherMatmul" or row["Type"] == "MatmulReduceScatter":
                mc2_total_time += float(row["Duration(us)"]) / NumberConstant.CONVERSION_TIME
            if row["Type"] == self.fw_norm_op:
                fw_norm_op_idx_list.append(idx)
            elif row["Type"] == self.bw_norm_op:
                bw_norm_op_idx_list.append(idx)
        return fw_norm_op_idx_list, bw_norm_op_idx_list, matmul_total_time, mc2_total_time

    def get_fw_norm_op(self):
        return self.fw_norm_op

    def _search_special_norm_op(self):
        """ Special norm op: rms_norm, layer_norm, rms_norm_grad """
        op_statistic_details = FileAnalyseTool.analyse_csv_info(self._rank_file_path, 'op_statistic.csv')
        for op in op_statistic_details:
            if SpecialOperatorName.FW_LAYER_NORM_TYPE in op['OP Type']:
                self.fw_norm_op = SpecialOperatorName.FW_LAYER_NORM_TYPE
                self.bw_norm_op = SpecialOperatorName.BW_LAYER_NORM_TYPE
                break
