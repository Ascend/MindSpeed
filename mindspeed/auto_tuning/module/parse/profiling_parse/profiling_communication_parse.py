import os
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_config import (
    TensorParallelCommunication,
    DataParallelCommunication,
    PipelineParallelCommunication,
    ContextParallelCommunication,
    ExpertParallelCommunication,
    ProfilingConfig
)
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import SpecialKeyName
from mindspeed.auto_tuning.utils.utils import NumberConstant

# WAIT_TIME_RATIO : 等待时间占总时间比例
if not os.environ.get('WAIT_TIME_RATIO'):
    os.environ['WAIT_TIME_RATIO'] = "0.2"
RATIO = float(os.environ.get('WAIT_TIME_RATIO'))


class CommGroupInfo():
    def __init__(self):
        self.stream_id = ''
        self.first_commit_time = 0
        self.first_commit_name = ''
        self.stream_num_without_allreduce = 0
        self.stream_list = []


class StreamInfo():
    def __init__(self, name, info):
        self.name = name
        self.info = info


class ParallelCommGroup():
    def __init__(self):
        self.tp_comm_group = []
        self.cp_comm_group = []
        self.ep_comm_group = []
        self.pp_comm_group = []
        self.dp_comm_group = []
        self.dp_mlp_comm_group = []


class AnalyseCommunicationMsg(ProfilingConfig):
    """ Analyse communication massage. """

    def __init__(self, search_cfg, communication_details, kernel_details):
        super(AnalyseCommunicationMsg, self).__init__(search_cfg)
        self.collective_hcom = communication_details.get('collective', {})
        self.p2p_hcom = communication_details.get('p2p', {})
        self.kernel_details = kernel_details
        self.tensor_parallel_comm = TensorParallelCommunication()
        self.pipeline_parallel_comm = PipelineParallelCommunication()
        self.data_parallel_comm = DataParallelCommunication()
        self.context_parallel_comm = ContextParallelCommunication()
        self.expert_parallel_comm = ExpertParallelCommunication()
        self.pp_stream_id = None
        self.tp_stream_id = None
        self.overlap_record = {}
        self.overlap_list = []

    @classmethod
    def is_send_or_recv_op(cls, op_name: str) -> bool:
        return 'send' in op_name or 'receive' in op_name

    def get_hcom_and_hcom_overlap(self, index, info):
        current_name = self.kernel_details[index][SpecialKeyName.NAME]
        next_name = self.kernel_details[index + 1][SpecialKeyName.NAME]
        if current_name in self.overlap_list or next_name in self.overlap_list:
            return

        if index + 1 >= len(self.kernel_details):
            return

        hcom_time1 = float(info[SpecialKeyName.DURATION_US])
        hcom_time2 = float(self.kernel_details[index + 1][SpecialKeyName.DURATION_US])
        shorter_hcom = current_name if hcom_time1 <= hcom_time2 else next_name
        self.overlap_list.append(shorter_hcom)

    def get_compute_and_hcom_overlap(self, index, info):
        overlap_record = {}
        op_name = self.kernel_details[index][SpecialKeyName.NAME]
        overlap_list = [op_name]
        op = self.kernel_details[index]
        op_before = self.kernel_details[index - 1]
        op_after = self.kernel_details[index + 1]
        start_time = float(op[SpecialKeyName.START_TIME_US])
        duration = float(op[SpecialKeyName.DURATION_US])
        op_before_start_time = float(op_before[SpecialKeyName.START_TIME_US])
        op_before_duration_time = float(op_before[SpecialKeyName.DURATION_US])
        op_after_start_time = float(op_after[SpecialKeyName.START_TIME_US])
        op_after_duration_time = float(op_after[SpecialKeyName.DURATION_US])
        
        # 计算与前一个算子的隐藏: 
        overlap_time = 0
        if op_before_start_time + op_before_duration_time > start_time:
            overlap_time = op_before_start_time + op_before_duration_time - start_time
        # 计算与后一个算子的隐藏:
        if op_after_start_time < start_time + duration:
            if op_after_start_time + op_after_duration_time < start_time + duration:
                overlap_time = overlap_time + op_after_duration_time
            else:
                overlap_time = overlap_time + (start_time + duration - op_after_start_time)

        if index - 2 > 0:
            op_before = self.kernel_details[index - 2]
            op_before_start_time = float(op_before[SpecialKeyName.START_TIME_US])
            op_before_duration_time = float(op_before[SpecialKeyName.DURATION_US])
            
            
            # 计算与前一个算子的隐藏: 
            if op_before_start_time + op_before_duration_time > start_time:
                overlap_time = op_before_start_time + op_before_duration_time - start_time
            
        if index + 2 < len(self.kernel_details):
            op_after = self.kernel_details[index + 2]
            op_after_start_time = float(op_after[SpecialKeyName.START_TIME_US])
            op_after_duration_time = float(op_after[SpecialKeyName.DURATION_US])
            # 计算与后一个算子的隐藏:
            if op_after_start_time < start_time + duration:
                if op_after_start_time + op_after_duration_time < start_time + duration:
                    overlap_time = overlap_time + op_after_duration_time
                else:
                    overlap_time = overlap_time + (start_time + duration - op_after_start_time)
        overlap_record[op_name] = min(overlap_time, duration)

        return overlap_record, overlap_list

    def is_compute_and_hcom_overlap(self, index, row):
        # row 作为通信算子，判断前一个和后一个算子的影藏
        if index + 1 >= len(self.kernel_details) or index < 1:
            return False
        op_before = self.kernel_details[index - 1]
        op_after = self.kernel_details[index + 1]
        if row[SpecialKeyName.ACCELERATOR_CORE] != 'HCCL':
            return False
        start_time = float(row[SpecialKeyName.START_TIME_US])
        duration = float(row[SpecialKeyName.DURATION_US])
        op_before_start_time = float(op_before[SpecialKeyName.START_TIME_US])
        op_before_duration_time = float(op_before[SpecialKeyName.DURATION_US])
        op_after_start_time = float(op_after[SpecialKeyName.START_TIME_US])
        return (op_before_start_time + op_before_duration_time > start_time) or (
                    op_after_start_time < start_time + duration)

    def is_hcom_hcom_overlap(self, index, row):
        if index + 1 >= len(self.kernel_details):
            return False
        op1 = self.kernel_details[index + 1]
        if row[SpecialKeyName.ACCELERATOR_CORE] != 'HCCL' or op1[SpecialKeyName.ACCELERATOR_CORE] != 'HCCL':
            return False
        start_time = float(row[SpecialKeyName.START_TIME_US])
        duration = float(row[SpecialKeyName.DURATION_US])
        op1_start_time = float(op1[SpecialKeyName.START_TIME_US])
        return op1_start_time < start_time + duration

    def get_parallel_comm_group(self, collective_group, index):
        if index >= len(collective_group):
            return []
        return collective_group[index]

    def judge_p2p_comm(self, name):
        if ("send" in name or "receive" in name):
            return True
        return False

    def judge_first_comm(self, name, info, item_info):
        if 'allReduce' not in name:
            item_info.stream_num_without_allreduce += 1
            # p2p通信中会在comm中掺杂通信的情况,优先排空此处记录值
            if self.judge_p2p_comm(name) and not \
                    self.judge_p2p_comm(item_info.first_commit_name):
                item_info.first_commit_time = 0
            if item_info.first_commit_time == 0:
                item_info.first_commit_time = info["Start Timestamp(us)"]
                item_info.first_commit_name = name
        return item_info

    def get_comm_group(self, hcom_info, comm_group_list):
        for (name, info) in hcom_info.items():
            if 'hcom' not in name:
                continue
            hcom_name = name.split('@')[0]
            stream_id = hcom_name.split('_')[3]
            flag_new_group = True
            for item_info in comm_group_list:
                if item_info.stream_id == stream_id:
                    item_info.stream_list.append(StreamInfo(name, info))
                    item_info = self.judge_first_comm(name, info, item_info)
                    flag_new_group = False
                    break
            if flag_new_group:
                group_info = CommGroupInfo()
                group_info.stream_id = stream_id
                group_info.stream_list.append(StreamInfo(name, info))
                if 'allReduce' not in name:
                    group_info.stream_num_without_allreduce += 1
                    group_info.first_commit_time = info["Start Timestamp(us)"]
                    group_info.first_commit_name = name
                comm_group_list.append(group_info)
        return comm_group_list

    def reset_comm_list(self, comm_group_list):
        comm_group_sord_list = sorted(comm_group_list, key=lambda group_info: group_info.first_commit_time)
        comm_group_orderly_list = []
        for item in comm_group_sord_list:
            if item.first_commit_time > 0 and item.stream_num_without_allreduce > 1:
                comm_group_orderly_list.append(item)
        return comm_group_orderly_list

    # 此处的不同通信域的取值强依赖于现有通信域的排列顺序，即
    # TP CP EP PP DP
    def analyse_parallel_comm(self):
        min_expert_time = None
        parallel_comm_group = ParallelCommGroup()
        self._analyse_communication_overlap()
        comm_group_list = []
        comm_group_list = self.get_comm_group(self.collective_hcom, comm_group_list)
        comm_group_list = self.get_comm_group(self.p2p_hcom, comm_group_list)
        comm_group_orderly_list = self.reset_comm_list(comm_group_list)
        comm_group_index = 0
        if self.search_cfg.tp > 1:
            parallel_comm_group.tp_comm_group = self.get_parallel_comm_group(comm_group_orderly_list, comm_group_index)
            comm_group_index += 1
            logits_info_flag = False
            reduceScatter_index = 0
            for stream_info in parallel_comm_group.tp_comm_group.stream_list:
                if logits_info_flag:
                    if 'reduceScatter' in stream_info.name:
                        reduceScatter_index += 1
                        logits_info_flag = False
                    continue
                self._analyse_tp_comm(stream_info.name, stream_info.info)
                if 'reduceScatter' in stream_info.name:
                    reduceScatter_index += 1
                if self.search_cfg.pp == 1 and 'allReduce' in stream_info.name and reduceScatter_index > 2:
                    logits_info_flag = True

        if self.search_cfg.cp > 1:
            parallel_comm_group.cp_comm_group = self.get_parallel_comm_group(comm_group_orderly_list, comm_group_index)
            comm_group_index += 1
            for stream_info in parallel_comm_group.cp_comm_group.stream_list:
                self._analyse_cp_comm(stream_info.name, stream_info.info)
        if self.search_cfg.num_experts:
            ep_group = self.search_cfg.ep
            if self.search_cfg.moe_tp_extend_ep:
                ep_group = ep_group * self.search_cfg.tp

            parallel_comm_group.ep_comm_group = self.get_parallel_comm_group(comm_group_orderly_list, comm_group_index)
            self._megatron_ep_adaptation(parallel_comm_group.ep_comm_group.stream_list)
            if ep_group > 1 or "alltoall" in parallel_comm_group.ep_comm_group.stream_list[0].name:
                comm_group_index += 1
                for stream_info in parallel_comm_group.ep_comm_group.stream_list:
                    min_expert_time = self._analyse_ep_comm(stream_info.name, stream_info.info, min_expert_time)
        if self.search_cfg.pp > 1:
            parallel_comm_group.pp_comm_group = self.get_parallel_comm_group(comm_group_orderly_list, comm_group_index)
            comm_group_index += 1
            for stream_info in parallel_comm_group.pp_comm_group.stream_list:
                self._analyse_pp_comm(stream_info.name, stream_info.info)
        if self.search_cfg.dp * self.search_cfg.cp > 1:
            if comm_group_orderly_list[comm_group_index].stream_num_without_allreduce <= 1:
                # 这里需要跳过一组allReduce通信域
                comm_group_index += 1
            parallel_comm_group.dp_comm_group = self.get_parallel_comm_group(comm_group_orderly_list, comm_group_index)
            comm_group_index += 1
            for stream_info in parallel_comm_group.dp_comm_group.stream_list:
                self._analyse_dp_comm(stream_info.name, stream_info.info)
                self._dp_comm_with_attention(stream_info.name, stream_info.info)

            if self.search_cfg.dp * self.search_cfg.cp != self.search_cfg.ep and comm_group_index < len(
                    comm_group_orderly_list):
                parallel_comm_group.dp_mlp_comm_group = self.get_parallel_comm_group(comm_group_orderly_list,
                                                                                     comm_group_index)
                comm_group_index += 1
                for stream_info in parallel_comm_group.dp_mlp_comm_group.stream_list:
                    self._analyse_dp_comm(stream_info.name, stream_info.info)
                    self._dp_comm_with_mlp(stream_info.name, stream_info.info)

        if min_expert_time:
            self.expert_parallel_comm.min_comm_time_ms = len(self.expert_parallel_comm.details) * min_expert_time
            self.expert_parallel_comm.wait_time_ms = self.expert_parallel_comm.total_time_ms - \
                                                     self.expert_parallel_comm.min_comm_time_ms

    def get_tp_comm(self):
        return self.tensor_parallel_comm

    def get_pp_comm(self):
        return self.pipeline_parallel_comm

    def get_dp_comm(self):
        return self.data_parallel_comm

    def get_cp_comm(self):
        return self.context_parallel_comm

    def get_ep_comm(self):
        return self.expert_parallel_comm

    def is_tp_communication(self, name):
        return "reduceScatter" in name or "allGather" in name

    def _accumulate_communication_stats(self, comm_obj, name, info):
        if isinstance(comm_obj, TensorParallelCommunication) and not self.is_tp_communication(name):
            comm_obj.details.append({name: info})
            return

        old_total_time = comm_obj.total_time_ms
        comm_obj.total_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        wait_time = info[SpecialKeyName.ELAPSE_TIME_MS] - info[SpecialKeyName.TRANSIT_TIME_MS]
        ratio = wait_time / info[SpecialKeyName.ELAPSE_TIME_MS]
        fixed_time = 0
        if ratio <= RATIO:
            comm_obj.avg_ratio = \
                ((old_total_time - comm_obj.wait_time_ms) * comm_obj.avg_ratio + wait_time) \
                / (comm_obj.total_time_ms - comm_obj.wait_time_ms)
        else:
            if comm_obj.avg_ratio > 0.0001:
                comm_obj.wait_time_ms = comm_obj.wait_time_ms + wait_time - comm_obj.avg_ratio * info[
                    SpecialKeyName.ELAPSE_TIME_MS]
                fixed_time = info[SpecialKeyName.ELAPSE_TIME_MS] - wait_time + comm_obj.avg_ratio * info[
                    SpecialKeyName.ELAPSE_TIME_MS]
            else:
                comm_obj.wait_time_ms = comm_obj.wait_time_ms + wait_time - RATIO * info[SpecialKeyName.ELAPSE_TIME_MS]
                comm_obj.avg_ratio = RATIO
                fixed_time = info[SpecialKeyName.ELAPSE_TIME_MS] - wait_time + RATIO * info[
                    SpecialKeyName.ELAPSE_TIME_MS]

        self._overlap_fix(comm_obj, name, info, ratio, fixed_time)
        comm_obj.details.append({name: info})

    def _overlap_fix(self, comm_obj, name, info, ratio, fixed_time):
        hcom_name = name.split('@')[0]
        if isinstance(comm_obj, TensorParallelCommunication):
            if hcom_name in self.overlap_record:
                overlap = self.overlap_record[hcom_name] / NumberConstant.CONVERSION_TIME
                comm_obj.overlap_time_ms += overlap
                if ratio > RATIO:
                    comm_obj.fixed_wait_time_ms += overlap
                    comm_obj.overlap_time_ms = comm_obj.overlap_time_ms - overlap + fixed_time
            else:
                comm_obj.fixed_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        elif hcom_name in self.overlap_record:
            comm_obj.overlap_time_ms += self.overlap_record[hcom_name] / NumberConstant.CONVERSION_TIME

    def _analyse_pp_cp_process_id(self):
        pp_and_cp_send_id = []
        pp_and_cp_receive_id = []
        pp_stream_id = None
        for name, _ in self.p2p_hcom.items():
            if 'hcom' not in name:
                continue
            hcom_name = name.split('@')[0]
            stream_id = hcom_name.split('_')[3]
            if 'send' in name:
                if len(pp_and_cp_receive_id) > 1 and stream_id in pp_and_cp_receive_id:
                    pp_stream_id = stream_id
                if stream_id not in pp_and_cp_send_id:
                    pp_and_cp_send_id.append(stream_id)
            elif 'receive' in name:
                if len(pp_and_cp_send_id) > 1 and stream_id in pp_and_cp_send_id:
                    pp_stream_id = stream_id
                if stream_id not in pp_and_cp_receive_id:
                    pp_and_cp_receive_id.append(stream_id)
            if pp_stream_id is not None:
                break
        return pp_stream_id

    def _dp_comm_with_mlp(self, name, info):
        self.data_parallel_comm.mlp_zero_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        if 'allGather' in name:
            self.data_parallel_comm.mlp_ag_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        if 'reduceScatter' in name:
            self.data_parallel_comm.mlp_rs_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]

    def _dp_comm_with_attention(self, name, info):
        self.data_parallel_comm.other_zero_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        if 'allGather' in name:
            self.data_parallel_comm.other_ag_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        if 'reduceScatter' in name:
            self.data_parallel_comm.other_rs_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]

    def _analyse_tp_comm(self, name, info):
        hcom_name = name.split('@')[0]
        if ('reduceScatter' in hcom_name or 'allGather' in hcom_name):
            self._accumulate_communication_stats(self.tensor_parallel_comm, name, info)

    def _analyse_pp_comm(self, name, info):
        if "send" in name or "receive" in name:
            if self.pipeline_parallel_comm.min_pp_time:
                self.pipeline_parallel_comm.min_pp_time = \
                    min(self.pipeline_parallel_comm.min_pp_time,
                        info["Elapse Time(ms)"])
            else:
                self.pipeline_parallel_comm.min_pp_time = \
                    info["Elapse Time(ms)"]
        self._accumulate_communication_stats(self.pipeline_parallel_comm, name, info)

    def _analyse_dp_comm(self, name, info):
        hcom_name = name.split('@')[0]
        stream_id = hcom_name.split('_')[3]
        if stream_id != self.tp_stream_id and hcom_name.split('_')[1] in ["reduceScatter", "allGather"]:
            self._accumulate_communication_stats(self.data_parallel_comm, name, info)

    def _analyse_cp_comm(self, name, info):
        self._accumulate_communication_stats(self.context_parallel_comm, name, info)

        cp_vector_time = self._analyse_cp_vector_time()
        self.context_parallel_comm.vector_time_ms = cp_vector_time

    def _megatron_ep_adaptation(self, stream_list):
        for index, _ in enumerate(stream_list):
            while index < len(stream_list) and "allGather" in stream_list[index].name:
                if index + 1 < len(stream_list):
                    stream_list[index + 1].info[SpecialKeyName.ELAPSE_TIME_MS] += \
                        stream_list[index].info[SpecialKeyName.ELAPSE_TIME_MS]
                del stream_list[index]

    def _analyse_ep_comm(self, name, info, min_expert_time):
        self.expert_parallel_comm.total_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        self.expert_parallel_comm.details.append({name: info})
        if not min_expert_time:
            min_expert_time = info[SpecialKeyName.ELAPSE_TIME_MS]
        else:
            min_expert_time = min(min_expert_time, info[SpecialKeyName.ELAPSE_TIME_MS])
        return min_expert_time

    def _analyse_communication_overlap(self):
        for index, row in enumerate(self.kernel_details):
            if "Name" not in row or "Type" not in row:
                continue
            if self.is_compute_and_hcom_overlap(index, row):
                per_overlap_record, per_overlap_list = self.get_compute_and_hcom_overlap(index, row)
                self.overlap_record = {**self.overlap_record, **per_overlap_record}
                self.overlap_list.extend(per_overlap_list)
            elif self.is_hcom_hcom_overlap(index, row):
                self.get_hcom_and_hcom_overlap(index, row)

    def _cp_vector_operator_overlap(self, index, row):
        if index >= len(self.kernel_details) - 1:
            return False
        is_hccl = row[SpecialKeyName.ACCELERATOR_CORE] == 'HCCL'
        is_ai_vector_core = self.kernel_details[index + 1][SpecialKeyName.ACCELERATOR_CORE] == 'AI_VECTOR_CORE'
        is_time_overlap = float(self.kernel_details[index + 1][SpecialKeyName.START_TIME_US]) < float(
            row[SpecialKeyName.START_TIME_US]) + float(row[SpecialKeyName.DURATION_US])
        is_overlap = is_hccl and is_ai_vector_core and is_time_overlap
        if is_overlap and self.is_send_or_recv_op(row[SpecialKeyName.NAME]):
            return True
        return False

    def _analyse_cp_vector_time(self):
        is_cp_vector = False
        total_cp_vector = 0
        for index, row in enumerate(self.kernel_details):
            if "Name" not in row or "Type" not in row:
                continue
            is_ai_vector_core = row[SpecialKeyName.ACCELERATOR_CORE] == 'AI_VECTOR_CORE'
            if is_cp_vector and is_ai_vector_core and 'Grad' not in row[SpecialKeyName.NAME]:
                total_cp_vector += float(row[SpecialKeyName.DURATION_US]) / NumberConstant.CONVERSION_TIME
            elif is_cp_vector and row[SpecialKeyName.ACCELERATOR_CORE] != 'AI_VECTOR_CORE':
                is_cp_vector = False
            if self._cp_vector_operator_overlap(index, row):
                is_cp_vector = True
        return total_cp_vector
