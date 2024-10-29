from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_config import (
    TensorParallelCommunication,
    DataParallelCommunication,
    PipelineParallelCommunication,
    ContextParallelCommunication,
    ExpertParallelCommunication,
    ProfilingConfig
)
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import NumberConstant, SpecialKeyName


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
        overlap_list = []
        overlap_time = float(info[SpecialKeyName.DURATION_US])
        op1 = self.kernel_details[index + 1]
        op2 = self.kernel_details[index + 2] if index + 2 < len(self.kernel_details) else None
        op1_name = op1[SpecialKeyName.NAME]
        hcom1_duration = float(op1[SpecialKeyName.DURATION_US])

        if op2 and op2[SpecialKeyName.ACCELERATOR_CORE] == 'HCCL':
            op2_name = op2[SpecialKeyName.NAME]
            hcom2_duration = float(op2[SpecialKeyName.DURATION_US])

            if hcom2_duration <= hcom1_duration:
                overlap_list.append(op2_name)
                overlap_record[op1_name] = min(overlap_time, hcom1_duration)
            else:
                overlap_list.append(op1_name)
                overlap_record[op1_name] = min(overlap_time, hcom2_duration)
        else:
            overlap_record[op1_name] = min(overlap_time, hcom1_duration)

        return overlap_record, overlap_list

    def is_compute_and_hcom_overlap(self, index, row):
        if index + 1 >= len(self.kernel_details):
            return False
        op1 = self.kernel_details[index + 1]
        if op1[SpecialKeyName.ACCELERATOR_CORE] != 'HCCL' or row[SpecialKeyName.ACCELERATOR_CORE] == 'HCCL':
            return False
        start_time = float(row[SpecialKeyName.START_TIME_US])
        duration = float(row[SpecialKeyName.DURATION_US])
        op1_start_time = float(op1[SpecialKeyName.START_TIME_US])
        return op1_start_time < start_time + duration

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

    def analyse_parallel_comm(self):
        self._analyse_communication_overlap()
        min_expert_time = None
        for name, info in self.collective_hcom.items():
            if 'hcom' not in name:
                continue
            if self.is_send_or_recv_op(name):
                self._analyse_pp_comm(name, info)
                continue
            if 'alltoall' in name:
                min_expert_time = self._analyse_ep_comm(name, info, min_expert_time)
                continue
            if self.search_cfg.tp > 1:
                self._analyse_tp_comm(name, info)
            self._analyse_dp_comm(name, info)
        if self.search_cfg.pp > 1 and self.search_cfg.cp > 1:
            self.pp_stream_id = self._analyse_pp_cp_process_id()
        else:
            self.pp_stream_id = None
        for name, info in self.p2p_hcom.items():
            if 'hcom' not in name:
                continue
            hcom_name = name.split('@')[0]
            stream_id = hcom_name.split('_')[3]
            if (self.pp_stream_id and self.pp_stream_id == stream_id) or self.search_cfg.cp == 1:
                self._analyse_pp_comm(name, info)
            else:
                self._analyse_cp_comm(name, info)

        self._get_zero1_hcom()
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
        comm_obj.total_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        comm_obj.wait_time_ms += (info[SpecialKeyName.WAIT_TIME_MS] + info[SpecialKeyName.IDLE_TIME_MS])
        hcom_name = name.split('@')[0]
        if isinstance(comm_obj, TensorParallelCommunication):
            if hcom_name in self.overlap_record:
                comm_obj.overlap_time_ms += self.overlap_record[hcom_name] / NumberConstant.CONVERSION_TIME
                comm_obj.fixed_wait_time_ms += (info[SpecialKeyName.WAIT_TIME_MS] + info[SpecialKeyName.IDLE_TIME_MS])
            else:
                comm_obj.fixed_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        elif hcom_name in self.overlap_record:
            comm_obj.overlap_time_ms += self.overlap_record[hcom_name] / NumberConstant.CONVERSION_TIME
        comm_obj.details.append({name: info})

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

    def _dp_comm_with_mlp_and_attention(self, mlp_process_id, process_id, name, info):
        if mlp_process_id and process_id == mlp_process_id:
            self.data_parallel_comm.mlp_zero_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
            if 'allGather' in name:
                self.data_parallel_comm.mlp_ag_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
            if 'reduceScatter' in name:
                self.data_parallel_comm.mlp_rs_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        else:
            self.data_parallel_comm.other_zero_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
            if 'allGather' in name:
                self.data_parallel_comm.other_ag_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
            if 'reduceScatter' in name:
                self.data_parallel_comm.other_rs_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]

    def _get_zero1_hcom(self):
        mlp_process_id = None
        if not self.data_parallel_comm.details:
            return
        if 'allGather' in list(self.data_parallel_comm.details[-1].keys())[0] \
                and (self.search_cfg.cp * self.search_cfg.dp / self.search_cfg.ep != 1):
            mlp_process_id = list(self.data_parallel_comm.details[-1].keys())[0].split('_')[3]
        for hcom in self.data_parallel_comm.details:
            for name, info in hcom.items():
                process_id = name.split('_')[3]
                if 'allReduce' in name and self.search_cfg.zero1:
                    continue
                self._dp_comm_with_mlp_and_attention(mlp_process_id, process_id, name, info)

    def _analyse_tp_comm(self, name, info):
        hcom_name = name.split('@')[0]
        if hcom_name in self.overlap_list:
            return
        if ('reduceScatter' in hcom_name or 'broadcast' in hcom_name) and not self.tp_stream_id:
            self.tp_stream_id = name.split('_')[3]
        if self.search_cfg.tp > 1 and self.tp_stream_id and name.split('_')[3] == self.tp_stream_id:
            self._accumulate_communication_stats(self.tensor_parallel_comm, name, info)

    def _analyse_pp_comm(self, name, info):
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

    def _analyse_ep_comm(self, name, info, min_expert_time):
        if not min_expert_time:
            min_expert_time = info[SpecialKeyName.ELAPSE_TIME_MS]
        else:
            min_expert_time = min(min_expert_time, info[SpecialKeyName.ELAPSE_TIME_MS])
        self.expert_parallel_comm.total_time_ms += info[SpecialKeyName.ELAPSE_TIME_MS]
        self.expert_parallel_comm.details.append({name: info})
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
