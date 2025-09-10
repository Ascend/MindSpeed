import contextlib
import copy
from enum import Enum
from typing import List, Union, Iterator
from functools import partial

import torch
from megatron.core import mpu
import torch.distributed as dist
from torch import nn
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import backward_step, deallocate_output_tensor, forward_step
from mindspeed.core.parallel_state import get_virtual_data_parallel_rank, get_virtual_data_parallel_world_size, set_virtual_data_parallel_rank

from mindspeed.core.pipeline_parallel.chimera.chimera_context import (
    ScheduleCell,
    BlockType,
    ChimeraBlock,
    ChimeraPipelineRankStageManager
)
from mindspeed.core.pipeline_parallel.chimera.weight_grad_store import WeightGradStore

from mindspeed.core.pipeline_parallel.chimera.communication import (
    send_forward_recv_backward,
    send_backward_recv_forward,
    send_forward_recv_forward,
    send_backward_recv_backward,
    send_forward,
    send_backward,
    recv_forward,
    recv_backward
)



class ChimeraBidirectionalPipelineScheduleManager:
    """
    Chimera is the bidirectional pipeline parallelism for large models published in SC21.
    "Shigang Li, Torsten Hoefler. Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines. 
    In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC21), 
    ACM, 2021. (Best Paper Finalist)"

    This is a class that produces the schedule of the Chimera bidirectional pipelines.
    """
    def __init__(self):
        self.is_first_run = True
        self.virtual_data_parallel_world_size = get_virtual_data_parallel_world_size()
        self.world_size = dist.get_world_size()
        self.pipeline_model_parallel_world_size = mpu.get_pipeline_model_parallel_world_size()
        self.rank = dist.get_rank()
        if self.virtual_data_parallel_world_size <= 0 or self.world_size <= 0 or self.pipeline_model_parallel_world_size <= 0:
            raise ValueError("The number of pipelines, devices, and stages should be positive integers")
        
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("The rank of the current process should be in the range of [0, self.world_size)")

        if self.world_size % self.pipeline_model_parallel_world_size != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages")
        
        if self.virtual_data_parallel_world_size & (self.virtual_data_parallel_world_size - 1) != 0:
            raise ValueError("The number of pipelines should be a power of 2")
        
        if self.virtual_data_parallel_world_size > self.pipeline_model_parallel_world_size:
            raise ValueError("The number of pipelines should not be greater than the number of stages")
        
        if self.pipeline_model_parallel_world_size % self.virtual_data_parallel_world_size != 0:
            raise ValueError("The number of stages should be a multiple of the number of pipelines")
        self.stage_mgr = ChimeraPipelineRankStageManager(self.virtual_data_parallel_world_size, self.world_size, self.pipeline_model_parallel_world_size, self.rank)


    def prepare(
        self,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):

        self.forward_step_func = forward_step_func
        if data_iterator is None:
            self.data_iterator = [None] * self.virtual_data_parallel_world_size
        else:
            self.data_iterator = data_iterator
        self.model = model
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.decoder_seq_length = decoder_seq_length
        self.forward_only = forward_only
        self.collect_non_loss_data = collect_non_loss_data

        config = get_model_config(model[0])
        self.config = config
        self.decouple_bw = config.decouple_bw
        self.model_type = get_model_type(model[0])
        if config.overlap_p2p_comm and config.batch_p2p_comm:
            raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
        # prepare for async grad allreduce. no_sync_func must be a list if overlap_grad_allreduce, reference to the megatron.training.train line 994
        no_sync_func = config.no_sync_func
        self.no_sync_func = no_sync_func
        self.no_sync_context = None

        self.tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        if config.sequence_parallel:
            self.tensor_shape[0] //= mpu.get_tensor_model_parallel_world_size()

        self.num_microbatches = num_microbatches
        
        if self.num_microbatches <= 0:
            raise ValueError("The size of micro-batch should be a positive integer")
        
        if self.num_microbatches % self.virtual_data_parallel_world_size != 0:
            raise ValueError("The size of micro-batch should be a multiple of the number of pipelines")
        
        if self.num_microbatches % self.pipeline_model_parallel_world_size != 0:
            raise ValueError("The size of micro-batch should be a multiple of the number of stages")

        self._construct()

        self.pipeline_rank_schedule = self.get_schedule(dist.get_rank())
        self.is_first_run = False


    def _merge_chimera_block(former_block: ChimeraBlock, latter_block: ChimeraBlock):
        """
        Merge two chimera blocks.

        Args:
            former_block (ChimeraBlock): The former chimera block.
            latter_block (ChimeraBlock): The latter chimera block.
        
        Returns:
            List[List[ScheduleCell]]: The merged schedule.
        """
        result = []
        former = former_block.schedule
        latter = latter_block.schedule

        # prepend the first half of the former block
        for i in range(len(former) // 2):
            result.append(former[i])

        # former index and latter index
        fid, lid = len(former) // 2, 0

        # merge with two pointers
        while fid < len(former):
            double_former = True
            double_latter = True
            merge_one_step = False
            
            for rank in range(latter_block.pipeline_model_parallel_world_size):
                if fid > len(former) - 2 or former[fid + 1][rank].is_idle() != former[fid][rank].is_idle():
                    double_former = False
                
                if lid > len(latter) - 2 or latter[lid + 1][rank].is_idle() != latter[lid][rank].is_idle():
                    double_latter = False
                
                if former[fid][rank].is_idle() and not latter[lid][rank].is_idle():
                    former[fid][rank] = latter[lid][rank]
                    merge_one_step = True
            
            result.append(former[fid])
            if double_former:
                result.append(former[fid + 1])
            
            if merge_one_step and double_latter:
                result.append(latter[lid + 1])
            
            if merge_one_step:
                lid += double_latter + 1

            fid += double_former + 1

        while lid < len(latter):
            result.append(latter[lid])
            lid += 1
        
        return result

    def _construct(self):
        """
        Construct the schedule.
        """
        cur_micro_id = 0
        blocks: List[ChimeraBlock] = []
        micro_per_pipeline = self.num_microbatches // self.virtual_data_parallel_world_size

        micro_per_pipeline_block = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_world_size
        micros = [[i * micro_per_pipeline + j for j in range(micro_per_pipeline)] for i in range(self.virtual_data_parallel_world_size)]

        def append_block(block_type: BlockType):
            blocks.append(ChimeraBlock(block_type, self.virtual_data_parallel_world_size, self.world_size, self.pipeline_model_parallel_world_size, self.rank, self.num_microbatches, micros, cur_micro_id, self.stage_mgr))

        while cur_micro_id < micro_per_pipeline:
            append_block(BlockType.FORWARD)
            append_block(BlockType.BACKWARD)
            cur_micro_id += micro_per_pipeline_block

        for i in range(len(blocks) - 1, 0, -1):
            blocks[i - 1].schedule = ChimeraBidirectionalPipelineScheduleManager._merge_chimera_block(blocks[i - 1], blocks[i])
        
        # transpose the schedule to normal dimensions
        sched_trans = blocks[0].schedule
        sched: List[List[ScheduleCell]] = []
        for i in range(self.pipeline_model_parallel_world_size):
            row = []
            for sched_tran in sched_trans:
                row.append(sched_tran[i])

            sched.append(row)

        self.schedule = sched
        if mpu.get_pipeline_model_parallel_rank() == 0:
            print(f"Chimera schdule: {self}")
    
    def get_schedule(self, rank: int) -> List[ScheduleCell]:
        """
        Get the schedule of the pipeline with the given rank.

        Args:
            rank (int): The global rank of the pipeline.

        Returns:
            List[ScheduleCell]: The schedule for this rank.
        """
        if rank < 0 or rank >= self.world_size:
            raise ValueError("The rank of the pipeline should be in the range of [0, num_devices)")

        per_pipeline_devices = self.world_size // self.pipeline_model_parallel_world_size
        group_rank = rank // per_pipeline_devices
        
        return self.schedule[group_rank]
    
    def __str__(self):
        result = 'ChimeraBidirectionalPipelineScheduleManager(\n'
        result += f'  num_pipelines = {self.virtual_data_parallel_world_size},\n'
        result += f'  num_stages = {self.pipeline_model_parallel_world_size},\n'
        result += f'  num_microbatches = {self.num_microbatches},\n'
        result += '  schedule = [\n'
        for row in self.schedule:
            result += '    '
            for cell in row:
                if cell.is_idle():
                    result += '    '
                else:
                    if cell.is_sync():
                        result += '{0: >3}{1}'.format(cell.pipeline_model_parallel_rank, cell.type.value)
                    else:
                        result += '{0: >3}{1}'.format(cell.micro_id, cell.type.value)
            result += '\n'
        result += '  ]\n'
        result += ')\n'
        return result

    def reset(self):
        self.input_tensors = [[] for _ in range(self.virtual_data_parallel_world_size)]
        self.output_tensors = [[] for _ in range(self.virtual_data_parallel_world_size)]
        self.output_grad_tensors = [[] for _ in range(self.virtual_data_parallel_world_size)]
        self.input_grad_tensors = [[] for _ in range(self.virtual_data_parallel_world_size)]

        self.send_forward_batch_id = [0 for _ in range(self.virtual_data_parallel_world_size)]
        self.send_backward_batch_id = [0 for _ in range(self.virtual_data_parallel_world_size)]
        self.recv_forward_batch_id = [0 for _ in range(self.virtual_data_parallel_world_size)]
        self.recv_backward_batch_id = [0 for _ in range(self.virtual_data_parallel_world_size)]

        self.deallocate_output_tensors = []
        self.comm_ops = []

        self.forward_batch_id = [0 for _ in range(self.virtual_data_parallel_world_size)]
        self.backward_batch_id = [0 for _ in range(self.virtual_data_parallel_world_size)]
        self.forward_data_store = []
        self.grad_sync_handles = []
        self.grad_sync_virtual_data_parallel_ranks = []
        self.total_num_tokens = 0

    def _send_forward(self):
        vdp_rank = get_virtual_data_parallel_rank()
        send_batch_id = self.send_forward_batch_id[vdp_rank]
        tensor_to_send = self.output_tensors[vdp_rank][send_batch_id]
        send_forward(tensor_to_send, self.config)
        deallocate_output_tensor(tensor_to_send, deallocate_pipeline_outputs=self.config.deallocate_pipeline_outputs)
        self.send_forward_batch_id[vdp_rank] += 1
    
    def _recv_forward(self):
        vdp_rank = get_virtual_data_parallel_rank()
        input_tensor = recv_forward(self.tensor_shape, self.config)
        self.input_tensors[vdp_rank].append(input_tensor)
        self.recv_forward_batch_id[vdp_rank] += 1
    
    def _send_backward(self):
        vdp_rank = get_virtual_data_parallel_rank()
        send_batch_id = self.send_backward_batch_id[vdp_rank]
        tensor_to_send = self.input_grad_tensors[vdp_rank][send_batch_id]
        send_backward(tensor_to_send, self.config)
        self.input_grad_tensors[vdp_rank][send_batch_id] = None
        self.send_backward_batch_id[vdp_rank] += 1
    
    def _recv_backward(self):
        vdp_rank = get_virtual_data_parallel_rank()
        output_grad_tensor = recv_backward(self.tensor_shape, self.config)
        self.output_grad_tensors[vdp_rank].append(output_grad_tensor)
        self.recv_backward_batch_id[vdp_rank] += 1
       
    def _send_forward_recv_backward(self, vdp_rank_1, vdp_rank_2):
        send_batch_id = self.send_forward_batch_id[vdp_rank_1]
        tensor_to_send = self.output_tensors[vdp_rank_1][send_batch_id]
        output_grad_tensor = send_forward_recv_backward(output_tensor=tensor_to_send, tensor_shape=self.tensor_shape, config=self.config, vdp_rank_1=vdp_rank_1, vdp_rank_2=vdp_rank_2)

        deallocate_output_tensor(tensor_to_send, deallocate_pipeline_outputs=self.config.deallocate_pipeline_outputs)
        self.output_grad_tensors[vdp_rank_2].append(output_grad_tensor)
        
        self.send_forward_batch_id[vdp_rank_1] += 1
        self.recv_backward_batch_id[vdp_rank_2] += 1

    def _send_backward_recv_forward(self, vdp_rank_1, vdp_rank_2):
        send_batch_id = self.send_backward_batch_id[vdp_rank_1]
        tensor_to_send = self.input_grad_tensors[vdp_rank_1][send_batch_id]
        input_tensor = send_backward_recv_forward(input_tensor_grad=tensor_to_send, tensor_shape=self.tensor_shape, config=self.config, vdp_rank_1=vdp_rank_1, vdp_rank_2=vdp_rank_2)

        self.input_tensors[vdp_rank_2].append(input_tensor)
        self.input_grad_tensors[vdp_rank_1][send_batch_id] = None

        self.send_backward_batch_id[vdp_rank_1] += 1
        self.recv_forward_batch_id[vdp_rank_2] += 1
    
    def _send_forward_recv_forward(self, vdp_rank_1, vdp_rank_2):
        send_batch_id = self.send_forward_batch_id[vdp_rank_1]
        tensor_to_send = self.output_tensors[vdp_rank_1][send_batch_id]
        input_tensor = send_forward_recv_forward(output_tensor=tensor_to_send, tensor_shape=self.tensor_shape, config=self.config, vdp_rank_1=vdp_rank_1, vdp_rank_2=vdp_rank_2)

        self.input_tensors[vdp_rank_2].append(input_tensor)
        deallocate_output_tensor(tensor_to_send, deallocate_pipeline_outputs=self.config.deallocate_pipeline_outputs)

        self.send_forward_batch_id[vdp_rank_1] += 1
        self.recv_forward_batch_id[vdp_rank_2] += 1
    
    def _send_backward_recv_backward(self, vdp_rank_1, vdp_rank_2):
        send_batch_id = self.send_backward_batch_id[vdp_rank_1]
        tensor_to_send = self.input_grad_tensors[vdp_rank_1][send_batch_id]

        output_grad_tensor = send_backward_recv_backward(input_tensor_grad=tensor_to_send, tensor_shape=self.tensor_shape, config=self.config, vdp_rank_1=vdp_rank_1, vdp_rank_2=vdp_rank_2)

        self.input_grad_tensors[vdp_rank_1][send_batch_id] = None
        self.output_grad_tensors[vdp_rank_2].append(output_grad_tensor)

        self.send_backward_batch_id[vdp_rank_1] += 1
        self.recv_backward_batch_id[vdp_rank_2] += 1
    
    def forward(self):
        vdp_rank = get_virtual_data_parallel_rank()
        batch_id = self.forward_batch_id[vdp_rank]
        if len(self.input_tensors[vdp_rank]) <= batch_id:
            self.input_tensors[vdp_rank].append(None)
        input_tensor = self.input_tensors[vdp_rank][batch_id]
        output_tensor, num_tokens = forward_step(
            forward_step_func=self.forward_step_func,
            data_iterator=self.data_iterator[vdp_rank],
            model=self.model[vdp_rank],
            num_microbatches=self.num_microbatches // self.virtual_data_parallel_world_size,
            input_tensor=input_tensor,
            forward_data_store=self.forward_data_store,
            config=self.config,
            collect_non_loss_data=self.collect_non_loss_data
        )
        self.total_num_tokens += num_tokens.item()
        self.output_tensors[vdp_rank].append(output_tensor)
        if self.forward_only:
            self.input_tensors[vdp_rank][batch_id] = None
            self.output_tensors[vdp_rank][batch_id] = None
        self.forward_batch_id[vdp_rank] += 1
    
    def backward(self):
        vdp_rank = get_virtual_data_parallel_rank()
        batch_id = self.backward_batch_id[vdp_rank]
        if len(self.output_grad_tensors[vdp_rank]) <= batch_id:
            self.output_grad_tensors[vdp_rank].append(None)
        input_tensor = self.input_tensors[vdp_rank][batch_id]
        output_tensor = self.output_tensors[vdp_rank][batch_id]
        output_grad_tensor = self.output_grad_tensors[vdp_rank][batch_id]
                
        input_grad_tensor = backward_step(input_tensor, output_tensor, output_grad_tensor, self.model_type, self.config)
        self.input_grad_tensors[vdp_rank].append(input_grad_tensor)
        self.input_tensors[vdp_rank][batch_id] = None
        self.output_tensors[vdp_rank][batch_id] = None
        self.output_grad_tensors[vdp_rank][batch_id] = None
        self.backward_batch_id[vdp_rank] += 1

    def wait_all(self):
        print(f">>> [Rank {mpu.get_pipeline_model_parallel_rank()}]: start to allreduce in data parallel group <<<", flush=True)
        for vdp_rank in self.grad_sync_virtual_data_parallel_ranks:
            set_virtual_data_parallel_rank(vdp_rank)
            self.config.finalize_model_grads_func([self.model[vdp_rank]])
        print(f">>> [Rank {mpu.get_pipeline_model_parallel_rank()}]: finish allreduce in data parallel group <<<", flush=True)
        self.grad_sync_virtual_data_parallel_ranks = []
    
    def assert_pipeline_finish(self):
        for i in range(self.virtual_data_parallel_world_size):
            if not all(ele is None for ele in self.input_tensors[i]):
                raise ValueError(f"input_tensors[{i}] contains non-None values")
            if not all(ele is None for ele in self.input_grad_tensors[i]):
                raise ValueError(f"input_grad_tensors[{i}] contains non-None values")
            if not all(ele is None for ele in self.output_tensors[i]):
                raise ValueError(f"output_tensors[{i}] contains non-None values")
            if not all(ele is None for ele in self.output_grad_tensors[i]):
                raise ValueError(f"output_grad_tensors[{i}] contains non-None values")
    
        expected_batch_id = self.num_microbatches // self.virtual_data_parallel_world_size
        if not all(ele == expected_batch_id for ele in self.forward_batch_id):
            raise ValueError(
                f"forward_batch_id contains invalid values (expected {expected_batch_id}), got {self.forward_batch_id}"
            )
        if not all(ele == expected_batch_id for ele in self.backward_batch_id):
            raise ValueError(
                f"backward_batch_id contains invalid values (expected {expected_batch_id}), got {self.backward_batch_id}"
            )
        if len(self.grad_sync_virtual_data_parallel_ranks) != 0:
            raise ValueError(
                f"grad_sync_virtual_data_parallel_ranks is not empty: {self.grad_sync_virtual_data_parallel_ranks}"
            )

    def run(self):
        def compute_warmup(world_size, rank, target_world_size):
            if world_size == target_world_size:
                return target_world_size - 1 - rank
            if rank < world_size // 2:
                return compute_warmup(world_size // 2, rank, target_world_size)
            else:
                rank = world_size - 1 - rank
                return compute_warmup(world_size // 2, rank, target_world_size)
            
        def next_cell_helper():
            nonlocal idx
            # skip idle
            while(self.pipeline_rank_schedule[idx].is_idle()):
                idx += 1
            cell = self.pipeline_rank_schedule[idx]
            idx += 1
            return cell
        
        def start_allreduce_helper():
            nonlocal cur_cell, vdp_rank
            # is last microbatch?
            if cur_cell.micro_id == (self.num_microbatches // self.virtual_data_parallel_world_size) * (vdp_rank + 1) - 1:
                if self.model[vdp_rank].ddp_config.overlap_grad_reduce:
                    self.model[vdp_rank].start_grad_sync()
                self.grad_sync_virtual_data_parallel_ranks.append(vdp_rank)
        target_world_size = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_world_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        num_warmup = compute_warmup(self.pipeline_model_parallel_world_size, pp_rank, target_world_size)

        idx = 0
        num_cool_down = num_warmup
        num_stable = self.num_microbatches * 2 - num_warmup - num_cool_down

        # warmup
        for _ in range(num_warmup):
            
            cell = next_cell_helper()
            vdp_rank = cell.virtual_data_parallel_rank
            set_virtual_data_parallel_rank(vdp_rank)
            self._recv_forward()
            self.forward()
            self._send_forward()
        
        # stable
        cur_cell = next_cell_helper()
        vdp_rank = cur_cell.virtual_data_parallel_rank
        set_virtual_data_parallel_rank(vdp_rank)
        self._recv_forward()
        if self.decouple_bw:
            WeightGradStore.start_decouple()
        for i in range(num_stable):            # compute
            if cur_cell.is_forward():
                self.forward()
                cur_type = True
            elif cur_cell.is_backward():
                self.backward()
                cur_type = False
            if i < num_stable - 1:
                # communicate with next cell
                next_cell = next_cell_helper()
                vdp_rank_2 = next_cell.virtual_data_parallel_rank
                if next_cell.is_forward():
                    next_type = True
                elif next_cell.is_backward():
                    next_type = False

                if cur_type and next_type:
                    self._send_forward_recv_forward(vdp_rank_1=vdp_rank, vdp_rank_2=vdp_rank_2)
                elif cur_type and not next_type:
                    self._send_forward_recv_backward(vdp_rank_1=vdp_rank, vdp_rank_2=vdp_rank_2)
                elif not cur_type and next_type:
                    self._send_backward_recv_forward(vdp_rank_1=vdp_rank, vdp_rank_2=vdp_rank_2)
                else:
                    self._send_backward_recv_backward(vdp_rank_1=vdp_rank, vdp_rank_2=vdp_rank_2)
                # prepare for next iteration
                cur_cell = next_cell
                vdp_rank = cur_cell.virtual_data_parallel_rank
                set_virtual_data_parallel_rank(vdp_rank)
            else:
                # last cell in stable phase, communicate separately
                if cur_type:
                    self._send_forward()
                else:
                    self._send_backward()
            if not cur_type:    # backward step
                if self.decouple_bw:
                    WeightGradStore.pop()
                start_allreduce_helper()
        if self.decouple_bw:
            WeightGradStore.start_decouple()
        # cool down
        for _ in range(num_cool_down):
            cur_cell = next_cell_helper()
            vdp_rank = cur_cell.virtual_data_parallel_rank
            set_virtual_data_parallel_rank(vdp_rank)
            self._recv_backward()
            self.backward()
            self._send_backward()
            if self.decouple_bw:
                WeightGradStore.pop()
            start_allreduce_helper()
        if self.decouple_bw:
            WeightGradStore.end_decouple()
            

        self.wait_all()
        self.assert_pipeline_finish()
        return self.forward_data_store


    def __call__(self, *args, **kwargs):
        if self.is_first_run:
            self.prepare(*args, **kwargs)
        self.reset()
        return self.run()



chimera_pipeline_schedule_manager = None


def get_chimera_forward_backward_func():
    global chimera_pipeline_schedule_manager
    if chimera_pipeline_schedule_manager is None:
        chimera_pipeline_schedule_manager = ChimeraBidirectionalPipelineScheduleManager()
    return chimera_pipeline_schedule_manager



if __name__ == "__main__":
    print(f"=" * 50)
    cps = ChimeraBidirectionalPipelineScheduleManager()
    cps.prepare(None, None, model=[None, None], num_microbatches=8, seq_length=1024, micro_batch_size=1, decoder_seq_length=1024)
    print(cps.get_schedule(0))