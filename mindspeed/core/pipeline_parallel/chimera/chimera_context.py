from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
import copy


class CellType(Enum):
    """
    An enumeration of the types of cells in the schedule table.
    """
    IDLE = 'i'
    FORWARD = 'f'
    BACKWARD = 'b'
    SYNC = 's'


@dataclass
class ScheduleCell:
    """
    A data class representing a cell in the schedule table.
    """
    type: CellType = CellType.IDLE
    micro_id: int = -1
    virtual_data_parallel_rank: int = 0
    pipeline_model_parallel_rank: int = -1

    def is_forward(self):
        return self.type == CellType.FORWARD
    
    def is_backward(self):
        return self.type == CellType.BACKWARD

    def is_sync(self):
        return self.type == CellType.SYNC

    def is_idle(self):
        return self.type == CellType.IDLE


class ChimeraPipelineRankStageManager:
    """
    PipelineRankStageManager for Chimera pipeline scheduling.

    The mapping is stored in two lists of lists:
    - `stage_to_rank_map[pipeline_id][pipeline_model_parallel_rank]` is the rank of the device
    that runs the stage with the given index in the pipeline with the given index.
    - `rank_to_stage_map[pipeline_id][rank]` is the index of the stage
    that runs on the device with the given rank in the pipeline with the given index.

    All of "index", "id" in the following description is 0-based.

    Notice that you have to provide the rank of current process.

    This is because for pipeline parallelism, we may have multiple devices
    for one single stage, which makes inner parallelism like DP, SP, TP possible.

    Then we have to know the rank of the current process to determine which device
    in the corresponding ranks should be used.

    For example, if we have 4 devices and 2 stages, and need two chimera pipelines.

    From the viewpoint of rank 0, the stage to rank map may look like:
    ```
    [
        [0, 2],
        [2, 0]
    ]
    ```
    From the viewpoint of rank 1, the stage to rank map may look like:
    ```
    [
        [1, 3],
        [3, 1]
    ]
    ```
    """
    def __init__(self, num_pipelines: int, num_devices: int, num_stages: int, rank: int):
        """
        Initialize the ChimeraPipelineRankStageManager.

        Args:
            num_pipelines (int): The number of pipelines.
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.

        Raises:
            ValueError: If current rank or the number of pipelines, devices,
            or stages is invalid.
        """

        if num_pipelines <= 0 or num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of pipelines, devices, and stages should be positive integers")

        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")

        if num_devices % num_stages != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages")
        
        if num_pipelines & (num_pipelines - 1) != 0:
            raise ValueError("The number of pipelines should be a power of 2")
        
        if num_pipelines > num_stages:
            raise ValueError("The number of pipelines should not be greater than the number of stages")
        
        if num_stages % num_pipelines != 0:
            raise ValueError("The number of stages should be a multiple of the number of pipelines")

        self.virtual_data_parallel_world_size = num_pipelines
        self.world_size = num_devices
        self.pipeline_model_parallel_world_size = num_stages
        self.rank = rank
        self._construct()

    @property
    def num_prs_keys(self):
        return self.virtual_data_parallel_world_size
    
    @property
    def num_stages(self):
        return self.pipeline_model_parallel_world_size
    
    @property
    def num_devices(self):
        return self.world_size

    def get_rank_to_stage_map(self, pipeline_id: int) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The rank to stage map.
        """
        return self._rank_to_stage_map[pipeline_id]

    def get_stage_to_rank_map(self, pipeline_id: int) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The stage to rank map.
        """
        return self._stage_to_rank_map[pipeline_id]

    def _construct(self):
        self._stage_to_rank_map = [[-1 for _ in range(self.pipeline_model_parallel_world_size)] for _ in range(self.virtual_data_parallel_world_size)]
        self._rank_to_stage_map = [[-1 for _ in range(self.world_size)] for _ in range(self.virtual_data_parallel_world_size)]
        
        devices_per_stage = self.world_size // self.pipeline_model_parallel_world_size

        def _get_pipeline_meta(pipeline_id):
            """
            Get the start rank and step of the pipeline with the given index.
            """
            # Up or down pipelines has different start ranks
            threshold = self.virtual_data_parallel_world_size // 2

            if pipeline_id < threshold:
                start_rank = pipeline_id * self.world_size // threshold
                step = 1
            else:
                start_rank = (pipeline_id - threshold) * self.world_size // threshold
                start_rank = self.world_size - start_rank - devices_per_stage
                step = -1
            return start_rank, step

        for pipeline_id in range(self.virtual_data_parallel_world_size):
            start_rank, step = _get_pipeline_meta(pipeline_id)

            for rank in range(self.world_size):
                if rank % devices_per_stage == 0:
                    calc_rank = start_rank + self.rank % devices_per_stage
                    self._stage_to_rank_map[pipeline_id][rank // devices_per_stage] = calc_rank
                
                offset = 0
                if step == -1:
                    offset = devices_per_stage - 1
                self._rank_to_stage_map[pipeline_id][(start_rank + offset) % self.world_size] = rank // devices_per_stage

                start_rank = (start_rank + step + self.world_size) % self.world_size


class BlockType(Enum):
    """
    An enumeration of the types of blocks in the schedule table.
    """
    FORWARD = 'f'
    FORWARD_DOUBLE = 'd'
    BACKWARD = 'b'

    def to_cell_type(self):
        if self == BlockType.FORWARD:
            return CellType.FORWARD
        if self == BlockType.FORWARD_DOUBLE:
            return CellType.FORWARD
        if self == BlockType.BACKWARD:
            return CellType.BACKWARD
        raise ValueError(f"Invalid block type: {self}")


class ChimeraBlock:
    """
    A class representing a Chimera block in the Chimera pipeline scheduling.
    """
    def __init__(self,
                 block_type: BlockType,
                 num_pipelines: int,
                 num_devices: int,
                 num_stages: int,
                 rank: int,
                 num_microbatches: int,
                 micros: List[List[int]],
                 start_micro_id: int,
                 stage_mgr: ChimeraPipelineRankStageManager):
        """
        Initialize the ChimeraBlock.

        Args:
            type (BlockType): The type of the block. It can be FORWARD, FORWARD_DOUBLE, or BACKWARD.
            num_pipelines (int): The number of pipelines.
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.
            num_microbatches (int): The size of micro-batch.
            micros (List[List[int]]): The micros to be scheduled.
            start_micro_id (int): The start micro id of this block.
        """
        self.stage_mgr = stage_mgr
        
        self._type = block_type
        self.virtual_data_parallel_world_size = num_pipelines
        self.world_size = num_devices
        self.pipeline_model_parallel_world_size = num_stages
        self.rank = rank
        self.num_microbatches = num_microbatches
        
        self.schedule: List[List[ScheduleCell]] = []

        self._construct(micros, start_micro_id)


    def _construct(self, micros: List[List[int]], start_micro_id: int):
        micro_per_pipeline = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_world_size
        devices_per_stage = self.world_size // self.pipeline_model_parallel_world_size

        stage_map = [[0 for _ in range(micro_per_pipeline)] for _ in range(self.virtual_data_parallel_world_size)]
        for pipeline_id in range(self.virtual_data_parallel_world_size):
            for micro_id in range(micro_per_pipeline):
                stage_map[pipeline_id][micro_id] = -2 * micro_id

        while True:
            micro_inserted = True
            subschedule = [ScheduleCell() for _ in range(self.pipeline_model_parallel_world_size)]
            subschedule_dup = [ScheduleCell() for _ in range(self.pipeline_model_parallel_world_size)]

            for pipeline_id in range(self.virtual_data_parallel_world_size):
                for micro_id in range(micro_per_pipeline):
                    pipeline_model_parallel_rank = stage_map[pipeline_id][micro_id]
                    if pipeline_model_parallel_rank < 0 or pipeline_model_parallel_rank >= self.pipeline_model_parallel_world_size:
                        stage_map[pipeline_id][micro_id] += 1
                        continue

                    micro_inserted = False
                    if self._type == BlockType.BACKWARD:
                        step = -1 if pipeline_id < self.virtual_data_parallel_world_size // 2 else 1
                        first_stage_rank = self.stage_mgr.get_stage_to_rank_map(pipeline_id)[-1]
                    else:
                        step = 1 if pipeline_id < self.virtual_data_parallel_world_size // 2 else -1
                        first_stage_rank = self.stage_mgr.get_stage_to_rank_map(pipeline_id)[0]

                    # group_rank is a virtual rank that does not consider inner parallelism
                    group_rank = first_stage_rank // devices_per_stage + step * pipeline_model_parallel_rank
                    group_rank = (group_rank + self.pipeline_model_parallel_world_size) % self.pipeline_model_parallel_world_size
                    
                    if self._type == BlockType.FORWARD_DOUBLE:
                        micro_index = micro_id * 2
                    else:
                        micro_index = micro_id
                    
                    cell_ref = subschedule[group_rank]
                    cell_ref.virtual_data_parallel_rank = pipeline_id
                    if self._type == BlockType.BACKWARD:
                        cell_ref.pipeline_model_parallel_rank = self.pipeline_model_parallel_world_size - 1 - pipeline_model_parallel_rank
                    else:
                        cell_ref.pipeline_model_parallel_rank = pipeline_model_parallel_rank
                    cell_ref.type = self._type.to_cell_type()
                    cell_ref.micro_id = micros[pipeline_id][micro_index + start_micro_id]

                    if self._type == BlockType.FORWARD_DOUBLE:
                        subschedule_dup[group_rank] = copy.copy(subschedule[group_rank])
                        subschedule_dup[group_rank].micro_id = micros[pipeline_id][micro_index + start_micro_id + 1]

                    stage_map[pipeline_id][micro_id] += 1
            
            if micro_inserted:
                break
            self.schedule.append(subschedule)
            if self._type == BlockType.FORWARD_DOUBLE:
                self.schedule.append(subschedule_dup)
    
    def __str__(self):
        result = 'ChimeraBlock(\n'
        result += f'  num_pipelines = {self.virtual_data_parallel_world_size},\n'
        result += f'  num_stages = {self.pipeline_model_parallel_world_size},\n'
        # the real num_microbatches of a block is num_stages
        result += f'  num_microbatches = {self.pipeline_model_parallel_world_size},\n'
        result += '  schedule = [\n'
        for i in range(self.pipeline_model_parallel_world_size):
            result += '    '
            for shd in self.schedule:
                cell = shd[i]
                if cell.is_idle():
                    result += '    '
                else:
                    result += '{0: >3}{1}'.format(cell.micro_id, cell.type.value)
            result += '\n'
        result += '  ]\n'
        result += ')\n'

        return result