import os
import stat
import pickle
import subprocess
import time

import torch
from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.utils.restricted_unpickler import restricted_loads
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_config import ProfilingModelInfo
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_parse import ProfilingParser


class GatherNodeProfiling:
    """
    Gather other node profiling result to rank0
    """

    def __init__(self, profiling_file_path):
        self.profiling_file_path = profiling_file_path
        self.fusion_model = ProfilingModelInfo()
        self.stage_id_list = []
        self.logger = get_logger('profiling_parser')

    @staticmethod
    def _extend_stage_lists(source, target):
        source.time.extend(target.time)
        source.start_memory.extend(target.start_memory)
        source.peak_memory.extend(target.peak_memory)
        source.communication_info.extend(target.communication_info)
        source.operator_info.extend(target.operator_info)

    def fuse_node_pkl(self):
        """
        Args:
            pkl_path: str

        Returns:
            fusion_model: ProfilingModelInfo
        """
        pkl_path = os.path.join(self.profiling_file_path, 'pkl_path')
        pkl_files = sorted(os.listdir(pkl_path))
        if len(pkl_files) > 1:
            self.logger.info(f'Get pp profiling parse result.')
            for pkl_file in pkl_files:
                node_pkl_path = os.path.join(pkl_path, pkl_file)
                with open(node_pkl_path, 'rb') as f:
                    pkl_model = restricted_loads(f)
                self._fuse_models(pkl_model)
        else:
            node_pkl_path = os.path.join(pkl_path, pkl_files[0])
            with open(node_pkl_path, 'rb') as f:
                pkl_model = restricted_loads(f)
            self.fusion_model = pkl_model
        return self.fusion_model

    def parse_node_pkl(self, args):
        parent_dir = os.path.dirname(self.profiling_file_path)
        ootb_node_path = os.path.join(parent_dir, f'ootb_{args.node_rank}.pkl')
        with open(ootb_node_path, 'rb') as f:
            cfg = restricted_loads(f)
        profiling_parser = ProfilingParser(self.profiling_file_path, search_cfg=cfg, args=args)
        profiling_res = profiling_parser.parser()
        if args.pipeline_model_parallel_size > 1 and profiling_parser.nodes > 1:
            ranks = [i * profiling_parser.devices_per_node for i in range(profiling_parser.nodes)]
            profiling_group = torch.distributed.new_group(ranks)
            gather_objects = [None for _ in range(profiling_parser.nodes)]
            torch.distributed.all_gather_object(gather_objects, profiling_res, group=profiling_group)
            for i in range(profiling_parser.nodes):
                pkl_path = os.path.join(self.profiling_file_path, 'pkl_path')
                if not os.path.exists(pkl_path):
                    os.mkdir(pkl_path)
                pkl_node_path = os.path.join(pkl_path, f'node_{i}.pkl')
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                mode = stat.S_IWUSR | stat.S_IRUSR
                with os.fdopen(os.open(pkl_node_path, flags, mode=mode), 'wb') as f:
                    pickle.dump(gather_objects[i], f)

            torch.distributed.barrier(group=profiling_group)
            torch.distributed.destroy_process_group(group=profiling_group)
        else:
            pkl_path = os.path.join(self.profiling_file_path, 'pkl_path')
            if not os.path.exists(pkl_path):
                os.mkdir(pkl_path)
            pkl_node_path = os.path.join(pkl_path, f'node_{args.node_rank}.pkl')
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(pkl_node_path, flags, mode=mode), 'wb') as f:
                pickle.dump(profiling_res, f)

    def _fuse_models(self, new_model):
        if new_model.stage_id not in self.stage_id_list:
            self.stage_id_list.append(new_model.stage_id)
            self.fusion_model.extend_stage_info(new_model)
