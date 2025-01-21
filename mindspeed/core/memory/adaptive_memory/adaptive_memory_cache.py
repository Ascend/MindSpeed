# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import stat
import sys
import json
import hashlib
from typing import List
from pathlib import Path

import torch
import torch_npu
from megatron.training import get_args, print_rank_0
from megatron.core import parallel_state

import mindspeed
from .adaptive_memory_tool import SingletonBase, ModuleAction, LayerAction


class AdaptiveLayerMemPolicy:
    def __init__(self, recompute=None, swap=None, memory=0.0, time=sys.maxsize, adapt_type=LayerAction.ADAPTIVE):
        self.recompute: List[str] = recompute or []
        self.swap: List[str] = swap or []
        self.memory: float = memory
        self.time = time
        self.adapt_type = adapt_type

    def get_modules_by_tag(self, tag):
        if ModuleAction.RECOMPUTE == tag:
            return self.recompute
        elif ModuleAction.SWAP == tag:
            return self.swap
        else:
            msg = f"unknown layer policy tag name:{tag}"
            raise ValueError(msg)

    @staticmethod
    def parse_from_json(src_json):
        alp = AdaptiveLayerMemPolicy(memory=src_json["memory"], time=src_json["time"], recompute=[], swap=[])
        alp.recompute = [str(r) for r in src_json["recompute"]]
        alp.swap = [str(r) for r in src_json["swap"]]
        return alp

    def identity(self) -> str:
        self.sort_modules()
        modules = ",".join(self.recompute) + ":" + ",".join(self.swap)
        return hashlib.md5(modules.encode('utf-8')).hexdigest()

    def sort_modules(self):
        self.recompute.sort()
        self.swap.sort()

    def __eq__(self, other):
        if not isinstance(other, AdaptiveLayerMemPolicy):
            return False
        if len(self.recompute) != len(other.recompute) or len(self.swap) != len(other.swap):
            return False

        # sort values before compare
        self.sort_modules()
        other.sort_modules()

        return self.recompute == other.recompute and self.swap == other.swap

    def __repr__(self):
        result = {'recompute': self.recompute, 'swap': self.swap, 'memory': self.memory, 'time': self.time, 'adapt_type': self.adapt_type}
        return str(result)


class AdaptiveModelMemPolicy:
    def __init__(self, policy_type, polices, memory=0.0, time=sys.maxsize):
        self.policy_type: str = policy_type
        self.polices: List[AdaptiveLayerMemPolicy] = polices
        self.memory: float = memory
        self.time = time

    def __post_init__(self):
        if self.policy_type not in ["normal", "oom"]:
            raise ValueError(f"unknown policy type:{self.policy_type}, {self.__repr__()}")

    def __repr__(self):
        return str(self.polices)

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__, sort_keys=True)

    @staticmethod
    def parse_from_json(src_json):
        amp = AdaptiveModelMemPolicy(policy_type=src_json["policy_type"], polices=[])
        amp.polices = [AdaptiveLayerMemPolicy.parse_from_json(p) for p in src_json["polices"]]
        return amp

    def __eq__(self, other):
        if not isinstance(other, AdaptiveModelMemPolicy):
            return False
        if self.policy_type != other.policy_type or len(self.polices) != len(other.polices):
            return False

        cur_hash = sorted([x.identity() for x in self.polices])
        other_hash = sorted([x.identity() for x in other.polices])
        return cur_hash == other_hash


class PolicyCacheManager(metaclass=SingletonBase):

    def __init__(self):
        self.local_file_name_list = []
        self.normal_policy_cache: List[AdaptiveModelMemPolicy] = []
        self.oom_policy_cache: List[AdaptiveModelMemPolicy] = []

    def load_cache_file(self):
        self.local_file_name_list = self._buildup_filename()
        self.load_stage_cache_file()

    def load_stage_cache_file(self):
        cur_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if not os.path.isfile(self.local_file_name_list[cur_pp_rank]):
            print_rank_0(f"load history oom policy False!!!!!!!!: {self.local_file_name_list[cur_pp_rank]}")
            return

        with open(self.local_file_name_list[cur_pp_rank], "r") as f:
            for line in f:
                json_format = json.loads(line)
                policy: AdaptiveModelMemPolicy = AdaptiveModelMemPolicy.parse_from_json(json_format)
                self.oom_policy_cache.append(policy)
        print_rank_0(f"load history oom policy Success!!!!!!!!: {self.local_file_name_list[cur_pp_rank]}")

    @staticmethod
    def _get_version_file(src_path, key, version_file_name):
        version_path = src_path[:src_path.index(key) + len(key)]
        return os.path.join(version_path, version_file_name)

    def _get_software_version(self):
        torch_version: str = torch.__version__
        torch_npu_version: str = torch_npu.__version__

        library_path = os.environ.get("LD_LIBRARY_PATH").split(":")
        ascend_toolkit_path = next((x for x in library_path if "ascend-toolkit" in x), None)
        driver_path = next((x for x in library_path if "driver" in x), None)
        if ascend_toolkit_path is None or driver_path is None:
            return {}

        ascend_toolkit_version_file = self._get_version_file(ascend_toolkit_path, "ascend-toolkit", "version.cfg")
        driver_version_file = self._get_version_file(driver_path, "driver", "version.info")
        if not os.path.isfile(ascend_toolkit_version_file) or not os.path.isfile(driver_version_file):
            return {}

        with open(ascend_toolkit_version_file, "r") as f:
            f.readline()
            ascend_version = f.readline()

        with open(driver_version_file, "r") as f:
            driver_version = f.readline()

        return {
            "torch": torch_version,
            "torch_npu": torch_npu_version,
            "ascend_toolkit": ascend_version,
            "driver": driver_version
        }

    def _scan_dir_recursively(self, dir_name, md5s):
        with os.scandir(dir_name) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    self._scan_dir_recursively(entry.path, md5s)
                elif entry.is_file(follow_symlinks=False):
                    if not entry.path.endswith(".py"):
                        return
                    md5_instance = hashlib.md5()
                    with open(entry.path, "rb") as f:
                        md5_instance.update(f.read())
                    md5s.append(md5_instance.hexdigest())

    def _get_source_code_hash(self):
        mindspeed_path, = mindspeed.__path__
        md5s = []
        self._scan_dir_recursively(mindspeed_path, md5s)
        sorted(md5s)
        md5_instance = hashlib.md5()
        for x in md5s:
            md5_instance.update(x.encode('utf-8'))
        return md5_instance.hexdigest()

    def _buildup_filename(self):
        args = get_args()
        gbs = args.global_batch_size
        mbs = args.micro_batch_size
        seq_len = args.seq_length
        hidden = args.hidden_size
        tp = 1 if not args.tensor_model_parallel_size else args.tensor_model_parallel_size
        cp = 1 if not args.context_parallel_size else args.context_parallel_size
        sp = 1 if not args.sequence_parallel else tp
        ep = 1 if not args.expert_model_parallel_size else args.expert_model_parallel_size
        pp = 1 if not args.pipeline_model_parallel_size else args.pipeline_model_parallel_size
        world_size = args.world_size
        dp = world_size // tp // cp // pp

        arguments = {
            "global_batch_size": gbs,
            "micro_batch_size": mbs,
            "sequence_len": seq_len,
            "hidden": hidden,
            "tp": tp, "cp": cp, "sp": sp, "ep": ep, "dp": dp,
            "world_size": world_size,
            "source_hash": self._get_source_code_hash()
        }
        software_versions = self._get_software_version()
        arguments.update(software_versions)
        args_content = json.dumps(arguments, sort_keys=True)
        args_md5 = hashlib.md5(args_content.encode('utf-8')).hexdigest()

        mindspeed_home = os.path.dirname(os.path.dirname(mindspeed.__file__))
        adaptive_home = os.path.join(mindspeed_home, "adaptive_mem")
        Path(adaptive_home).mkdir(parents=True, exist_ok=True)
        file_abs_name_list = []

        for i in range(pp):
            file_name = f"b{mbs}_s{seq_len}_h{hidden}_tp{tp}_cp{cp}_w{world_size}_sp{sp}_ep{ep}_dp{dp}_stage{i}_{args_md5}.policy"
            file_abs_name = os.path.join(adaptive_home, file_name)
            file_abs_name_list.append(file_abs_name)

        return file_abs_name_list

    def _persistence(self):
        cur_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        cur_device_ranks = torch.cuda.device_count()
        total_ranks = torch.distributed.get_world_size()
        pp = 1 if not get_args().pipeline_model_parallel_size else get_args().pipeline_model_parallel_size
        rank_per_pp = total_ranks // pp
        # 不同节点的rank0需要存policy 以及 相同节点不同pp stage中的rank0需要存一下policy
        if torch.distributed.get_rank() % cur_device_ranks == 0 or (
                torch.distributed.get_rank() % rank_per_pp == 0 and torch.distributed.get_rank() % cur_device_ranks != 0):
            flags = os.O_WRONLY | os.O_CREAT
            mode = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.local_file_name_list[cur_pp_rank], flags, mode), 'w') as fout:
                fout.write("")
                for p in self.oom_policy_cache:
                    fout.write(p.to_json() + "\n")


    def add_normal_policy_cache(self, policy):
        if policy in self.normal_policy_cache:
            return

        self.normal_policy_cache.append(policy)

    def add_oom_policy_cache(self, policy):
        if policy in self.oom_policy_cache:
            return

        self.oom_policy_cache.append(policy)
        self._persistence()

    def delete_normal_policy_cache(self, policy):
        if policy not in self.normal_policy_cache:
            return

        self.normal_policy_cache.remove(policy)

    def check_in_cache(self, policy: AdaptiveModelMemPolicy):
        if policy is None:
            raise ValueError(f"unexpect policy")

        in_normal = next((x for x in self.normal_policy_cache if x == policy), None) is not None
        return in_normal or next((x for x in self.oom_policy_cache if x == policy), None) is not None

    def check_in_normal_cache(self, policy: AdaptiveModelMemPolicy):
        if policy is None:
            raise ValueError(f"unexpect policy")

        return next((x for x in self.normal_policy_cache if x == policy), None) is not None

    def check_in_oom_cache(self, policy: AdaptiveModelMemPolicy):
        if policy is None:
            raise ValueError(f"unexpect policy")

        return next((x for x in self.oom_policy_cache if x == policy), None) is not None
