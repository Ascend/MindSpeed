from typing import Dict, List, Optional, Tuple
import os
import stat
from argparse import Namespace

import pickle
from torch.nn import Module
import torch.distributed as dist

from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.utils.restricted_unpickler import restricted_loads
from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.module.memory.model_param import ModelParam
from mindspeed.auto_tuning.config.model_config import ModelConfig


_logger = get_logger("MindSpeedAdaptor")


class MindSpeedAdaptor:

    def __new__(cls):
        raise NotImplementedError("MindSpeedAdaptor is a static class.")

    @staticmethod
    def get_hardware(working_dir: str = str()) -> Hardware:
        import acl
        from .utils.mem_utils import mem_b_to_mb

        device_type = acl.get_soc_name()

        devices_per_node, _ = acl.rt.get_device_count()

        num_nodes = dist.get_world_size() // devices_per_node
        device_rank = dist.get_rank()
        node_rank = device_rank // devices_per_node
        device_id = device_rank % devices_per_node
        acl.rt.set_device(device_id)
        _, memory_limit, _ = acl.rt.get_mem_info(1)
        acl.rt.reset_device(device_id)

        host_ip = os.environ.get("MASTER_ADDR", None)

        if device_rank == 0:
            import getpass
            user_name = getpass.getuser()

            object_list = [user_name]
        else:
            object_list = [None]

        dist.broadcast_object_list(object_list)
        user_name: str = object_list[0]  # type: ignore

        hardware = Hardware()
        hardware.device_type = device_type
        hardware.host_ip = host_ip
        hardware.user_name = user_name
        hardware.memory_limit = mem_b_to_mb(memory_limit) - 2 * 1024
        hardware.devices_per_node = devices_per_node
        hardware.num_nodes = num_nodes
        hardware.node_rank = node_rank

        if working_dir and device_id == 0:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            hardware_filename = os.path.join(working_dir, Hardware.HARDWARE_PARSE_FILENAME)
            with os.fdopen(os.open(hardware_filename, flags, mode=mode), 'wb') as f:
                pickle.dump(hardware, f)

        return hardware

    @staticmethod
    def get_model_args(args: Namespace, hardware: Hardware, working_dir: str) -> ModelConfig:
        model_config = ModelConfig()
        for arg_name, arg_value in vars(args).items():
            if arg_name in model_config.__dict__:
                model_config.__dict__[arg_name] = arg_value
        model_config.global_world_size = args.auto_tuning_ranks

        if dist.get_rank() % hardware.devices_per_node == 0:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            model_config_filename = os.path.join(working_dir, ModelConfig.ARGS_PARSE_FILENAME)
            with os.fdopen(os.open(model_config_filename, flags, mode=mode), 'wb') as f:
                pickle.dump(model_config, f)

        return model_config

    @staticmethod
    def get_model_params(model: List[Module],
                         pipeline_model_parallel_rank: int,
                         hardware: Hardware,
                         output_path: str
                         ) -> List[ModelParam]:
        model_params: List[ModelParam] = list()

        def traverse_module_layers(module: Module, prefix: str):
            new_prefix = f"{prefix}{module.__class__.__name__}."

            if all(False for _ in module.children()):
                for param_name, param in module.named_parameters():
                    model_params.append(ModelParam(f"{new_prefix}{param_name}", param.numel()))
                return

            for sub_module in module.children():
                traverse_module_layers(sub_module, new_prefix)

        for module in model:
            traverse_module_layers(module, str())

        total_model_params = [None] * dist.get_world_size()
        dist.all_gather_object(total_model_params, (pipeline_model_parallel_rank, model_params))
        if dist.get_rank() % hardware.devices_per_node == 0:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(output_path, flags, mode=mode), 'wb') as f:
                pickle.dump(total_model_params, f)

        return model_params

    @staticmethod
    def set_argv(argv: List[str], input_path: str) -> List[str]:
        with open(input_path, mode="rb") as file:
            try:
                modified_argv: Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]] = \
                    restricted_loads(file)  # type: ignore
            except pickle.UnpicklingError as e:
                _logger.warning(f"Incorrect pickle format. UnpicklingError: {e}")
                raise e

        enabled_argv, disabled_argv = modified_argv

        for arg_name, arg_value in enabled_argv.items():
            # Flag args
            if arg_name == "--profile-ranks" and arg_value:
                argv.extend([arg_name, *[s.strip() for s in arg_value.strip("[]").split(",")]])
                continue
            if arg_value is None:
                try:
                    argv.index(arg_name)
                except ValueError:
                    argv.append(arg_name)
            # Non-flag args
            else:
                try:
                    argv[argv.index(arg_name) + 1] = arg_value
                except ValueError:
                    argv.extend([arg_name, arg_value])

        for arg_name, arg_value in disabled_argv.items():
            # Flag args
            if arg_value is None:
                try:
                    argv.pop(argv.index(arg_name))
                except ValueError:
                    continue
            # Non-flag args
            else:
                try:
                    i = argv.index(arg_name)
                    argv.pop(i)
                    argv.pop(i)
                except ValueError:
                    continue

        return argv
