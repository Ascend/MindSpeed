from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import IntEnum
import os
import torch
import torch.distributed as dist

from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.mindspeed_adaptor.mindspeed_runner import MindSpeedRunner
from mindspeed.auto_settings.utils.file_utils import restricted_write
from mindspeed.auto_settings.utils.logger import get_logger


class ExecutorFlag(IntEnum):
    PARSE_ARGS = 1
    PARSE_MODEL = 2
    PROFILE = 3


@dataclass
class Parameters:
    num_layers_config: str = "--num-layers"
    num_experts_config: str = "--num-experts"
    seq_length_config: str = "--seq-length"
    max_position_embeddings_config: str = "--max-position-embeddings"
    micro_batch_size_config: str = "--micro-batch-size"
    global_batch_size_config: str = "--global-batch-size"
    recompute_granularity_config: str = "--recompute-granularity"
    recompute_method_config: str = "--recompute-method"
    recompute_num_layers_config: str = "--recompute-num-layers"
    adaptive_recompute_device_swap_config: str = "--adaptive-recompute-device-swap"
    enable_token_rearrange_opt_config: str = "--enable-token-rearrange-opt"
    tensor_model_parallel_size_config: str = "--tensor-model-parallel-size"
    pipeline_model_parallel_size_config: str = "--pipeline-model-parallel-size"
    num_layers_per_virtual_pipeline_stage_config: str = "--num-layers-per-virtual-pipeline-stage"
    expert_model_parallel_size_config: str = "--expert-model-parallel-size"
    context_parallel_size_config: str = "--context-parallel-size"
    use_distributed_optimizer_config: str = "--use-distributed-optimizer"
    use_ascend_mc2_config: str = "--use-ascend-mc2"
    train_iters_config: str = "--train-iters"
    untie_embeddings_and_output_weights_config: str = "--untie-embeddings-and-output-weights"
    moe_tp_extend_ep_config: str = "--moe-tp-extend-ep"
    moe_alltoall_overlap_comm_config: str = "--moe-alltoall-overlap-comm"
    moe_allgather_overlap_comm_config: str = "--moe-allgather-overlap-comm"
    dist_train_config: str = "--hetero-parallel"
    mm_model_config: str = "--mm-model"
    mm_data_config: str = "--mm-data"
    mm_tool_config: str = "--mm-tool"

    profile_config: str = "--profile"
    profile_step_start_config: str = "--profile-step-start"
    profile_step_end_config: str = "--profile-step-end"
    profile_ranks_config: str = "--profile-ranks"
    profile_level_config: str = "--profile-level"
    profile_with_cpu_config: str = "--profile-with-cpu"
    profile_with_stack_config: str = "--profile-with-stack"
    profile_with_memory_config: str = "--profile-with-memory"
    profile_record_shapes_config: str = "--profile-record-shapes"
    profile_save_path_config: str = "--profile-save-path"


class MindSpeedExecutor:
    PARSE_ARGS_ENV = "OOTB_OPTIMIZER_PARSE_ARGS"
    PARSE_MODEL_ENV = "OOTB_OPTIMIZER_PARSE_MODEL"
    PROFILING_ENV = "OOTB_OPTIMIZER_PROFILING"
    ENABLED_ENV_MARKER = "TRUE"

    def __init__(self, runner: MindSpeedRunner, param: Parameters = Parameters()) -> None:
        self.runner = runner
        self.param = param
        self._logger = get_logger("MindSpeedExecutor")

    def execute(
        self,
        output_filename: str,
        model_config: Optional[ModelConfig] = None,
        cfg: Optional[SearchConfig] = None,
        flag: ExecutorFlag = ExecutorFlag.PROFILE,
        gloo_group=None
    ) -> int:
        if flag == ExecutorFlag.PARSE_ARGS:
            returncode = self._prepare(output_filename, model_config, cfg=cfg, flag=flag)
            return returncode
        else:
            if gloo_group:
                dist.monitored_barrier(group=gloo_group, wait_all_ranks=True)
                dist.broadcast_object_list([output_filename, cfg, flag], group=gloo_group)
                returncode = self._prepare(output_filename, model_config, cfg=cfg, flag=flag)
                dist.barrier(group=gloo_group)
            else:
                dist.monitored_barrier(wait_all_ranks=True)
                dist.broadcast_object_list([output_filename, cfg, flag])
                returncode = self._prepare(output_filename, cfg=cfg, flag=flag)
                dist.barrier()
                
            return returncode

    def wait(self,
        cfg=None,
        model_config: Optional[ModelConfig] = None,
        gloo_group=None
    ):
        count = 0
        while True:
            try:
                self._logger.info(f"[#{count}] Waiting for master.....")
                if gloo_group:
                    dist.monitored_barrier(group=gloo_group, wait_all_ranks=True)
                    bcast_list: List[Any] = [None] * 3
                    dist.broadcast_object_list(bcast_list, group=gloo_group)
                    output_filename, cfg, flag = bcast_list
                    self._prepare(output_filename, model_config, cfg=cfg, flag=flag)
                    dist.barrier(group=gloo_group)
                else:
                    dist.monitored_barrier(wait_all_ranks=True)
                    bcast_list: List[Any] = [None] * 3
                    dist.broadcast_object_list(bcast_list)
                    output_filename, cfg, flag = bcast_list
                    self._prepare(output_filename, model_config, cfg=cfg, flag=flag)
                    dist.barrier()
            except RuntimeError as e:
                if "successfully reached monitoredBarrier" in str(e):
                    count += 1
                    if count > 10:
                        self._logger.critical(f"Wait timeout, shutting down.")
                        raise e
                elif "Connection closed by peer" in str(e):
                    self._logger.info("Master shuts down, exiting.....")
                    return
                else:
                    raise e

    def _prepare(
        self,
        output_filename: str,
        model_config: Optional[ModelConfig] = None,
        cfg: Optional[SearchConfig] = None,
        flag: ExecutorFlag = ExecutorFlag.PROFILE
    ) -> int:
        from mindspeed.auto_settings.config.system_config import get_system_config
        modified_env = self._prepare_env(flag)
        modified_argv = self._prepare_argv(model_config.sub_work_dir, output_filename, cfg)
        if not os.path.exists(model_config.sub_work_dir):
            os.mkdir(model_config.sub_work_dir)

        if cfg:
            restricted_write(os.path.join(model_config.sub_work_dir, f"at_{get_system_config().node_rank}.pkl"), cfg)

        returncode = self.runner.run(modified_argv, modified_env)

        return returncode

    def _prepare_env(self, flag: ExecutorFlag) -> Dict[str, str]:
        env = self.runner.get_base_env()

        env.pop(self.PARSE_ARGS_ENV, None)
        env.pop(self.PARSE_MODEL_ENV, None)
        env.pop(self.PROFILING_ENV, None)

        if flag == ExecutorFlag.PARSE_ARGS:
            env.update({self.PARSE_ARGS_ENV: self.ENABLED_ENV_MARKER})
        elif flag == ExecutorFlag.PARSE_MODEL:
            env.update({self.PARSE_MODEL_ENV: self.ENABLED_ENV_MARKER})
        elif flag == ExecutorFlag.PROFILE:
            env.update({self.PROFILING_ENV: self.ENABLED_ENV_MARKER})

        return env

    def _prepare_argv(
        self,
        work_dir: str,
        output_filename: str,
        cfg: Optional[SearchConfig],
        model_config: Optional[ModelConfig] = None,
    ) -> List[str]:
        argv = self.runner.get_base_argv()
        self._update_flag(argv, "--auto-settings", False)
        if self.param.profile_save_path_config:
            print(f"work_dir:{work_dir}, output_filename{output_filename}")
            self._update_param(argv, self.param.profile_save_path_config, os.path.join(work_dir, output_filename))

        filter_param = [
            "--ampipe-degree",
            "--pipe-experts-multi-data"
        ]
        for f in filter_param:
            self._update_param(argv, f, None)
        filter_flag = [
            "--swap-attention",
            "--ampipe-tp-sp-comm-overlap",
            "--use-pipe-experts",
            "--pipe-experts-multi-stream",
            "--recompute-in-advance",
            "--recompute-in-bubble",
            "--use-nanopipe"
        ]
        for f in filter_flag:
            self._update_flag(argv, f, False)

        if cfg:
            self._modify_model_argv(argv, cfg)
            self._modify_parallel_argv(argv, cfg)
            self._modify_feature_argv(argv, cfg)
            self._modify_profile_argv(argv, cfg)
            # multi_model auto-tuning only functional under dist train feature for now
            if self.param.dist_train_config:
                self._modify_multi_model_argv(argv, cfg)
        else:
            cfg = SearchConfig(
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                pipeline_model_parallel_size=1,
                num_layers_per_virtual_pipeline_stage=None,
                expert_model_parallel_size=1
            )
            self._modify_parallel_argv(argv, cfg)

        return argv

    @staticmethod
    def _update_param(argv: List[str], arg_name: str, arg_value: Optional[Any]):
        while arg_name in argv:
            i = argv.index(arg_name)
            argv.pop(i + 1)
            argv.pop(i)

        if arg_value:
            argv.extend((arg_name, str(arg_value)))

    @staticmethod
    def _update_flag(argv: List[str], arg_name: str, switch: bool):
        while arg_name in argv:
            argv.remove(arg_name)

        if switch:
            argv.append(arg_name)

    def _modify_model_argv(self, argv: List[str], cfg: SearchConfig):
        if self.param.num_layers_config:
            self._update_param(argv, self.param.num_layers_config, cfg.num_layers)

        if self.param.num_experts_config:
            self._update_param(argv, self.param.num_experts_config, cfg.num_experts)

        if self.param.seq_length_config and self.param.max_position_embeddings_config:
            self._update_param(argv, self.param.seq_length_config, cfg.seq_length)
            self._update_param(argv, self.param.max_position_embeddings_config, cfg.seq_length)

        if self.param.global_batch_size_config:
            self._update_param(argv, self.param.global_batch_size_config, cfg.global_batch_size)

        if self.param.micro_batch_size_config:
            self._update_param(argv, self.param.micro_batch_size_config, cfg.micro_batch_size)

    def _modify_parallel_argv(self, argv: List[str], cfg: SearchConfig):
        if self.param.tensor_model_parallel_size_config:
            self._update_param(argv, self.param.tensor_model_parallel_size_config, cfg.tensor_model_parallel_size)

        if self.param.context_parallel_size_config:
            self._update_param(argv, self.param.context_parallel_size_config, cfg.context_parallel_size)

        if self.param.pipeline_model_parallel_size_config:
            self._update_param(argv, self.param.pipeline_model_parallel_size_config, cfg.pipeline_model_parallel_size)

        if self.param.num_layers_per_virtual_pipeline_stage_config:
            self._update_param(argv, self.param.num_layers_per_virtual_pipeline_stage_config, cfg.num_layers_per_virtual_pipeline_stage)

        if self.param.expert_model_parallel_size_config:
            self._update_param(argv, self.param.expert_model_parallel_size_config, cfg.expert_model_parallel_size)

    def _modify_feature_argv(self, argv: List[str], cfg: SearchConfig):
        if self.param.untie_embeddings_and_output_weights_config:
            self._update_flag(argv, self.param.untie_embeddings_and_output_weights_config, cfg.untie_embeddings_and_output_weights)

        if self.param.recompute_granularity_config and \
                self.param.recompute_method_config and \
                self.param.recompute_num_layers_config:
            if cfg.recompute_granularity and \
                    cfg.recompute_method and \
                    cfg.recompute_num_layers:
                self._update_param(argv, self.param.recompute_granularity_config, cfg.recompute_granularity)
                self._update_param(argv, self.param.recompute_method_config, cfg.recompute_method)
                self._update_param(argv, self.param.recompute_num_layers_config, cfg.recompute_num_layers)
            else:
                self._update_param(argv, self.param.recompute_granularity_config, None)
                self._update_param(argv, self.param.recompute_method_config, None)
                self._update_param(argv, self.param.recompute_num_layers_config, None)

        if self.param.use_distributed_optimizer_config:
            self._update_flag(argv, self.param.use_distributed_optimizer_config, cfg.use_distributed_optimizer)

        if self.param.use_ascend_mc2_config:
            self._update_flag(argv, self.param.use_ascend_mc2_config, cfg.use_ascend_mc2)

        if self.param.moe_tp_extend_ep_config:
            self._update_flag(argv, self.param.moe_tp_extend_ep_config, cfg.moe_tp_extend_ep)
            if not cfg.moe_tp_extend_ep:
                if cfg.moe_token_dispatcher_type == "alltoall" and self.param.moe_alltoall_overlap_comm_config:
                    self._update_flag(argv, self.param.moe_alltoall_overlap_comm_config, False)
                if cfg.moe_token_dispatcher_type == "allgather" and self.param.moe_allgather_overlap_comm_config:
                    self._update_flag(argv, self.param.moe_allgather_overlap_comm_config, False)

        if self.param.enable_token_rearrange_opt_config:
            self._update_flag(argv, self.param.enable_token_rearrange_opt_config, cfg.enable_token_rearrange_opt)

        if self.param.dist_train_config:
            self._update_flag(argv, self.param.dist_train_config, cfg.dist_train)


    def _modify_profile_argv(self, argv: List[str], cfg: SearchConfig):
        from mindspeed.auto_settings.config.system_config import get_system_config

        if self.param.train_iters_config:
            self._update_param(argv, self.param.train_iters_config, cfg.train_iters)

        if self.param.profile_config:
            self._update_flag(argv, self.param.profile_config, cfg.profile)

        if self.param.profile_step_start_config:
            self._update_param(argv, self.param.profile_step_start_config, cfg.profile_step_start)

        if self.param.profile_step_end_config:
            self._update_param(argv, self.param.profile_step_end_config, cfg.profile_step_end)

        if self.param.profile_ranks_config:
            argv.append(self.param.profile_ranks_config)
            argv.extend([str(i) for i in range(get_system_config().world_size)])

        if self.param.profile_level_config:
            self._update_param(argv, self.param.profile_level_config, cfg.profile_level)

        if self.param.profile_with_cpu_config:
            self._update_flag(argv, self.param.profile_with_cpu_config, cfg.profile_with_cpu)

        if self.param.profile_with_stack_config:
            self._update_flag(argv, self.param.profile_with_stack_config, cfg.profile_with_stack)

        if self.param.profile_with_memory_config:
            self._update_flag(argv, self.param.profile_with_memory_config, cfg.profile_with_memory)

        if self.param.profile_record_shapes_config:
            self._update_flag(argv, self.param.profile_record_shapes_config, cfg.profile_record_shapes)

    def _modify_multi_model_argv(self, argv: List[str], cfg: SearchConfig):
        if self.param.mm_model_config:
            self._update_param(argv, self.param.mm_model_config, cfg.mm_model)

        if self.param.mm_data_config:
            self._update_param(argv, self.param.mm_data_config, cfg.mm_data)

        if self.param.mm_tool_config:
            self._update_param(argv, self.param.mm_tool_config, cfg.mm_tool)
