from typing import Dict, Optional, Tuple
import os
import stat
from enum import Enum

import pickle

from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.utils.runner.irunner import _Env, IRunner


_Argv = Dict[str, Optional[str]]


class ExecutorFlag(Enum):
    RUN = 0
    PARSE_ARGS = 1
    PARSE_MODEL = 2
    PROFILE = 3


class ModelExecutor:
    """
    Execute the model with different configs.
    """
    MODIFIED_ARGV_FILENAME = "auto_tuning_modified_argv.json"
    PARSE_ARGS_ENV = "OOTB_OPTIMIZER_PARSE_ARGS"
    PARSE_MODEL_ENV = "OOTB_OPTIMIZER_PARSE_MODEL"
    PROFILING_ENV = "OOTB_OPTIMIZER_PROFILING"
    MODIFIED_ARGV_PATH_ENV = "OOTB_OPTIMIZER_MODIFIED_ARGV_PATH"
    ENABLED_ENV_MARKER = "TRUE"

    def __init__(self,
                 runner: IRunner,
                 num_layers_config="--num-layers",
                 num_experts_config="--num-experts",
                 seq_length_config="--seq-length",
                 max_position_embeddings_config="--max-position-embeddings",
                 micro_batch_size_config="--micro-batch-size",
                 global_batch_size_config="--global-batch-size",
                 recompute_granularity_config="--recompute-granularity",
                 recompute_method_config="--recompute-method",
                 recompute_num_layers_config="--recompute-num-layers",
                 adaptive_recompute_device_swap_config="--adaptive-recompute-device-swap",
                 enable_token_rearrange_opt_config="--enable-token-rearrange-opt",
                 tensor_model_parallel_size_config="--tensor-model-parallel-size",
                 pipeline_model_parallel_size_config="--pipeline-model-parallel-size",
                 num_layers_per_virtual_pipeline_stage_config="--num-layers-per-virtual-pipeline-stage",
                 expert_model_parallel_size_config="--expert-model-parallel-size",
                 context_parallel_size_config="--context-parallel-size",
                 use_distributed_optimizer_config="--use-distributed-optimizer",
                 use_ascend_mc2_config="--use-ascend-mc2",
                 train_iters_config="--train-iters",
                 profile_config="--profile",
                 profile_step_start_config="--profile-step-start",
                 profile_step_end_config="--profile-step-end",
                 profile_ranks_config="--profile-ranks",
                 profile_level_config="--profile-level",
                 profile_with_cpu_config="--profile-with-cpu",
                 profile_with_stack_config="--profile-with-stack",
                 profile_with_memory_config="--profile-with-memory",
                 profile_record_shapes_config="--profile-record-shapes",
                 profile_save_path_config="--profile-save-path"
                 ) -> None:
        self.runner = runner
        self.num_layers_config = num_layers_config
        self.num_experts_config = num_experts_config
        self.seq_length_config = seq_length_config
        self.max_position_embeddings_config = max_position_embeddings_config
        self.micro_batch_size_config = micro_batch_size_config
        self.global_batch_size_config = global_batch_size_config
        self.recompute_granularity_config = recompute_granularity_config
        self.recompute_method_config = recompute_method_config
        self.recompute_num_layers_config = recompute_num_layers_config
        self.adaptive_recompute_device_swap_config = adaptive_recompute_device_swap_config
        self.enable_token_rearrange_opt_config = enable_token_rearrange_opt_config
        self.tensor_model_parallel_size_config = tensor_model_parallel_size_config
        self.pipeline_model_parallel_size_config = pipeline_model_parallel_size_config
        self.num_layers_per_virutal_pipeline_stage_config = num_layers_per_virtual_pipeline_stage_config
        self.expert_model_parallel_size_config = expert_model_parallel_size_config
        self.context_parallel_size_config = context_parallel_size_config
        self.use_distributed_optimizer_config = use_distributed_optimizer_config
        self.use_ascend_mc2_config = use_ascend_mc2_config
        self.train_iters_config = train_iters_config
        self.profile_config = profile_config
        self.profile_step_start_config = profile_step_start_config
        self.profile_step_end_config = profile_step_end_config
        self.profile_ranks_config = profile_ranks_config
        self.profile_level_config = profile_level_config
        self.profile_with_cpu_config = profile_with_cpu_config
        self.profile_with_stack_config = profile_with_stack_config
        self.profile_with_memory_config = profile_with_memory_config
        self.profile_record_shapes_config = profile_record_shapes_config
        self.profile_save_path_config = profile_save_path_config

    def execute(self,
                working_dir: str,
                output_filename: str = str(),
                cfg: Optional[SearchConfig] = None,
                flag: ExecutorFlag = ExecutorFlag.RUN
                ) -> int:
        env = self.runner.get_base_env()
        self._prepare_envvars(env, flag)

        modified_argv_path = os.path.join(working_dir, self.MODIFIED_ARGV_FILENAME)

        self._prepare_modified_argv_envvars(env, modified_argv_path)

        modified_argv = self._prepare_modified_argv(cfg, working_dir, output_filename, flag)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(modified_argv_path, flags, mode=mode), 'wb') as f:
            pickle.dump(modified_argv, f)

        returncode = self.runner.run(env)

        return returncode

    def _prepare_envvars(self,
                         env: _Env,
                         flag: ExecutorFlag
                         ) -> _Env:
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

    def _prepare_modified_argv_envvars(self,
                                       env: _Env,
                                       modified_argv_path: str
                                       ) -> _Env:
        env.update({self.MODIFIED_ARGV_PATH_ENV: modified_argv_path})

        return env

    def _prepare_modified_argv(
        self,
        cfg: Optional[SearchConfig],
        working_dir: str,
        output_filename: str,
        flag: ExecutorFlag
    ) -> Tuple[_Argv, _Argv]:
        enabled_argv: _Argv = dict()
        disabled_argv: _Argv = dict()
        if cfg:
            cfg.normalize()

            def _modify_model_argv():
                if self.recompute_granularity_config and self.recompute_method_config and self.recompute_num_layers_config:
                    if cfg.is_full_recompute():
                        enabled_argv.update({self.recompute_granularity_config: cfg.recompute_granularity})
                        enabled_argv.update({self.recompute_method_config: cfg.recompute_method})
                        enabled_argv.update({self.recompute_num_layers_config: str(cfg.recompute_num_layers)})
                    else:
                        disabled_argv.update({self.recompute_granularity_config: str()})
                        disabled_argv.update({self.recompute_method_config: str()})
                        disabled_argv.update({self.recompute_num_layers_config: str()})

                if self.num_layers_config:
                    enabled_argv.update({self.num_layers_config: str(cfg.num_layers)})

                if self.num_experts_config:
                    if cfg.num_experts:
                        enabled_argv.update({self.num_experts_config: str(cfg.num_experts)})
                    else:
                        disabled_argv.update({self.num_experts_config: str()})

                if self.seq_length_config:
                    enabled_argv.update({self.seq_length_config: str(cfg.seq_length)})
                    enabled_argv.update({self.max_position_embeddings_config: str(cfg.seq_length)})

                if self.micro_batch_size_config:
                    enabled_argv.update({self.micro_batch_size_config: str(cfg.micro_batch_size)})

                if self.global_batch_size_config:
                    enabled_argv.update({self.global_batch_size_config: str(cfg.global_batch_size)})

                if self.adaptive_recompute_device_swap_config:
                    if cfg.adaptive_recompute_device_swap:
                        enabled_argv.update({self.adaptive_recompute_device_swap_config: None})
                    else:
                        disabled_argv.update({self.adaptive_recompute_device_swap_config: None})

                if self.enable_token_rearrange_opt_config:
                    if cfg.enable_token_rearrange_opt:
                        enabled_argv.update({self.enable_token_rearrange_opt_config: None})
                    else:
                        disabled_argv.update({self.enable_token_rearrange_opt_config: None})

                if self.use_ascend_mc2_config:
                    if cfg.use_ascend_mc2:
                        enabled_argv.update({self.use_ascend_mc2_config: None})
                    else:
                        disabled_argv.update({self.use_ascend_mc2_config: None})

            def _modify_parallel_argv():
                if self.tensor_model_parallel_size_config:
                    enabled_argv.update({self.tensor_model_parallel_size_config: str(cfg.tensor_model_parallel_size)})

                if self.pipeline_model_parallel_size_config:
                    enabled_argv.update({self.pipeline_model_parallel_size_config: str(cfg.pipeline_model_parallel_size)})

                if self.num_layers_per_virutal_pipeline_stage_config:
                    if cfg.num_layers_per_virtual_pipeline_stage:
                        enabled_argv.update({self.num_layers_per_virutal_pipeline_stage_config:
                                            str(cfg.num_layers_per_virtual_pipeline_stage)})
                    else:
                        disabled_argv.update({self.num_layers_per_virutal_pipeline_stage_config: str()})

                if self.expert_model_parallel_size_config:
                    if cfg.expert_model_parallel_size:
                        enabled_argv.update({self.expert_model_parallel_size_config: str(cfg.expert_model_parallel_size)})
                    else:
                        disabled_argv.update({self.expert_model_parallel_size_config: str()})

                if self.context_parallel_size_config:
                    enabled_argv.update({self.context_parallel_size_config: str(cfg.context_parallel_size)})

                if self.use_distributed_optimizer_config:
                    if cfg.use_distributed_optimizer:
                        enabled_argv.update({self.use_distributed_optimizer_config: None})
                    else:
                        disabled_argv.update({self.use_distributed_optimizer_config: None})

            def _modify_profile_argv():
                if cfg.profile:
                    enabled_argv.update({self.train_iters_config: str(cfg.train_iters)})
                    enabled_argv.update({self.profile_config: None})
                    enabled_argv.update({self.profile_step_start_config: str(cfg.profile_step_start)})
                    enabled_argv.update({self.profile_step_end_config: str(cfg.profile_step_end)})
                    enabled_argv.update({self.profile_ranks_config: str(cfg.profile_ranks)})
                    enabled_argv.update({self.profile_level_config: cfg.profile_level})
                if cfg.profile_with_cpu:
                    enabled_argv.update({self.profile_with_cpu_config: None})
                else:
                    disabled_argv.update({self.profile_with_cpu_config: None})
                if cfg.profile_with_stack:
                    enabled_argv.update({self.profile_with_stack_config: None})
                else:
                    disabled_argv.update({self.profile_with_stack_config: None})
                if cfg.profile_with_memory:
                    enabled_argv.update({self.profile_with_memory_config: None})
                else:
                    enabled_argv.update({self.profile_with_memory_config: None})
                if cfg.profile_record_shapes:
                    enabled_argv.update({self.profile_record_shapes_config: None})
                else:
                    disabled_argv.update({self.profile_record_shapes_config: None})

            _modify_model_argv()
            _modify_parallel_argv()
            _modify_profile_argv()

        if flag == ExecutorFlag.PARSE_ARGS:
            enabled_argv.update({self.profile_save_path_config: working_dir})
        elif flag == ExecutorFlag.PARSE_MODEL or flag == ExecutorFlag.PROFILE:
            enabled_argv.update({self.profile_save_path_config: os.path.join(working_dir, output_filename)})

        return enabled_argv, disabled_argv
