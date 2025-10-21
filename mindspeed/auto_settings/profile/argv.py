"""
定义torchrun的运行参数
"""
from typing import List, Optional, Any

from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.utils.env_utils import update_param, update_flag


class BaseArgv(object):

    @staticmethod
    def base_argv(argv, save_path):
        """
        基础配置
        """
        update_flag(argv, "--auto-settings", False)
        update_param(argv, "--profile-save-path", save_path)


class FilterArgv(object):
    @staticmethod
    def filter_argv(argv):
        """
        过滤部分参数
        """
        filter_param = [
            "--ampipe-degree",
            "--pipe-experts-multi-data"
        ]
        for f in filter_param:
            update_param(argv, f, None)
        filter_flag = [
            "--swap-attention",
            "--ampipe-tp-sp-comm-overlap",
            "--use-pipe-experts",
            "--pipe-experts-multi-stream",
            "--recompute-in-advance",
            "--recompute-in-bubble",
            "--use-nanopipe",
            "--save",
            "--load"
        ]
        for f in filter_flag:
            update_flag(argv, f, False)


class SearchConfigArgv(object):
    adaptive_recompute_device_swap_config: str = "--adaptive-recompute-device-swap"
    enable_token_rearrange_opt_config: str = "--enable-token-rearrange-opt"
    profile_level_config: str = "--profile-level"

    @staticmethod
    def update_argv(argv, config):
        """
        根据config配置，获取profile的参数
        """
        if config:
            SearchConfigArgv._update_model_argv(argv, config)
            SearchConfigArgv._update_parallel_argv(argv, config)
            SearchConfigArgv._update_feature_argv(argv, config)
            SearchConfigArgv._update_profile_argv(argv, config)
        else:
            config = SearchConfig(
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                pipeline_model_parallel_size=1,
                num_layers_per_virtual_pipeline_stage=None,
                expert_model_parallel_size=1
            )
            SearchConfigArgv._update_parallel_argv(argv, config)

    @staticmethod
    def _update_model_argv(argv: List[str], cfg: SearchConfig):
        """
        更新模型参数
        """
        update_param(argv, "--num-layers", cfg.num_layers)
        update_param(argv, "--num-experts", cfg.num_experts)
        update_param(argv, "--seq-length", cfg.seq_length)
        update_param(argv, "--max-position-embeddings", cfg.seq_length)
        update_param(argv, "--global-batch-size", cfg.global_batch_size)
        update_param(argv, "--micro-batch-size", cfg.micro_batch_size)
        update_param(argv, "--make-vocab-size-divisible-by", cfg.make_vocab_size_divisible_by)
        if cfg.lr_warmup_iters:
            update_flag(argv, "--lr-warmup-iters", cfg.lr_warmup_iters)

    @staticmethod
    def _update_parallel_argv(argv: List[str], cfg: SearchConfig):
        """
        更新并行参数
        """
        update_param(argv, "--tensor-model-parallel-size", cfg.tensor_model_parallel_size)
        update_param(argv, "--context-parallel-size", cfg.context_parallel_size)
        update_param(argv, "--pipeline-model-parallel-size", cfg.pipeline_model_parallel_size)
        update_param(argv, "--num-layers-per-virtual-pipeline-stage",
                     cfg.num_layers_per_virtual_pipeline_stage)
        update_param(argv, "--expert-model-parallel-size", cfg.expert_model_parallel_size)
        update_flag(argv, "--sequence-parallel", cfg.sequence_parallel)
        if cfg.ulysses_size:
            update_param(argv, "--context-parallel-size", cfg.ulysses_size)
            update_param(argv, "--context-parallel-algo", "ulysses_cp_algo")
        if cfg.ring_attention_size:
            update_param(argv, "--context-parallel-size", cfg.ring_attention_size)
            update_param(argv, "--context-parallel-algo", "megatron_cp_algo")

    @staticmethod
    def _update_feature_argv(argv: List[str], cfg: SearchConfig):
        """
        更新特性参数
        """
        update_flag(argv, "--untie-embeddings-and-output-weights",
                    cfg.untie_embeddings_and_output_weights)

        if cfg.recompute_granularity and cfg.recompute_method and cfg.recompute_num_layers:
            update_param(argv, "--recompute-granularity", cfg.recompute_granularity)
            update_param(argv, "--recompute-method", cfg.recompute_method)
            update_param(argv, "--recompute-num-layers", cfg.recompute_num_layers)
        else:
            update_param(argv, "--recompute-granularity", None)
            update_param(argv, "--recompute-method", None)
            update_param(argv, "--recompute-num-layers", None)

        update_flag(argv, "--use-distributed-optimizer", cfg.use_distributed_optimizer)

        update_flag(argv, "--use-ascend-mc2", cfg.use_ascend_mc2)

        update_param(argv, "--noop-layers", cfg.noop_layers)

        update_flag(argv, "--moe-tp-extend-ep", cfg.moe_tp_extend_ep)
        if not cfg.moe_tp_extend_ep:
            if cfg.moe_token_dispatcher_type == "alltoall" and "--moe-alltoall-overlap-comm":
                update_flag(argv, "--moe-alltoall-overlap-comm", False)
            if cfg.moe_token_dispatcher_type == "allgather" and "--moe-allgather-overlap-comm":
                update_flag(argv, "--moe-allgather-overlap-comm", False)
        if cfg.use_ascend_mc2:
            # adaptor for te support v2
            # The 'mc2' feature requires sequence parallelism to be enabled
            update_flag(argv, "--sequence-parallel", True)

        update_flag(argv, "--enable-token-rearrange-opt", cfg.enable_token_rearrange_opt)

    def _update_profile_argv(argv: List[str], cfg: SearchConfig):
        """
        更新profile相关参数
        """
        system_config = get_system_config()
        update_param(argv, "--train-iters", cfg.train_iters)
        update_flag(argv, "--profile", cfg.profile)
        update_param(argv, "--profile-step-start", cfg.profile_step_start)
        update_param(argv, "--profile-step-end", cfg.profile_step_end)
        if cfg.auto_settings_ranks is None:
            argv.append("--profile-ranks")
            argv.extend([str(i) for i in range(system_config.world_size)])
        else:
            update_param(argv, "--profile-ranks", cfg.auto_settings_ranks)
        update_param(argv, "--profile-level", cfg.profile_level)
        update_flag(argv, "--profile-with-cpu", cfg.profile_with_cpu)
        update_flag(argv, "--profile-with-stack", cfg.profile_with_stack)
        update_flag(argv, "--profile-with-memory", cfg.profile_with_memory)
        update_flag(argv, "--profile-record-shapes", cfg.profile_record_shapes)
        update_param(argv, "--prof-file", cfg.prof_file)
