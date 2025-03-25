"""Handle cli arguments by features granularity.

Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser, Namespace
from typing import List
from functools import wraps

import warnings
from mindspeed.features_manager import FEATURES
from .arguments import process_args

_ARGS = None


def extra_args_provider_decorator(extra_args_provider):
    """Make a extra args parser  for magatron."""
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        for feature in FEATURES:
            feature.register_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    """Decorate parse_args function of megatron."""

    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def core_transformer_config_from_args_wrapper(fn):
    """A decorator for transformer config args."""
    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        config.context_parallel_algo = args.context_parallel_algo
        config.batch_p2p_comm = False
        if args.use_multiparameter_pipeline_model_parallel:
            config.deallocate_pipeline_outputs = False
        return config

    return wrapper


def get_mindspeed_args() -> Namespace:
    """Get cli arguments of mindspeed."""
    global _ARGS

    if not _ARGS:
        parser = ArgumentParser(
            description="MindSpeed Arguments",
            allow_abbrev=False,
        )
        parser = process_args(parser)
        for feature in FEATURES:
            feature.register_args(parser)
        _ARGS, unknown = parser.parse_known_args()
        parse_unknown_args(_ARGS, unknown)

    return _ARGS


def add_args(args, key, value):
    """Add args to parser."""
    if key is not None:
        key = key[2:].replace("-", "_")
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parse_unknown_args(args: Namespace, unknown: List[str]):
    """Parse special unknown args.

    Args:
        args (Namespace): regular arguments.
        unknown (List[str]): special arguments string.
    """
    i = 0
    key, value = None, None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def validate_args_wrapper(validate_args):
    """A decorator for megatron arguments validation function."""

    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
        # make prev validation and copy some args.
        origin = _pre_validate(args)
        for feature in FEATURES:
            feature.pre_validate_args(args)

        # make megatron args validation then restore args thar are copied.
        args = validate_args(args, defaults)
        for feature in FEATURES:
            feature.validate_args(args=args)

        # make post validation after megatron validation.
        _post_validate(args, origin)
        for feature in FEATURES:
            feature.post_validate_args(args=args)
        from megatron.training.arguments import _print_args

        # _print_args is patched, so it has three arguments.
        _print_args("arguments", args, True)

        return args

    return wrapper


def _pre_validate(args: Namespace) -> tuple:
    replace_model_type_for_deepspeed_moe = False
    if args.num_experts:
        if args.use_ascend_coc:
            raise AssertionError("coc is not compatible with moe models")
        if args.use_ascend_mc2:
            raise AssertionError("mc2 is not compatible with moe models")
        if args.use_legacy_models:
            if args.moe_model_type == "megatron_moe":
                raise AssertionError(
                    "megatron_moe is not compatible with --use-legacy-models"
                )
            replace_model_type_for_deepspeed_moe = True
        else:
            if args.moe_model_type == "deepspeed_moe":
                raise AssertionError(
                    "deepspeed_moe only support with --use-legacy-models"
                )

    # validate optimizer
    if args.optimizer_selection == "fused_adamw":
        print(
            "[WARNING] The default AdamW optimizer "
            "is no longer recommended for new edition, "
            "Use the torch fused AdamW optimizer by "
            "argument --optimizer-selection fused_torch_adamw"
        )
    elif args.optimizer_selection == "fused_ema_adamw":
        if args.reuse_fp32_param:
            raise AssertionError(
                "fused_ema_adamw optimizer "
                "is not compatible with reuse_fp32_param"  # type: ignore
            )

    # validate mla
    if args.multi_head_latent_attention:
        if args.kv_lora_rank is None:
            raise AssertionError(
                "The parameter kv-lora-rank should be set "
                "when use multi_head_latent_attention."
            )
        elif args.v_head_dim is None:
            raise AssertionError(
                "The parameter v-head-dim should be set "
                "when use multi_head_latent_attention."
            )
        elif args.qk_rope_head_dim is None:
            raise AssertionError(
                "The parameter qk-rope-head-dim should be set "
                "when use multi_head_latent_attention."
            )
        elif args.qk_nope_head_dim is None:
            raise AssertionError(
                "The parameter qk-nope-head-dim should be set "
                "when use multi_head_latent_attention."
            )

    # validate yarn
    if args.rope_scaling_type == "yarn":
        if args.rope_scaling_original_max_position_embeddings is None:
            raise AssertionError(
                "The parameter rope_scaling_original_max_position_embeddings "
                "should be set when use yarn."
            )

    # alibi type [2, 3] is only support FA2
    if args.alibi_fusion_attn_type in [2, 3]:
        args.use_fusion_attn_v2 = True
    if args.use_fusion_attn_v2:
        args.use_flash_attn = True
        print(
            '[WARNING] "use_fusion_attn_v2" is not recommended. '
            "This feature is not officially released."
        )

    # for vpp assert pp should > 2
    flag_num_layers_per_virtual_pipeline_stage = None
    flag_overlap_p2p_comm = False
    if (
        args.num_layers_per_virtual_pipeline_stage is not None
        and args.pipeline_model_parallel_size == 2
    ):
        flag_num_layers_per_virtual_pipeline_stage = (
            args.num_layers_per_virtual_pipeline_stage
        )
        args.num_layers_per_virtual_pipeline_stage = None
        if args.overlap_p2p_comm:
            flag_overlap_p2p_comm = True

    # skip validation for deepspeed_moe with CP
    origin_use_legacy_models = args.use_legacy_models
    if replace_model_type_for_deepspeed_moe:
        args.use_legacy_models = False
    origin_context_parallel_size = args.context_parallel_size
    args.context_parallel_size = 1
    original_variable_seq_lengths = args.variable_seq_lengths
    return (
        replace_model_type_for_deepspeed_moe,
        flag_num_layers_per_virtual_pipeline_stage,
        flag_overlap_p2p_comm,
        origin_use_legacy_models,
        origin_context_parallel_size,
        original_variable_seq_lengths,
    )


def _post_validate(args: Namespace, origin: tuple):
    (
        replace_model_type_for_deepspeed_moe,
        flag_num_layers_per_virtual_pipeline_stage,
        flag_overlap_p2p_comm,
        origin_use_legacy_models,
        origin_context_parallel_size,
        original_variable_seq_lengths,
    ) = origin
    args.variable_seq_lengths = original_variable_seq_lengths
    args.context_parallel_size = origin_context_parallel_size

    encoder_model_parallel_size = (
        args.encoder_tensor_model_parallel_size
        * args.encoder_pipeline_model_parallel_size
        * args.context_parallel_size
    )
    decoder_model_parallel_size = (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.context_parallel_size
    )
    total_model_parallel_size = (
        encoder_model_parallel_size + decoder_model_parallel_size
    )
    # Total model size.
    assert (
        args.world_size % total_model_parallel_size == 0
    ), f"world size ({args.world_size}) is not divisible"
    f"by total_model_parallel_size "
    f"({encoder_model_parallel_size=} + {decoder_model_parallel_size=})"

    args.data_parallel_size = args.world_size // total_model_parallel_size
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:

            print(
                f"Resetting global batch size to {args.global_batch_size}",
                flush=True,
            )
    if (
        args.optimize_vpp_send_recv_comm
        and args.num_layers_per_virtual_pipeline_stage is None
    ):
        raise AssertionError(
            "--optimize-vpp-send-recv-comm "
            "can only be used with pipeline with interleaving."
        )

    if replace_model_type_for_deepspeed_moe:
        args.use_legacy_models = origin_use_legacy_models
    if args.enable_zero3:
        print("[WARNING] zero3 currently does not support model save and load")
        if (
            args.use_ascend_mc2
            or args.reuse_fp32_param
            or args.recompute_granularity is not None
            or args.use_pipe_experts
        ):
            raise AssertionError(
                "zero3 cannot be used together with MC2(--use-ascend-mc2), "
                "parameter copy reuse(--reuse-fp32-param),"
                "recompute(--recompute-granularity)"
                "and pipe_experts(use-pipe-experts)"
            )

    # for vpp assert pp should > 2
    if (
        flag_num_layers_per_virtual_pipeline_stage is not None
        and args.pipeline_model_parallel_size == 2
    ):
        args.num_layers_per_virtual_pipeline_stage = (
            flag_num_layers_per_virtual_pipeline_stage
        )
        args.overlap_p2p_comm = flag_overlap_p2p_comm
        if args.num_layers_per_virtual_pipeline_stage is not None:
            assert (
                args.num_layers % args.transformer_pipeline_model_parallel_size == 0
            ), "number of layers should be divisible by the pipeline parallel size"
            num_layers_per_pipeline_stage = (
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
            assert (
                num_layers_per_pipeline_stage
                % args.num_layers_per_virtual_pipeline_stage
                == 0
            ), "number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage"
            args.virtual_pipeline_model_parallel_size = (
                num_layers_per_pipeline_stage
                // args.num_layers_per_virtual_pipeline_stage
            )

    # num_layers_per_virtual_pipeline_stage should be meaningful
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = (
            args.num_layers // args.pipeline_model_parallel_size
        )
        assert (
            num_layers_per_pipeline_stage // args.num_layers_per_virtual_pipeline_stage
            > 1
        ), "considering args of num_layers and pipeline_model_parallel_size, vpp setting should be meaningful"

    # deepspeed dropless does not support pp
    if args.moe_no_drop and args.pipeline_model_parallel_size > 1:
        raise AssertionError("--moe-no-drop is not compatible with pp")

    if args.param_and_grad_buffer_pad and args.param_and_grad_buffer_pad <= 0:
        raise AssertionError("--param-and-grad-buffer-pad must be greater than 0")

    if args.use_fused_rmsnorm:
        if args.normalization != "RMSNorm":
            raise AssertionError(
                "--use-fused-rmsnorm must enable with "
                "--normalization=RMSNorm, but got normalization"
                "={}.".format(args.normalization)
            )
        if args.use_nd_matmul:
            raise AssertionError("ND_MatMul is not compatible with fused_rmsnorm.")
    if args.use_fused_swiglu:
        if not args.swiglu:
            raise AssertionError(
                "--use-fused-swiglu must enable with --swiglu, "
                "but --swiglu={}.".format(args.swiglu)
            )
    if args.use_fused_rotary_pos_emb:
        if args.position_embedding_type != "rope":
            raise AssertionError(
                "--use-fused-rotary-pos-emb must enable with"
                "--position-embedding-type=rope"
            )
    if args.alibi_fusion_attn_type is not None and args.alibi_fusion_attn_type not in [
        0,
        2,
        3,
    ]:
        raise AssertionError("--alibi-fusion-attn-type only support for `0, 2, 3`")
    if args.reuse_fp32_param and not args.bf16:
        raise AssertionError("--reuse-fp32-param only support for `bf16`")
    if args.use_pipe_experts:
        if args.pipe_experts_multi_data <= 0:
            raise AssertionError("--pipe-experts-multi-data must greater than 0")
        if not args.sequence_parallel and args.pipe_experts_multi_stream:
            raise AssertionError(
                "--pipe-experts-multi-stream can only be used with --sequence-parallel."
            )
        local_experts = args.num_experts // args.expert_model_parallel_size
        if local_experts == 1 and args.pipe_experts_multi_data == 1:
            print(
                "[WARNING] if local_experts = num_experts // expert_model_parallel_size is equal to 1 "
                "and --pipe-experts-multi-data is set to 1, "
                "--use-pipe-experts will be turned off."
            )
            args.use_pipe_experts = False
    if (
        args.moe_alltoall_overlap_comm
        and not args.moe_token_dispatcher_type == "alltoall_seq"
    ):
        raise AssertionError(
            "`--moe-alltoall-overlap-comm` only support with `--moe-token-dispatcher-type alltoall_seq`."
        )

    if (
        args.moe_adaptive_recompute_activation
        and args.moe_token_dispatcher_type == "alltoall_seq"
    ):
        raise AssertionError(
            "`--moe-adaptive-recompute-activation` only support with `--moe-token-dispatcher-type allgather`."
        )

    if (
        args.moe_allgather_overlap_comm
        and not args.moe_token_dispatcher_type == "allgather"
    ):
        raise AssertionError(
            "`--moe-allgather-overlap-comm` only support with `--moe-token-dispatcher-type allgather`."
        )

    if args.moe_alltoall_overlap_comm or args.moe_allgather_overlap_comm:
        if not args.moe_permutation_async_comm:
            raise AssertionError(
                "`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-permutation-async-comm`."
            )
        if not args.moe_grouped_gemm:
            raise AssertionError(
                "`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-grouped-gemm`."
            )
    if (
        not args.moe_tp_extend_ep
        and args.moe_alltoall_overlap_comm
        and args.tensor_model_parallel_size > 1
    ):
        raise AssertionError(
            "`--moe-alltoall-overlap-comm` do not support tp for now. only support with moe_tp_extend_ep when tp > 1."
        )
    if args.moe_tp_extend_ep:
        if (
            args.num_experts
            % (args.tensor_model_parallel_size * args.expert_model_parallel_size)
            != 0
        ):
            raise AssertionError(
                "`--moe-tp-extend-ep` only support when num_experts % ( tp * ep ) == 0"
            )
        if not (args.moe_permutation_async_comm and args.moe_grouped_gemm):
            raise AssertionError(
                "`--moe-tp-extend-ep` needs `--moe-permutation-async-comm` and `--moe-grouped-gemm`."
            )
        if args.moe_expert_capacity_factor is not None:
            raise AssertionError(
                "`--moe-tp-extend-ep` only support when moe_expert_capacity_factor is None."
            )
    if args.moe_zero_memory_num_layers is not None:
        num_layers_per_pipeline_stage = (
            args.num_layers // args.pipeline_model_parallel_size
        )
        if (
            args.moe_zero_memory_num_layers < 0
            or args.moe_zero_memory_num_layers > num_layers_per_pipeline_stage
        ):
            raise AssertionError(
                "`--moe-zero-memory-num-layers` must be between 0 and num layers per pipeline stage"
            )
        if args.moe_zero_memory == "disable":
            raise AssertionError(
                "`--moe-zero-memory` must be enabled when using `--moe-zero-memory-num-layers`"
            )
    if args.moe_zero_memory != "disable" and args.moe_allgather_overlap_comm:
        raise AssertionError(
            "`--moe-zero-memory` do not support `--moe-allgather-overlap-comm` for now."
        )
    if args.moe_dynamic_padding and not args.moe_no_drop:
        raise AssertionError(
            "`--moe-dynamic-padding` only support for `--moe-no-drop`."
        )
    if args.moe_permutation_async_comm and args.moe_model_type != "megatron_moe":
        raise AssertionError(
            "`--moe-permutation-async-comm` only support for megatron core moe."
        )
    if args.moe_bmm_mc2:
        if (
            args.moe_model_type != "megatron_moe"
            or not args.moe_token_dispatcher_type == "alltoall"
        ):
            raise AssertionError(
                "`--moe-bmm-mc2` only support for megatron core moe and dispatcher is alltoall."
            )
        if not args.moe_grouped_gemm:
            raise AssertionError(
                "`--moe-bmm-mc2` only support when `--moe-grouped-gemm` is true."
            )
        if args.moe_tp_extend_ep or args.moe_alltoall_overlap_comm:
            raise AssertionError(
                "`--moe-bmm-mc2` not support with `--moe-tp-extend-ep` and `--moe-alltoall-overlap-comm`."
            )

    if args.context_parallel_size > 1 and args.position_embedding_type == "alibi":
        assert (
            args.context_parallel_algo == "megatron_cp_algo"
        ), f"alibi only support megatron_cp_algo"
    if (
        args.context_parallel_size > 1
        and args.context_parallel_algo == "ulysses_cp_algo"
    ):
        assert (
            args.seq_length % args.context_parallel_size == 0
        ), f"sequence length must be divisible by context_parallel_size"
        head, remainder = divmod(
            args.num_attention_heads,
            args.context_parallel_size * args.tensor_model_parallel_size,
        )
        assert (
            head >= 1 and remainder == 0
        ), f"num_attention_heads must be divisible by context_parallel_size * tensor_model_parallel_size"
        args.use_flash_attn = True
    if (
        args.context_parallel_size > 1
        and args.context_parallel_algo == "megatron_cp_algo"
    ):
        assert (
            args.seq_length % (2 * args.context_parallel_size) == 0
        ), f"sequence length must be divisible by 2 * context_parallel_size"
        if args.position_embedding_type == "alibi":
            assert (
                args.alibi_fusion_attn_type in [2, 3]
                and args.attention_mask_type == "causal"
            ), f"megatron_cp_algo only support alibi type in [2, 3] and attention_mask_type is causal"

        assert (
            args.cp_window_size >= 1
            and args.cp_window_size < args.context_parallel_size
        ), f"cp_window_size should in range [1, context_parallel_size) when using double_ring_attention."
        n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
        assert (
            n_window >= 1 and remainder == 0
        ), f"context parallel size must be divisible by cp_window_size when using double ring attention."
        args.use_flash_attn = True
    if (
        args.context_parallel_size > 1
        and args.context_parallel_algo == "hybrid_cp_algo"
    ):
        assert (
            args.ulysses_degree_in_cp is not None
        ), "--ulysses-degree-in-cp must be specified in hybrid_cp_algo"
        ring_degree, remainder = divmod(
            args.context_parallel_size, args.ulysses_degree_in_cp
        )
        assert (
            ring_degree > 1 and remainder == 0
        ), "--ulysses-degree-in-cp must be devisible by --context-parallel-size"
        args.ring_degree = ring_degree

        head, remainder = divmod(
            args.num_attention_heads,
            args.ulysses_degree_in_cp * args.tensor_model_parallel_size,
        )
        assert (
            head >= 1 and remainder == 0
        ), f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"

        assert (
            args.seq_length % (2 * args.context_parallel_size) == 0
        ), f"sequence length must be divisible by 2 * context_parallel_size in hybrid cp"

        assert (
            args.cp_window_size >= 1 and args.cp_window_size < ring_degree
        ), f"cp_window_size should be in range [1, ring_degree) when using double ring attention with hybrid context parallelism."
        n_window, remainder = divmod(ring_degree, args.cp_window_size)
        assert (
            n_window >= 1 and remainder == 0
        ), f"ring_degree should be divisible by cp_window_size when using double ring with hybrid context parallelism."
        args.use_flash_attn = True

    if (
        args.context_parallel_size > 1
        and args.context_parallel_algo == "adaptive_cp_algo"
    ):
        assert (
            args.seq_length % args.context_parallel_size == 0
        ), f"sequence length must be divisible by context_parallel_size"
        args.use_flash_attn = True
    if (
        args.context_parallel_size > 1
        and args.context_parallel_algo == "hybrid_adaptive_cp_algo"
    ):
        assert (
            args.ulysses_degree_in_cp is not None
        ), "--ulysses-degree-in-cp must be specified in hybrid_adaptive_cp_algo"
        ring_degree, remainder = divmod(
            args.context_parallel_size, args.ulysses_degree_in_cp
        )
        assert (
            ring_degree > 1 and remainder == 0
        ), "--ulysses-degree-in-cp must be devisible by --context-parallel-size"
        head, remainder = divmod(
            args.num_attention_heads,
            args.ulysses_degree_in_cp * args.tensor_model_parallel_size,
        )
        assert (
            head >= 1 and remainder == 0
        ), f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"
        assert (
            args.seq_length % args.context_parallel_size == 0
        ), f"sequence length must be divisible by context_parallel_size in hybrid cp"
        args.use_flash_attn = True

    # Mandatory modification to SBH, subsequent abandonment of other formats such as BSH,BSND
    if args.shape_order != "SBH":
        args.shape_order = "SBH"
    if args.transformer_impl == "transformer_engine":
        args.transformer_impl = "local"
    if args.fp8:
        raise AssertionError("NPU not supported FP8.")
    if args.tp_comm_overlap:
        args.tp_comm_overlap = False
    if args.recompute_method == "uniform":
        assert not args.recompute_activation_function, (
            "uniform recomputation is not compatible "
            "with activation function recomputation "
        )
        assert not args.recompute_norm, (
            "uniform recomputation is not compatible " "with norm recomputation "
        )
    if args.recompute_activation_function and args.recompute_granularity == "selective":
        raise AssertionError(
            "--recompute-activation-function is not compatible with selective recomputation"
        )
    adaptive_recompute_enable = (
        args.adaptive_recompute_device_size > 0 or args.adaptive_recompute_device_swap
    )
    if args.recompute_norm and args.recompute_granularity == "selective":
        raise AssertionError(
            "--recompute-norm is not compatible with selective recomputation"
        )
    if args.recompute_norm and args.use_legacy_models:
        raise AssertionError("--recompute-norm is only supported with mcore models")
    if args.use_nanopipe and not args.use_legacy_models:
        raise AssertionError("--use-nanopipe is not available with mcore models")
    if args.adaptive_recompute_device_swap and not args.use_legacy_models:
        raise AssertionError(
            "--adaptive-recompute-device-swap is not available with mcore models"
        )
    if adaptive_recompute_enable:
        assert args.recompute_granularity is None and args.recompute_method is None, (
            "adaptive selective recompute is not compatible with "
            "recompute_granularity and recompute_method. "
        )
        assert not args.recompute_activation_function, (
            "adaptive selective recompute is not compatible "
            "with activation function recomputation "
        )
        assert (
            not args.swap_attention
        ), "adaptive selective recompute is not compatible with swap_attention feature"
        assert not args.recompute_in_advance and not args.recompute_in_bubble, (
            "adaptive selective recompute " "is not compatible with ripipe schedule"
        )
        assert (
            not args.memory_fragmentation
        ), "adaptive selective recompute is not compatible with memory fragmentation"
    if args.memory_fragmentation:
        assert (
            not args.use_fused_rotary_pos_emb
        ), "memory fragmentation is not compatible with use_fused_rotary_pos_emb"
    if args.smart_swap:
        assert (
            not adaptive_recompute_enable
        ), "smart swap is not compatible with adaptive selective recompute"
        assert (
            not args.memory_fragmentation
        ), "smart swap is not compatible with memory fragmentation"
    if args.adaptive_memory_optimization:
        assert (
            args.ampipe_degree <= 1
        ), "adaptive memory optimization is not compatible with ampipe"
        assert (
            not adaptive_recompute_enable
        ), "adaptive memory optimization is not compatible with adaptive recomputing"
        assert (
            args.recompute_granularity is None and args.recompute_method is None
        ), "adaptive memory optimization is not compatible with recompute_granularity or recompute_method"
        assert (
            not args.recompute_activation_function
        ), "adaptive memory optimization is not compatible with recompute_activation_function"
        assert (
            not args.swap_attention
        ), "adaptive memory optimization is not compatible with swap_attention feature"
        assert (
            not args.recompute_in_bubble
        ), "adaptive memory optimization is not compatible with recompute_in_bubble"
        assert (
            not args.memory_fragmentation
        ), "adaptive memory optimization is not compatible with memory_fragmentation"
    if args.use_flash_attn:
        assert (
            args.sparse_mode == 0 or args.sparse_mode == 2
        ), f"Only supports sparse modes 0 and 2"
    args.create_attention_mask_in_dataloader = False
    if args.automated_pipeline:
        if args.recompute_activation_function:
            print(
                "[WARNING] disable activation function recomputation when enabling automated pipeline"
            )
            args.recompute_activation_function = False
        if args.recompute_granularity is not None or args.recompute_method is not None:
            print(
                "[WARNING] disable recompute granularity and recompute method when enabling automated pipeline"
            )
            args.recompute_granularity = None
            args.recompute_method = None
        if args.noop_layers:
            print("[WARNING] disable noop_layers when enabling automated pipeline")
            args.noop_layers = None
    if args.automated_pipeline_perf:
        if args.automated_pipeline:
            print(
                "[WARNING] disable automated pipeline when enabling automated pipeline performance version"
            )
            args.automated_pipeline = False
        if args.num_layers_per_virtual_pipeline_stage is not None:
            raise AssertionError(
                "automated pipeline performance is temporarily incompatible with virtual pipeline"
            )
    if args.use_ascend_mc2:
        if args.use_ascend_coc:
            raise AssertionError("--mc2 and coc can not be used together")
    if args.use_nd_matmul:
        if args.normalization == "LayerNorm":
            raise AssertionError("ND_MatMul is temporarily incompatible with LayerNorm")
        if args.load is not None or args.pretrained_checkpoint is not None:
            raise AssertionError(
                "ND_MatMul does not support loading weights for training temporarily"
            )
        if args.tensor_model_parallel_size % args.nd1_dim1_size != 0:
            raise AssertionError(
                "tensor_model_parallel_size must be divisible by nd1_dim1_size"
            )
        if args.tensor_model_parallel_size % args.nd2_dim1_size != 0:
            raise AssertionError(
                "tensor_model_parallel_size must be divisible by nd2_dim1_size"
            )

    args.reduce_recompute_for_last_chunk = False
    if args.recompute_in_advance:
        args.reduce_recompute_for_last_chunk = True
        if args.recompute_method == "uniform":
            raise AssertionError(
                "recompute_in_advance does not support uniform recompute_method"
            )
        if not args.recompute_num_layers and not args.adaptive_memory_optimization:
            raise AssertionError(
                "recompute_num_layers can not be None or 0 when using recompute_in_advance"
            )
        if (
            args.pipeline_model_parallel_size <= 1
            or args.num_layers_per_virtual_pipeline_stage is None
        ):
            raise AssertionError(
                "recompute_in_advance only support pipelining with interleaving"
            )
        if args.num_layers_per_virtual_pipeline_stage != 1:
            args.recompute_in_advance = False
    if args.recompute_in_bubble:
        if args.recompute_num_layers:
            raise AssertionError(
                "recompute_num_layers must be None or 0 when using recompute_in_bubble"
            )
        if (
            args.pipeline_model_parallel_size <= 1
            or args.num_layers_per_virtual_pipeline_stage is None
        ):
            raise AssertionError(
                "recompute_in_bubble only support pipelining with interleaving"
            )
        if not args.swap_attention:
            # Following is a trick to realize bubble recomputation. We first enable all recomputation,
            # and then disable recomputation for all layers except the ones chosen for bubble recomputation.
            args.recompute_granularity = "full"
            args.recompute_method = "block"
        if args.enable_recompute_layers_per_pp_rank:
            args.recompute_num_layers = (
                args.num_layers // args.pipeline_model_parallel_size
            )
        else:
            args.recompute_num_layers = args.num_layers_per_virtual_pipeline_stage
    if isinstance(args.noop_layers, str):
        noop_layers = set()
        for x in args.noop_layers.split(","):
            if int(x) >= args.num_layers or int(x) < 0:
                raise AssertionError(
                    f"each element in args.noop_layers({args.noop_layers}) should bigger or equal "
                    f"to 0 and smaller than args.num_layers({args.num_layers})"
                )
            noop_layers.add(int(x))
        args.noop_layers = noop_layers

    if args.ampipe_degree > 1:
        assert (
            args.use_flash_attn
        ), "ampipe only supports flash attention, please enable '--use-flash-attn'."
        assert args.num_experts is not None, "ampipe only supports MoE model."
        assert (
            args.expert_model_parallel_size > 1
        ), "ampipe only supports expert_model_parallel_size > 1"
        assert (
            args.moe_model_type == "deepspeed_moe"
        ), "ampipe only supports deepspeed_moe."
        assert not args.use_ascend_mc2, "ampipe does't supports ascend mc2 for now."
        assert not args.add_bias_linear, "ampipe does't supports bias linear for now."
        assert (
            not args.overlap_grad_reduce
        ), "ampipe does't supports overlap_grad_reduce for now."
        assert (
            not args.overlap_param_gather
        ), "ampipe does't supports overlap_param_gather for now."
        assert not args.use_nanopipe, "ampipe does't supports use_nanopipe for now."
        assert (
            not args.recompute_in_bubble
        ), "ampipe does't supports ripipe recompute_in_bubble for now."
        assert (
            not args.recompute_in_advance
        ), "ampipe does't supports ripipe recompute_in_advance for now."
        assert (
            not args.adaptive_recompute_device_swap
        ), "ampipe does't supports ripipe recompute_in_advance for now."
        if args.sequence_parallel:
            assert (
                args.seq_length % (args.ampipe_degree * args.tensor_model_parallel_size)
                == 0
            ), "sequence length must be divisible by ampipe_degree * tensor_model_parallel_size"
        if args.context_parallel_size > 1:
            assert (
                args.context_parallel_algo == "megatron_cp_algo"
            ), "ampipe only supports megatron_cp_algo"
            assert (
                args.ampipe_degree == 2
            ), "ampipe only supports ampipe_degree=2 when context_parallel_size>1"
            slice_size, remainder = divmod(
                args.seq_length, 2 * args.ampipe_degree * args.context_parallel_size
            )
            assert (
                remainder == 0
            ), "sequence length must be divisible by 2 * ampipe_degree * context_parallel_size"
            if args.sequence_parallel:
                assert (
                    slice_size % (args.tensor_model_parallel_size) == 0
                ), "sequence length must be divisible by 2 * ampipe_degree * context_parallel_size * tensor_model_parallel_size"
        if args.use_pipe_experts:
            if args.pipe_experts_multi_data % args.ampipe_degree != 0:
                print(
                    "[WARNING] if pipe_experts_multi_data isn't divisible by ampipe_degree "
                    "--use-pipe-experts will be turned off."
                )
                args.use_pipe_experts = False
                args.pipe_experts_multi_stream = False
                args.pipe_experts_multi_data = 1
    if args.tp_2d:
        if args.sequence_parallel:
            raise AssertionError("2d tp does not support sequence parallel")
        if args.use_fused_rmsnorm:
            raise AssertionError("2d tp does not support fused rmsnorm")
        if args.use_nanopipe:
            raise AssertionError("tp-2d does not support nano-pipe")
        if args.ampipe_degree > 1:
            raise AssertionError("tp-2d does not support ampipe")
        if args.context_parallel_algo not in ["megatron_cp_algo", "ulysses_cp_algo"]:
            raise AssertionError(
                "tp-2d now only support megatron_cp_algo or ulysses_cp_algo"
            )
        if args.use_ascend_coc:
            raise AssertionError("tp-2d does not support ascend coc")
        if args.tensor_model_parallel_size // args.tp_x != args.tp_y:
            raise AssertionError("need satisfy tp = tp_x * tp_y")
        if args.expert_model_parallel_size > 1:
            raise AssertionError("2d tp does not support moe")

    if args.expert_interval <= 0 or args.expert_interval > args.num_layers:
        raise AssertionError("--expert-interval must be between 1 and num layers")
    if args.moe_train_capacity_factor <= 0.0:
        raise AssertionError("--moe-train-capacity-factor must be greater than 0.0")

    if args.gemm_gradient_accumulation_fusion:
        if not args.moe_grouped_gemm:
            raise AssertionError(
                "`--gemm-gradient-accumulation-fusion` only support with `--moe-grouped-gemm`."
            )

    if args.use_legacy_models:
        if args.overlap_param_gather and args.reuse_fp32_param:
            raise AssertionError(
                "In legacy, `overlap_param_gather` does not support `reuse_fp32_param`."
            )

    if args.fp16:
        args.gradient_accumulation_fusion = False
        warnings.warn("Unsupported gradient fp16 bf16 for gradient accumulation fusion")

    if (
        args.context_parallel_size > 1
        and args.reset_attention_mask
        and args.attention_mask_type == "causal"
    ):
        assert (
            args.context_parallel_algo == "megatron_cp_algo"
        ), "accelerated eod reset mode only support ring attention"

    if args.context_parallel_kv_cache_policy:
        if args.context_parallel_size == 1:
            raise AssertionError(
                "context parallel size must larger than 1 when --context-parallel-kv-cache-policy is set."
            )
        if not args.use_flash_attn:
            raise AssertionError(
                "--context-parallel-kv-cache-policy only support use flash attention."
            )

    if args.context_parallel_cache_interval != 0:
        if not args.context_parallel_kv_cache_policy:
            raise AssertionError(
                "--context-parallel-cache-interval only can be used when --context-parallel-kv-cache-policy is set."
            )
        if args.context_parallel_cache_interval >= args.num_layers:
            raise AssertionError(
                "--context-parallel-cache-interval should be smaller than the number of layers."
            )
        if args.context_parallel_cache_interval < 0:
            raise AssertionError(
                "--context-parallel-cache-interval cannot be negative number."
            )

    if args.use_ulysses_allgather_kv:
        if args.context_parallel_size == 1:
            raise AssertionError(
                "context parallel size must larger than 1 when --use-ulysses-allgather-kv is set."
            )
        if args.context_parallel_algo != "ulysses_cp_algo":
            raise AssertionError(
                "--context_parallel-algo should be ulysses_cp_algo when using --use-ulysses-allgather-kv."
            )
        if not args.group_query_attention:
            raise AssertionError(
                "--use-ulysses-allgather-kv needs to enable --group-query-attention."
            )

    if args.save or args.load:
        if args.ckpt_format != "torch":
            raise AssertionError("Only ckpt-format = torch is supported.")

    return args
