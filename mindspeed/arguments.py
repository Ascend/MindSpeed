# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from dataclasses import make_dataclass, field
from functools import wraps
import argparse
import warnings
import torch
from mindspeed.features_manager import FEATURES_LIST


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_network_size_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_training_args(parser)
    parser = _add_data_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_cp_args(parser)
    parser = _add_network_args(parser)
    parser = _add_algorithm_args(parser)
    parser = _add_automated_pipeline_args(parser)
    parser = _add_alibi_args(parser)
    parser = _add_ndmm_args(parser)
    parser = _add_2d_tp_args(parser)
    parser = _add_coc_args(parser)
    parser = _add_profile_args(parser)
    parser = _add_auto_parallel_args(parser)
    parser = _add_deepseek_args(parser)
    parser = _auto_tuning_args(parser)
    parser = _add_auto_parallel_mm_args(parser)
    parser = _add_hccl_group_buffer_args(parser)
    parser = _add_layerzero_args(parser)
    parser = _add_dist_train_args(parser)

    for feature in FEATURES_LIST:
        feature.register_args(parser)

    return parser


def _add_deepseek_args(parser):
    group = parser.add_argument_group(title='deepseek')
    # deepseek moe arguments
    group.add_argument('--n-shared-experts', type=int, default=None)
    # mla arguments
    group.add_argument('--multi-head-latent-attention', action='store_true', default=False,
                       help='Use Multi-head Latent Attention(MLA)')
    group.add_argument('--q-lora-rank', type=int, default=None, help='The low rank of q')
    group.add_argument('--kv-lora-rank', type=int, default=None, help='The low rank of k and v')
    group.add_argument('--v-head-dim', type=int, default=None, help='The head dim of v')
    group.add_argument('--qk-rope-head-dim', type=int, default=None, help='The qk head dim for rope')
    group.add_argument('--qk-nope-head-dim', type=int, default=None, help='The qk head dim for only self-attn')
    # yarn arguments
    group.add_argument('--rope-scaling-type', type=str, default=None, choices=['yarn', ],
                       help='Set the rope scaling type, only support "yarn" type now')
    group.add_argument('--rope-scaling-beta-fast', type=int, default=32, help='Yarn rope: rope beta fast')
    group.add_argument('--rope-scaling-beta-slow', type=int, default=1, help='Yarn rope: rope beta slow')
    group.add_argument('--rope-scaling-factor', type=float, default=1.0, help='Yarn rope: rope factor')
    group.add_argument('--rope-scaling-mscale', type=float, default=1.0, help='Yarn rope: rope mscale')
    group.add_argument('--rope-scaling-mscale-all-dim', type=float, default=0.0, help='Yarn rope: rope mscale all dim')
    group.add_argument('--rope-scaling-original-max-position-embeddings', type=int, default=None,
                       help='Yarn rope: rope original max position embeddings')
    group.add_argument('--moe-hierarchical-alltoallv', action='store_true',
                       help='Reduce communication cost between nodes')

    return parser


def _auto_tuning_args(parser):
    group = parser.add_argument_group(title='auto_tuning')

    group.add_argument('--auto-tuning', action='store_true', help='enable auto tuning')
    group.add_argument('--auto-tuning-work-dir', type=str, default='./auto_tuning_dir',
                       help="auto tuning working path.")
    group.add_argument('--auto-tuning-ranks', type=int, default=16, help='the global size of auto tuning')
    group.add_argument('--auto-tuning-log-level', type=str, default='info', choices=['debug', 'info', 'warning'],
                       help='auto tuning log level, could be debug, info or warning')

    return parser


def _add_profile_args(parser):
    group = parser.add_argument_group(title='profile')
    group.add_argument("--profile-level", type=str, default='level0',
                       choices=['level0', 'level1', 'level2'],
                       help="Profile level default level0.")
    group.add_argument("--profile-with-cpu", action='store_true', default=False,
                       help="Profile with cpu info.")
    group.add_argument("--profile-with-stack", action='store_true', default=False,
                       help="Profile without stack info.")
    group.add_argument("--profile-with-memory", action='store_true', default=False,
                       help="Profile without memory info.")
    group.add_argument("--profile-record-shapes", action='store_true', default=False,
                       help="Profile record shape info.")
    group.add_argument("--profile-save-path", type=str, default='./profile_dir',
                       help="Profile save path.")
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[-1],
                       help='Global ranks to profile.The default value of -1 means to profile all ranks')
    return parser


def _add_coc_args(parser):
    group = parser.add_argument_group(title='coc')
    # ascend mc2 arguments
    group.add_argument("--use-ascend-mc2", action='store_true',
                       help="Use ascend mc2")
    # ascend coc arguments
    group.add_argument("--use-ascend-coc", action='store_true',
                       help="Use ascend coc")
    group.add_argument('--coc-mode', type=int, default=-1,
                       help='coc-mode: 0=original, 1=rewrite, 2=coc default')
    group.add_argument('--coc-parallel-num', type=int, default=1,
                       help='coc parallel num')
    group.add_argument('--coc-fused-kernel', action='store_true',
                       help='use coc fused kernel')
    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title='moe')
    # deepspeed moe arguments
    group.add_argument('--moe-model-type', type=str, default='megatron_moe',
                       choices=['deepspeed_moe', 'megatron_moe'], help='moe model type default megatron moe')
    group.add_argument('--expert-interval', type=int, default=1,
                       help='Use experts in every "expert-interval" layers')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')
    group.add_argument('--noisy-gate-policy', type=str, default=None, choices=['Jitter', 'RSample', 'None'],
                       help="noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.")
    group.add_argument('--enable-token-rearrange-opt', action='store_true',
                       help="Use this flag to enable token rearrange optimize")
    group.add_argument('--no-use-rts',
                       action='store_false', default=False,
                       help='whether to use Random Token Selection.',
                       dest='use_rts')
    group.add_argument("--moe-no-drop", action='store_true',
                       help="Use no drop policy in moe layer, no tokens will be discarded.")
    group.add_argument("--moe-dynamic-padding", action='store_true',
                       help="Reducing AllReduce communication under the no drop policy through the sliding window mechanism.")
    group.add_argument("--moe-use-sinkhorn", action='store_true',
                       help="Use sinkhorn load balancing in the gate.")

    # megatron mcore moe arguments
    group.add_argument("--moe-tp-extend-ep", action='store_true',
                       help="use tp group to extend experts parallelism"
                            "instead of sharding weight tensor of experts in tp group")
    group.add_argument("--moe-permutation-async-comm", action='store_true',
                       help="overlap moe permutation 3 all gather communications")
    group.add_argument("--moe-adaptive-recompute-activation", action='store_true',
                       help="MoE adaptive recompute, avoiding memory imbalance in the early stage.")
    group.add_argument('--moe-adaptive-recompute-activation-scale', type=float, default=2.0,
                       help='MoE adaptive recompute threshold factor.')
    group.add_argument("--use-fused-moe-token-permute-and-unpermute", action='store_true',
                       help="Use fused moe permute and unpermute.")
    group.add_argument("--gemm-gradient-accumulation-fusion", action='store_true',
                       help="Use gradient-accumulation-fusion in gemm.")
    # moe optimization arguments
    group.add_argument('--moe-alltoall-overlap-comm', action='store_true', default=False,
                       help='moe_alltoall_overlap_comm')
    group.add_argument('--moe-allgather-overlap-comm', action='store_true', default=False,
                       help='moe_allgather_overlap_comm')
    group.add_argument('--moe-experts-pipeline-degree', type=int, default=0,
                       help='Group experts into pipeline stages to overlap computation and communication.')
    group.add_argument("--moe-zero-memory", type=str, default='disable',
                       choices=['disable', 'level0', 'level1'],
                       help="Save activation memory in moe layer.")
    group.add_argument('--moe-zero-memory-num-layers', type=int, default=None,
                       help='the number of layers using moe-zero-memory level1'
                            'in each pp stage.')
    group.add_argument('--moe-bmm-mc2', action='store_true', default=False,
                       help='moe_bmm_mc2')
    return parser


def _add_cp_args(parser):
    group = parser.add_argument_group(title='cp parallel')
    group.add_argument('--context-parallel-algo', type=str, default='ulysses_cp_algo',
                       choices=['ulysses_cp_algo', 'megatron_cp_algo', 'hybrid_cp_algo', 'adaptive_cp_algo',
                                'hybrid_adaptive_cp_algo'],
                       help='context parallel algorithm')
    group.add_argument('--ulysses-degree-in-cp', type=int, default=None)
    group.add_argument('--cp-window-size', type=int, default=1)
    group.add_argument('--attention-mask-type', type=str, default='causal',
                       choices=['causal', 'general'], help='context parallel attention mask type')
    group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                       help='use this flag to enable cp send-recv-overlap.')
    group.add_argument("--use-fused-ring-attention-update", action='store_true',
                       help="Use fused ring attention update.")
    group.add_argument("--megatron-cp-in-bnsd", action='store_true',
                       help="Megatron CP in bnsd.")
    group.add_argument('--attention-mask-on-cpu', action='store_true',
                       help='store full attention mask on CPU instead of NPU')
    group.add_argument('--adaptive-cp-without-coarse', action='store_true',
                       help='does not coarse the attention mask in adaptive_cp feature, only recommended when full'
                            'sequence length is less than 8K and dynamic attention mask is not feasible')
    group.add_argument('--adaptive-cp-dynamic-attn-mask', action='store_true',
                       help='if the attention mask is dynamic across batches')
    group.add_argument('--adaptive-cp-only-reschedule', action='store_true',
                       help='not apply remapping but only rescheduling process in adaptive-cp feature')
    group.add_argument('--adaptive-cp-manually-set-mask-list', action='store_true',
                       help='manually set pre-cooked attention mask list')
    group.add_argument('--context-parallel-kv-cache-policy', type=str, default=None,
                       choices=['full', 'half'],
                       help='Selectivity cache K, V in process of cp.'
                            'Default is None, means not used cache K, V.'
                            'If para is full, cache all K, V.'
                            'If para is half, cache only K')
    group.add_argument('--context-parallel-cache-interval', type=int, default=0,
                       help='Set the interval of cache layers in cp.'
                            'Default is 0, means cache K, V in all layers.')
    group.add_argument('--use-ulysses-allgather-kv', action='store_true',
                       help='use this flag to enable allgather kv + repeat all2all q in ulysses cp.')
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'Llama2Tokenizer',
                                'PretrainedFromHF',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    group.add_argument('--param-and-grad-buffer-pad', type=int, default=None,
                       help='Use this argument to ensure that all buckets start at a memory address that is needed-byte. Set 512 for Ascend')
    group.add_argument('--use-nanopipe', action='store_true',
                       default=False, help='use nano pipeline parallelism for reduce bubble.')
    group.add_argument('--use-nanopipe-swap', action='store_true',
                       default=False, help='use nano pipeline parallelism with swap for reduce bubble.')
    group.add_argument('--use-pipe-experts', action='store_true',
                       help='Use this flag to enable pipe moe, overlap all2all and expert')
    group.add_argument('--disable-gloo-group', action='store_true',
                       help='Replace the communication method of the DP group in the distributed optimizer from gloo to hccl.')
    group.add_argument('--hccl-slice-size', type=int, default=10 * 1024 * 1024,
                       help='data slice size on each dp rank in distributed optimizer')
    group.add_argument('--variable-seq-lengths', action='store_true',
                       help='Supports variable sequence lengths across batches/microbatches. Set this if the data '
                            'loader supports variable sequence length generation across batches/microbatches. Because '
                            'of the additional communication overhead incurred during pipeline parallelism, it should '
                            'not be set if the sequence length is constant during training. if sequence length is '
                            'constant during training.')
    return parser


def _add_training_args(parser):

    group = parser.add_argument_group(title='training')

    group.add_argument('--pre-tockens', type=int, default=65536,
                       help='pre-tockens is used by Flash attention')
    group.add_argument('--next-tockens', type=int, default=0,
                       help='next-tockens is used by Flash attention')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND'],
                       help='input shape order used by Flash attention')
    group.add_argument('--sparse-mode', type=int, default=0,
                       help='To improve performance in different modes of attention mask')
    group.add_argument('--adaptive-recompute-device-size',
                       type=int, default=-1,
                       help='The memory size for adaptive selective recompute strategy. '
                            'The default is -1. If this parameter > 0, '
                            'will activate adaptive selective recompute. ')
    group.add_argument('--adaptive-recompute-profiling-step',
                       type=int, default=10,
                       help='The profiling step for adaptive selective recompute strategy. '
                            'The default is 10. If activate adaptive selective recompute, '
                            'will solve graph after step 10. ')
    group.add_argument('--adaptive-recompute-device-swap',
                       action='store_true', default=False,
                       help='switch to open adaptive recompute feature. '
                            'The default is False.')
    group.add_argument('--enable-recompute-layers-per-pp-rank',
                       action='store_true', default=False,
                       help='If enabled, --recompute-num-layers will mean the number of '
                       'layers recomputed in each pp rank. Otherwise it means the number '
                       'of layers recomputed in each vpp rank.')
    group.add_argument('--recompute-activation-function', action='store_true',
                       help='Recompute the activation function in MLP layers.')
    group.add_argument('--recompute-activation-function-num-layers', type=int, default=None,
                       help='Can be used together with "--recompute-method block." '
                       'and "--recompute-num-layers". ')
    group.add_argument('--recompute-norm', action='store_true',
                       help='Recompute norm in Transformer Layers')
    group.add_argument('--recompute-norm-num-layers', type=int, default=None,
                       help='Recompute norm num layers, can be used together with activation function recompute. ')
    group.add_argument('--recompute-in-bubble', action='store_true',
                       help='use bubble to do recompute to reduce memory')
    group.add_argument('--recompute-in-advance', action='store_true',
                       help='recompute early to reduce bubble and improve training.')
    group.add_argument('--jit-compile', action='store_true', default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--swap-attention', action='store_true', default=False,
                       help='switch to open swap-attention feature.'
                            'The default is False.')
    group.add_argument('--swap-modules', type=str, default="input_norm,self_attention,post_attention_norm",
                       help='Swap modules for model. Can be used together with "--swap-attention."')
    group.add_argument('--adaptive-memory-optimization', action='store_true', default=False,
                       help='Switch to open adaptive memory optimization feature, default is False.')
    group.add_argument('--use-fusion-attn-v2', action='store_true', default=False,
                       help='use fusion_attention ops version 2')
    group.add_argument('--pipe-experts-multi-data', type=int, default=1,
                       help='Use multi data to split the input tensor to implement masking when --use-pipe-experts. '
                            'The default is 1.')
    group.add_argument('--pipe-experts-multi-stream', action='store_true', default=False,
                       help='Use multi stream to avoid link collision in collective communication when --use-pipe-experts. '
                            'The default is False.')
    group.add_argument("--additional-config", help="additional model config file path")
    group.add_argument('--use-ema', action='store_true', default=False,
                       help='use ema when training')
    group.add_argument('--use-multiparameter-pipeline-model-parallel', action='store_true', default=False,
                       help='can transfer multi parameters from stage to stage in pipeline model parallel')
    group.add_argument('--ampipe-degree', type=int, default=1,
                       help='Set Attention MoE pipe(AMPipe) degree, 1 means not enable '
                            'AMPipe, greater than 1 means enable this feature.')
    group.add_argument('--ampipe-tp-sp-comm-overlap', action='store_true', default=False,
                       help='enable computation and tp or sp communication overlap in ampipe')
    group.add_argument('--op-cal-tflops', action='store_true', default=False,
                       help='use for cal mfu and hfu')
    group.add_argument('--npu-deterministic', action='store_true', default=False,
                       help='enable deterministic computing for npu')
    group.add_argument('--optimizer-selection', type=str, default='fused_adamw',
                       choices=['fused_adamw', 'fused_torch_adamw', 'fused_ema_adamw'],
                       help='Select from the former fused AdamW optimizer and Torch fused AdamW optimizer')
    group.add_argument('--ema-decay', type=float, default=0.9999,
                       help='Set ema_decay of fused_ema_adamw optimizer.')
    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    group.add_argument("--add-qkv-bias", action="store_true", default=False,
                       help='Configuration for the qkv bias.')
    group.add_argument("--add-dense-bias", action="store_true", default=False,
                       help='Configuration for the dense bias.')
    group.add_argument("--skip-bias-add", action="store_false", default=True,
                       help='Configuration for the skip bias.')
    group.add_argument("--noop-layers", type=str,
                       help='Specity the noop layers.') 
    return parser


def _add_automated_pipeline_args(parser):
    group = parser.add_argument_group(title='automated_pipeline_allocation')
    group.add_argument('--automated-pipeline',
                       action='store_true',
                       help='To enable automated pipeline memory saving process'
                      )
    group.add_argument('--automated-pipeline-perf',
                       action='store_true',
                       help='To enable automated pipeline performance acceleration process'
                       )
    group.add_argument('--save-memory-ratio',
                       type=float, default=0.20,
                       help='To set memory saving rate in automated pipeline'
                       )
    group.add_argument('--num-layer-list',
                       type=str, help='To store the layer policy of automated pipeline'
                       )
    group.add_argument('--recompute-module-list',
                       type=str, help='To store the recompute policy of automated pipeline'
                       )
    group.add_argument('--recompute-type',
                       type=int, default=2,
                       help='To store the recompute type of automated pipeline, 0 for mlp block '
                       '1 for attention block and 2 for transformer layer'
                       )
    group.add_argument('--optimized-mbs-list',
                       type=str,
                       help='To store the optimized mbs policy of automated pipeline performance'
                       )
    group.add_argument('--mbs-idx',
                       type=int,
                       help='To store the index of mbs list'
                       )
    group.add_argument('--pp-schedule-list',
                       type=str,
                       help='To store the pipeline schedule policy of automated pipeline performance'
                       )
    group.add_argument('--optimized-mbs-mode',
                       action='store_false',
                       help='To store the status of optimized mbs in automated pipeline performance'
                       )
    group.add_argument('--memory-fragmentation',
                       action='store_true', default=False,
                       help='Enable the memory fragmentation feature.')
    group.add_argument('--smart-swap',
                       action='store_true', default=False, help='Enable the smart swap feature.')
    return parser


def _add_algorithm_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--optimization-level', type=int, choices=[0, 1, 2], default=2,
                       help='0: The minimum patch set for megatron to adapt to NPU,'
                            '1: Affinity optimization (fusion operator, etc.), '
                            '2: Advanced acceleration algorithm')
    group.add_argument('--reuse-fp32-param', action='store_true',
                       help='The distributed training optimizer frees up '
                            'param copies of FP32 to save memory.')

    group.add_argument('--optimize-send-recv-comm', action='store_true', 
                       help='optimize send_recv communication in pipeline without interleaving.')
    group.add_argument('--optimize-vpp-send-recv-comm', action='store_true', 
                       help='optimize send_recv communication in pipeline with interleaving.')
    group.add_argument('--enable-zero3', action='store_true', default=False,
                       help='Use this flag to enable zero3, including the segmentation of the parameters, gradients, and optimizers of the row-parallel and column-parallel models, as well as the overlap optimization of the gradient reduce sactter and weight all gather.')
    return parser


def _add_layerzero_args(parser):
    group = parser.add_argument_group(title='layerzero')
    group.add_argument('--layerzero', action='store_true', default=False,
                       help='Use this flag to enable layerzero, including the segmentation of the parameters, gradients, and optimizers of the row-parallel and column-parallel models, as well as the overlap optimization of the gradient reduce sactter and weight all gather.')
    group.add_argument('--layerzero-config', type=str,
                       help='Use this yaml file to config layerzero behaviours')
    return parser


def _add_dist_train_args(parser):
    group = parser.add_argument_group(title='dist_train')
    group.add_argument('--dist-train', action='store_true', help='Enable dist-train feature.')
    return parser


def core_transformer_config_from_args_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        config.context_parallel_algo = args.context_parallel_algo
        config.batch_p2p_comm = False
        if args.use_multiparameter_pipeline_model_parallel:
            config.deallocate_pipeline_outputs = False
        return config

    return wrapper


def validate_args_wrapper(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if args.dist_train:
            if not hasattr(args, 'mm_model'):
                raise ValueError('DistTrain must work with MindSpeed-MM')
            from mindspeed.multi_modal.dist_train.config.dist_train_config import validate_configs_world_size, \
                get_dist_model_config, merge_dist_train_args
            merge_dist_train_args(args.mm_model)
            validate_configs_world_size(args)
            cfg = get_dist_model_config(rank=args.rank)
            args.world_size = cfg.world_size
            args.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            args.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            args.context_parallel_size = cfg.context_parallel_size
        seq_parallel_enabled = args.sequence_parallel

        if defaults is None:
            defaults = {}
        replace_model_type_for_deepspeed_moe = False
        if args.num_experts:
            if args.use_ascend_coc:
                raise AssertionError('coc is not compatible with moe models')
            if args.use_ascend_mc2:
                raise AssertionError('mc2 is not compatible with moe models')
            if args.use_legacy_models:
                if args.moe_model_type == 'megatron_moe':
                    raise AssertionError('megatron_moe is not compatible with --use-legacy-models')
                replace_model_type_for_deepspeed_moe = True
            else:
                if args.moe_model_type == 'deepspeed_moe':
                    raise AssertionError('deepspeed_moe only support with --use-legacy-models')
        overlap_param_gather_without_mcore_models = False
        if args.overlap_param_gather and args.use_legacy_models:
            args.use_legacy_models = False
            overlap_param_gather_without_mcore_models = True

        #validate optimizer
        if args.optimizer_selection == 'fused_adamw': 
            print("[WARNING] The default AdamW optimizer is no longer recommended for new edition, Use the torch fused AdamW optimizer by argument --optimizer-selection fused_torch_adamw")
        elif args.optimizer_selection == 'fused_ema_adamw':
            if args.reuse_fp32_param:
                raise AssertionError('fused_ema_adamw optimizer is not compatible with reuse_fp32_param')

        # validate mla
        if args.multi_head_latent_attention:
            if args.kv_lora_rank is None:
                raise AssertionError('The parameter kv-lora-rank should be set when use multi_head_latent_attention.')
            elif args.v_head_dim is None:
                raise AssertionError('The parameter v-head-dim should be set when use multi_head_latent_attention.')
            elif args.qk_rope_head_dim is None:
                raise AssertionError(
                    'The parameter qk-rope-head-dim should be set when use multi_head_latent_attention.')
            elif args.qk_nope_head_dim is None:
                raise AssertionError(
                    'The parameter qk-nope-head-dim should be set when use multi_head_latent_attention.')

        # validate yarn
        if args.rope_scaling_type == "yarn":
            if args.rope_scaling_original_max_position_embeddings is None:
                raise AssertionError('The parameter rope_scaling_original_max_position_embeddings should be set '
                                     'when use yarn.')

        # alibi type [2, 3] is only support FA2
        if args.alibi_fusion_attn_type in [2, 3]:
            args.use_fusion_attn_v2 = True
        if args.use_fusion_attn_v2:
            args.use_flash_attn = True
            print("[WARNING] \"use_fusion_attn_v2\" is not recommended. This feature is not officially released.")            

        # for vpp assert pp should > 2
        flag_num_layers_per_virtual_pipeline_stage = None
        flag_overlap_p2p_comm = False
        if args.num_layers_per_virtual_pipeline_stage is not None and args.pipeline_model_parallel_size == 2:
            flag_num_layers_per_virtual_pipeline_stage = args.num_layers_per_virtual_pipeline_stage
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
        args = validate_args(args, defaults)

        args.variable_seq_lengths = original_variable_seq_lengths
        args.context_parallel_size = origin_context_parallel_size
        model_parallel_size = args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        args.data_parallel_size = args.world_size // (model_parallel_size * args.context_parallel_size)

        if args.dist_train:
            from mindspeed.multi_modal.dist_train.config.dist_train_config import get_all_config
            if args.tensor_model_parallel_size > 1 and seq_parallel_enabled:
                args.sequence_parallel = True
            if any(cfg.main_dp for cfg in get_all_config().values()):
                from mindspeed.multi_modal.dist_train.inner_data_parallel.utils import get_global_data_parallel_size
                args.data_parallel_size = get_global_data_parallel_size()

        if args.global_batch_size is None:
            args.global_batch_size = args.micro_batch_size * args.data_parallel_size
            if args.rank == 0:
                print('Resetting global batch size to {}'.format(
                    args.global_batch_size), flush=True)
        if args.optimize_vpp_send_recv_comm and args.num_layers_per_virtual_pipeline_stage is None:
            raise AssertionError('--optimize-vpp-send-recv-comm can only be used with pipeline with interleaving.')

        if replace_model_type_for_deepspeed_moe:
            args.use_legacy_models = origin_use_legacy_models
        if args.enable_zero3:
            print("[WARNING] zero3 currently does not support model save and load")
            if args.use_ascend_mc2 or args.reuse_fp32_param or args.recompute_granularity is not None or args.use_pipe_experts:
                raise AssertionError('zero3 cannot be used together with MC2(--use-ascend-mc2), '
                                    'parameter copy reuse(--reuse-fp32-param),'
                                    'recompute(--recompute-granularity)'
                                    'and pipe_experts(use-pipe-experts)')

        # for vpp assert pp should > 2
        if flag_num_layers_per_virtual_pipeline_stage is not None and args.pipeline_model_parallel_size == 2:
            args.num_layers_per_virtual_pipeline_stage = flag_num_layers_per_virtual_pipeline_stage
            args.overlap_p2p_comm = flag_overlap_p2p_comm
            if args.num_layers_per_virtual_pipeline_stage is not None:
                assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                    'number of layers should be divisible by the pipeline parallel size'
                num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
                assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
                    'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
                args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                                                            args.num_layers_per_virtual_pipeline_stage

        # num_layers_per_virtual_pipeline_stage should be meaningful
        if args.num_layers_per_virtual_pipeline_stage is not None:
            num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
            assert num_layers_per_pipeline_stage // args.num_layers_per_virtual_pipeline_stage > 1, \
            'considering args of num_layers and pipeline_model_parallel_size, vpp setting should be meaningful'

        # deepspeed dropless does not support pp
        if args.moe_no_drop and args.pipeline_model_parallel_size > 1:
            raise AssertionError("--moe-no-drop is not compatible with pp")

        if args.param_and_grad_buffer_pad and args.param_and_grad_buffer_pad <= 0:
            raise AssertionError('--param-and-grad-buffer-pad must be greater than 0')

        if args.use_fused_rmsnorm:
            if args.normalization != "RMSNorm":
                raise AssertionError(
                    '--use-fused-rmsnorm must enable with '
                    '--normalization=RMSNorm, but got normalization'
                    '={}.'.format(args.normalization))
            if args.use_nd_matmul:
                raise AssertionError("ND_MatMul is not compatible with fused_rmsnorm.")
        if args.use_fused_swiglu:
            if not args.swiglu:
                raise AssertionError(
                    '--use-fused-swiglu must enable with --swiglu, '
                    'but --swiglu={}.'.format(args.swiglu))
        if args.use_fused_rotary_pos_emb:
            if args.position_embedding_type != 'rope':
                raise AssertionError(
                    '--use-fused-rotary-pos-emb must enable with'
                    '--position-embedding-type=rope')
        if args.alibi_fusion_attn_type is not None and args.alibi_fusion_attn_type not in [0, 2, 3]:
            raise AssertionError('--alibi-fusion-attn-type only support for `0, 2, 3`')
        if args.reuse_fp32_param and not args.bf16:
            raise AssertionError('--reuse-fp32-param only support for `bf16`')
        if args.use_pipe_experts:
            if args.pipe_experts_multi_data <= 0:
                raise AssertionError('--pipe-experts-multi-data must greater than 0')
            if not args.sequence_parallel and args.pipe_experts_multi_stream:
                raise AssertionError('--pipe-experts-multi-stream can only be used with --sequence-parallel.')
            local_experts = args.num_experts // args.expert_model_parallel_size
            if local_experts == 1 and args.pipe_experts_multi_data == 1:
                print("[WARNING] if local_experts = num_experts // expert_model_parallel_size is equal to 1 "
                      "and --pipe-experts-multi-data is set to 1, "
                      "--use-pipe-experts will be turned off.")
                args.use_pipe_experts = False
        if args.moe_alltoall_overlap_comm and not args.moe_token_dispatcher_type == 'alltoall':
            raise AssertionError('`--moe-alltoall-overlap-comm` only support with `--moe-token-dispatcher-type alltoall`.')
        
        if args.moe_adaptive_recompute_activation and args.moe_token_dispatcher_type == 'alltoall':
            raise AssertionError('`--moe-adaptive-recompute-activation` only support with `--moe-token-dispatcher-type allgather`.')
        
        if args.moe_allgather_overlap_comm and not args.moe_token_dispatcher_type == 'allgather':
            raise AssertionError('`--moe-allgather-overlap-comm` only support with `--moe-token-dispatcher-type allgather`.')

        if args.moe_alltoall_overlap_comm or args.moe_allgather_overlap_comm:
            if not args.moe_permutation_async_comm:
                raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-permutation-async-comm`.')
            if not args.moe_grouped_gemm:
                raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-grouped-gemm`.')
        if not (args.moe_tp_extend_ep or args.moe_experts_pipeline_degree) and args.moe_alltoall_overlap_comm and args.tensor_model_parallel_size > 1:
            raise AssertionError('`--moe-alltoall-overlap-comm` do not support tp for now. only support with moe_tp_extend_ep or moe_experts_pipeline_degree when tp > 1.')
        if args.moe_experts_pipeline_degree:
            if args.moe_experts_pipeline_degree < 2:
                raise AssertionError("`--moe-experts-pipeline-degree` should be at least 2. ")
            if args.moe_experts_pipeline_degree > args.num_experts or args.num_experts % args.moe_experts_pipeline_degree != 0:
                raise AssertionError("`--moe-experts-pipeline-degree` must smaller than `--num-experts` and `--num-experts` divided by `--moe-experts-pipeline-degree` is an integer.")
            if args.moe_zero_memory != "disable":
                raise AssertionError("`--moe-experts-pipeline-degree` is not compatible with `--moe-zero-memory`")
            if not args.tensor_model_parallel_size or args.tensor_model_parallel_size <= 1:
                raise AssertionError("`--moe-experts-pipeline-degree` only support when '--tensor-model-parallel-size' is bigger than 1.")
            if args.expert_model_parallel_size > 1:
                raise AssertionError("`--moe-experts-pipeline-degree` is not compatible with expert model parallel.")
            if args.moe_tp_extend_ep:
                raise AssertionError("`--moe-experts-pipeline-degree` is not compatible with `--moe-tp-extend-ep`.")
        if args.moe_tp_extend_ep:
            if args.num_experts % (args.tensor_model_parallel_size * args.expert_model_parallel_size) != 0:
                raise AssertionError('`--moe-tp-extend-ep` only support when num_experts % ( tp * ep ) == 0')
            if not (args.moe_permutation_async_comm and args.moe_grouped_gemm):
                raise AssertionError('`--moe-tp-extend-ep` needs `--moe-permutation-async-comm` and `--moe-grouped-gemm`.')
            if args.moe_expert_capacity_factor is not None:
                raise AssertionError('`--moe-tp-extend-ep` only support when moe_expert_capacity_factor is None.')
        if args.moe_hierarchical_alltoallv:
            tp = args.tensor_model_parallel_size
            ep = args.expert_model_parallel_size
            if ((not args.moe_alltoall_overlap_comm) or (not args.moe_tp_extend_ep) or tp <= 1 or tp > torch.npu.device_count() or
                    ep * tp <= torch.npu.device_count() or args.world_size <= torch.npu.device_count()):
                raise AssertionError(
                    '`--moe-hierarchical-alltoallv` must have  `--moe-alltoall-overlap-comm` on and '
                    '`--moe-tp-extend-ep` on and 1 < tp <= torch.npu.device_count() and cross-device communication')
        if args.moe_zero_memory_num_layers is not None:
            num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
            if args.moe_zero_memory_num_layers < 0 or args.moe_zero_memory_num_layers > num_layers_per_pipeline_stage:
                raise AssertionError('`--moe-zero-memory-num-layers` must be between 0 and num layers per pipeline stage')
            if args.moe_zero_memory == "disable":
                raise AssertionError('`--moe-zero-memory` must be enabled when using `--moe-zero-memory-num-layers`')
        if args.moe_zero_memory != "disable" and args.moe_allgather_overlap_comm:
            raise AssertionError('`--moe-zero-memory` do not support `--moe-allgather-overlap-comm` for now.')
        if args.moe_dynamic_padding and not args.moe_no_drop:
            raise AssertionError('`--moe-dynamic-padding` only support for `--moe-no-drop`.')
        if args.moe_permutation_async_comm and args.moe_model_type != 'megatron_moe':
            raise AssertionError('`--moe-permutation-async-comm` only support for megatron core moe.')
        if args.moe_bmm_mc2:
            if args.moe_model_type != 'megatron_moe' or not args.moe_token_dispatcher_type == 'alltoall':
                raise AssertionError('`--moe-bmm-mc2` only support for megatron core moe and dispatcher is alltoall.')
            if not args.moe_grouped_gemm:
                raise AssertionError('`--moe-bmm-mc2` only support when `--moe-grouped-gemm` is true.')
            if args.moe_tp_extend_ep or args.moe_alltoall_overlap_comm:
                raise AssertionError(
                    '`--moe-bmm-mc2` not support with `--moe-tp-extend-ep` and `--moe-alltoall-overlap-comm`.')

        if args.context_parallel_size > 1 and args.position_embedding_type == 'alibi':
            assert args.context_parallel_algo == 'megatron_cp_algo', f"alibi only support megatron_cp_algo"
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo':
            assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size"
            head, remainder = divmod(args.num_attention_heads, args.context_parallel_size * args.tensor_model_parallel_size)
            assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by context_parallel_size * tensor_model_parallel_size"
            args.use_flash_attn = True
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'megatron_cp_algo':
            assert args.seq_length % (2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size"
            if args.position_embedding_type == 'alibi':
                assert args.alibi_fusion_attn_type in [2, 3] and args.attention_mask_type == 'causal', f"megatron_cp_algo only support alibi type in [2, 3] and attention_mask_type is causal"
            
            assert args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size, f'cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.'
            n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
            assert n_window >= 1 and remainder == 0, f'context parallel size must be divisible by cp_window_size when using double ring attention.'
            args.use_flash_attn = True
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_cp_algo':
            assert args.ulysses_degree_in_cp is not None, "--ulysses-degree-in-cp must be specified in hybrid_cp_algo"
            ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
            assert ring_degree > 1 and remainder == 0, "--ulysses-degree-in-cp must be devisible by --context-parallel-size"
            args.ring_degree = ring_degree

            head, remainder = divmod(args.num_attention_heads, args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
            assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"

            assert args.seq_length % (2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size in hybrid cp"
            
            assert args.cp_window_size >= 1 and args.cp_window_size < ring_degree, f'cp_window_size should be in range [1, ring_degree) when using double ring attention with hybrid context parallelism.'
            n_window, remainder = divmod(ring_degree, args.cp_window_size)
            assert n_window >= 1 and remainder == 0, f'ring_degree should be divisible by cp_window_size when using double ring with hybrid context parallelism.'
            args.use_flash_attn = True

        if args.context_parallel_size > 1 and args.context_parallel_algo == 'adaptive_cp_algo':
            assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size"
            args.use_flash_attn = True
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
            assert args.ulysses_degree_in_cp is not None, "--ulysses-degree-in-cp must be specified in hybrid_adaptive_cp_algo"
            ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
            assert ring_degree > 1 and remainder == 0, "--ulysses-degree-in-cp must be devisible by --context-parallel-size"
            head, remainder = divmod(args.num_attention_heads, args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
            assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"
            assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size in hybrid cp"
            args.use_flash_attn = True

        # Mandatory modification to SBH, subsequent abandonment of other formats such as BSH,BSND
        if args.shape_order != 'SBH':
            args.shape_order = 'SBH'
        if overlap_param_gather_without_mcore_models:
            args.use_legacy_models = True
        if args.transformer_impl == 'transformer_engine':
            args.transformer_impl = 'local'
        if args.fp8:
            raise AssertionError('NPU not supported FP8.')
        if args.tp_comm_overlap:
            args.tp_comm_overlap = False
        if args.recompute_method == "uniform":
            assert not args.recompute_activation_function, \
                'uniform recomputation is not compatible ' \
                'with activation function recomputation '
            assert not args.recompute_norm, \
                'uniform recomputation is not compatible ' \
                'with norm recomputation '
        if args.recompute_activation_function and args.recompute_granularity == "selective":
            raise AssertionError('--recompute-activation-function is not compatible with selective recomputation')
        adaptive_recompute_enable = args.adaptive_recompute_device_size > 0 or args.adaptive_recompute_device_swap
        if args.recompute_norm and args.recompute_granularity == "selective":
            raise AssertionError('--recompute-norm is not compatible with selective recomputation')
        if args.recompute_norm and args.use_legacy_models:
            raise AssertionError('--recompute-norm is only supported with mcore models')
        if args.use_nanopipe and not args.use_legacy_models:
            raise AssertionError('--use-nanopipe is not available with mcore models')
        if args.adaptive_recompute_device_swap and not args.use_legacy_models:
            raise AssertionError('--adaptive-recompute-device-swap is not available with mcore models')                 
        if adaptive_recompute_enable:
            assert args.recompute_granularity is None and args.recompute_method is None, \
                'adaptive selective recompute is not compatible with ' \
                'recompute_granularity and recompute_method. '
            assert not args.recompute_activation_function, \
                'adaptive selective recompute is not compatible ' \
                'with activation function recomputation '
            assert not args.swap_attention, 'adaptive selective recompute is not compatible with swap_attention feature'
            assert not args.recompute_in_advance and not args.recompute_in_bubble, 'adaptive selective recompute ' \
                'is not compatible with ripipe schedule'
            assert not args.memory_fragmentation, \
                'adaptive selective recompute is not compatible with memory fragmentation'
        if args.memory_fragmentation:
            assert not args.use_fused_rotary_pos_emb, \
                'memory fragmentation is not compatible with use_fused_rotary_pos_emb'
        if args.smart_swap:
            assert not adaptive_recompute_enable, 'smart swap is not compatible with adaptive selective recompute'
            assert not args.memory_fragmentation, 'smart swap is not compatible with memory fragmentation'
        if args.adaptive_memory_optimization:
            assert args.ampipe_degree <= 1, 'adaptive memory optimization is not compatible with ampipe'
            assert not adaptive_recompute_enable, 'adaptive memory optimization is not compatible with adaptive recomputing'
            assert args.recompute_granularity is None and args.recompute_method is None, \
                'adaptive memory optimization is not compatible with recompute_granularity or recompute_method'
            assert not args.recompute_activation_function, \
                'adaptive memory optimization is not compatible with recompute_activation_function'
            assert not args.swap_attention, 'adaptive memory optimization is not compatible with swap_attention feature'
            assert not args.recompute_in_bubble, 'adaptive memory optimization is not compatible with recompute_in_bubble'
            assert not args.memory_fragmentation, \
                'adaptive memory optimization is not compatible with memory_fragmentation'
        if args.use_flash_attn:
            assert args.sparse_mode == 0 or args.sparse_mode == 2, f"Only supports sparse modes 0 and 2"
        args.create_attention_mask_in_dataloader = False
        if args.automated_pipeline:
            if args.recompute_activation_function:
                print("[WARNING] disable activation function recomputation when enabling automated pipeline")
                args.recompute_activation_function = False
            if args.recompute_granularity is not None or args.recompute_method is not None:
                print("[WARNING] disable recompute granularity and recompute method when enabling automated pipeline")
                args.recompute_granularity = None
                args.recompute_method = None
            if args.noop_layers:
                print("[WARNING] disable noop_layers when enabling automated pipeline")
                args.noop_layers = None
        if args.automated_pipeline_perf:
            if args.automated_pipeline:
                print("[WARNING] disable automated pipeline when enabling automated pipeline performance version")
                args.automated_pipeline = False
            if args.num_layers_per_virtual_pipeline_stage is not None:
                raise AssertionError('automated pipeline performance is temporarily incompatible with virtual pipeline')
        if args.use_ascend_mc2:
            if args.use_ascend_coc:
                raise AssertionError('--mc2 and coc can not be used together')
        if args.use_nd_matmul:
            if args.normalization == 'LayerNorm':
                raise AssertionError('ND_MatMul is temporarily incompatible with LayerNorm')
            if args.load is not None or args.pretrained_checkpoint is not None:
                raise AssertionError('ND_MatMul does not support loading weights for training temporarily')
            if args.tensor_model_parallel_size % args.nd1_dim1_size != 0:
                raise AssertionError('tensor_model_parallel_size must be divisible by nd1_dim1_size')
            if args.tensor_model_parallel_size % args.nd2_dim1_size != 0:
                raise AssertionError('tensor_model_parallel_size must be divisible by nd2_dim1_size')

        args.reduce_recompute_for_last_chunk = False
        if args.recompute_in_advance:
            args.reduce_recompute_for_last_chunk = True
            if args.recompute_method == "uniform":
                raise AssertionError('recompute_in_advance does not support uniform recompute_method')
            if not args.recompute_num_layers and not args.adaptive_memory_optimization:
                raise AssertionError('recompute_num_layers can not be None or 0 when using recompute_in_advance')
            if args.pipeline_model_parallel_size <= 1 or args.num_layers_per_virtual_pipeline_stage is None:
                raise AssertionError('recompute_in_advance only support pipelining with interleaving')
            if args.num_layers_per_virtual_pipeline_stage != 1:
                args.recompute_in_advance = False
        if args.recompute_in_bubble:
            if args.recompute_num_layers:
                raise AssertionError('recompute_num_layers must be None or 0 when using recompute_in_bubble')
            if args.pipeline_model_parallel_size <= 1 or args.num_layers_per_virtual_pipeline_stage is None:
                raise AssertionError('recompute_in_bubble only support pipelining with interleaving')
            if not args.swap_attention:
                # Following is a trick to realize bubble recomputation. We first enable all recomputation,
                # and then disable recomputation for all layers except the ones chosen for bubble recomputation.
                args.recompute_granularity = "full"
                args.recompute_method = "block"
            if args.enable_recompute_layers_per_pp_rank:
                args.recompute_num_layers = args.num_layers // args.pipeline_model_parallel_size
            else:
                args.recompute_num_layers = args.num_layers_per_virtual_pipeline_stage
        if isinstance(args.noop_layers, str):
            noop_layers = set()
            for x in args.noop_layers.split(','):
                if int(x) >= args.num_layers or int(x) < 0:
                    raise AssertionError(f'each element in args.noop_layers({args.noop_layers}) should bigger or equal '
                                         f'to 0 and smaller than args.num_layers({args.num_layers})')
                noop_layers.add(int(x))
            args.noop_layers = noop_layers

        if args.ampipe_degree > 1:
            assert args.use_flash_attn, "ampipe only supports flash attention, please enable '--use-flash-attn'."
            assert args.num_experts is not None, "ampipe only supports MoE model."
            assert args.expert_model_parallel_size > 1, "ampipe only supports expert_model_parallel_size > 1"
            assert args.moe_model_type == 'deepspeed_moe', "ampipe only supports deepspeed_moe."
            assert not args.use_ascend_mc2, "ampipe does't supports ascend mc2 for now."
            assert not args.add_bias_linear, "ampipe does't supports bias linear for now."
            assert not args.overlap_grad_reduce, "ampipe does't supports overlap_grad_reduce for now."
            assert not args.overlap_param_gather, "ampipe does't supports overlap_param_gather for now."
            assert not args.use_nanopipe, "ampipe does't supports use_nanopipe for now."
            assert not args.recompute_in_bubble, "ampipe does't supports ripipe recompute_in_bubble for now."
            assert not args.recompute_in_advance, "ampipe does't supports ripipe recompute_in_advance for now."
            assert not args.adaptive_recompute_device_swap, "ampipe does't supports ripipe recompute_in_advance for now."
            if args.sequence_parallel:
                assert args.seq_length % (args.ampipe_degree * args.tensor_model_parallel_size) == 0, \
                    "sequence length must be divisible by ampipe_degree * tensor_model_parallel_size"
            if args.context_parallel_size > 1:
                assert args.context_parallel_algo == 'megatron_cp_algo', "ampipe only supports megatron_cp_algo"
                assert args.ampipe_degree == 2, "ampipe only supports ampipe_degree=2 when context_parallel_size>1"
                slice_size, remainder = divmod(args.seq_length, 2 * args.ampipe_degree * args.context_parallel_size)
                assert remainder == 0, \
                    "sequence length must be divisible by 2 * ampipe_degree * context_parallel_size"
                if args.sequence_parallel:
                    assert slice_size % (args.tensor_model_parallel_size) == 0, \
                        "sequence length must be divisible by 2 * ampipe_degree * context_parallel_size * tensor_model_parallel_size"
            if args.use_pipe_experts:
                if args.pipe_experts_multi_data % args.ampipe_degree != 0:
                    print("[WARNING] if pipe_experts_multi_data isn't divisible by ampipe_degree "
                          "--use-pipe-experts will be turned off.")
                    args.use_pipe_experts = False
                    args.pipe_experts_multi_stream = False
                    args.pipe_experts_multi_data = 1
        if args.tp_2d:
            if args.sequence_parallel:
                raise AssertionError('2d tp does not support sequence parallel')
            if args.use_fused_rmsnorm:
                raise AssertionError('2d tp does not support fused rmsnorm')
            if args.use_nanopipe:
                raise AssertionError('tp-2d does not support nano-pipe')
            if args.ampipe_degree > 1:
                raise AssertionError('tp-2d does not support ampipe')
            if args.context_parallel_algo not in ['megatron_cp_algo', 'ulysses_cp_algo']:
                raise AssertionError('tp-2d now only support megatron_cp_algo or ulysses_cp_algo')
            if args.use_ascend_coc:
                raise AssertionError('tp-2d does not support ascend coc')
            if args.tensor_model_parallel_size // args.tp_x != args.tp_y:
                raise AssertionError('need satisfy tp = tp_x * tp_y')
            if args.expert_model_parallel_size > 1:
                if args.moe_token_dispatcher_type != "allgather":
                    raise AssertionError('2d tp only support allgather megatron-moe now')

        if args.expert_interval <= 0 or args.expert_interval > args.num_layers:
            raise AssertionError("--expert-interval must be between 1 and num layers")
        if args.moe_train_capacity_factor <= 0.0:
            raise AssertionError("--moe-train-capacity-factor must be greater than 0.0")

        if args.gemm_gradient_accumulation_fusion:
            if not args.moe_grouped_gemm:
                raise AssertionError('`--gemm-gradient-accumulation-fusion` only support with `--moe-grouped-gemm`.')

        if args.use_legacy_models:
            if args.overlap_param_gather and args.reuse_fp32_param:
                raise AssertionError('In legacy, `overlap_param_gather` does not support `reuse_fp32_param`.')

        if args.fp16:
            args.gradient_accumulation_fusion = False
            warnings.warn("Unsupported gradient fp16 bf16 for gradient accumulation fusion")
        
        if args.reset_attention_mask and args.attention_mask_type == 'causal':
            assert args.context_parallel_algo == 'megatron_cp_algo', 'accelerated eod reset mode only support ring attention'

        if args.context_parallel_kv_cache_policy:
            if args.context_parallel_size == 1:
                raise AssertionError(
                    'context parallel size must larger than 1 when --context-parallel-kv-cache-policy is set.')
            if not args.use_flash_attn:
                raise AssertionError(
                    '--context-parallel-kv-cache-policy only support use flash attention.'
                )

        if args.context_parallel_cache_interval != 0:
            if not args.context_parallel_kv_cache_policy:
                raise AssertionError(
                    '--context-parallel-cache-interval only can be used when --context-parallel-kv-cache-policy is set.'
                )
            if args.context_parallel_cache_interval >= args.num_layers:
                raise AssertionError(
                    '--context-parallel-cache-interval should be smaller than the number of layers.'
                )
            if args.context_parallel_cache_interval < 0:
                raise AssertionError(
                    '--context-parallel-cache-interval cannot be negative number.'
                )

        if args.use_ulysses_allgather_kv:
            if args.context_parallel_size == 1:
                raise AssertionError(
                    'context parallel size must larger than 1 when --use-ulysses-allgather-kv is set.')
            if args.context_parallel_algo != 'ulysses_cp_algo':
                raise AssertionError(
                    '--context_parallel-algo should be ulysses_cp_algo when using --use-ulysses-allgather-kv.'
                )
            if not args.group_query_attention:
                raise AssertionError(
                    '--use-ulysses-allgather-kv needs to enable --group-query-attention.'
                )

        from megatron.training.arguments import _print_args
        _print_args('arguments', args, True)

        for feature in FEATURES_LIST:
            if feature.optimization_level <= args.optimization_level and \
                    (getattr(args, feature.feature_name, None) or feature.default_patches):
                feature.pre_validate_args(args)
                feature.validate_args(args)
                feature.post_validate_args(args)

        return args

    return wrapper


def add_parser_argument_choices_value(parser, argument_name, value):
    if parser._actions:
        for action in parser._actions:
            if isinstance(action, argparse._ArgumentGroup):
                add_parser_argument_choices_value(action, argument_name)
            elif isinstance(action, argparse.Action) and argument_name in action.option_strings:
                action.choices.append(value)


def _add_alibi_args(parser):
    add_parser_argument_choices_value(parser, "--position-embedding-type", 'alibi')

    group = parser.add_argument_group(title='alibi')
    group.add_argument('--square-alibi-mask',
                       action='store_true',
                       default=False,
                       help='attention mask of alibi is squared')
    group.add_argument('--fill-neg-inf',
                       action='store_true',
                       default=False,
                       help='fill alibi with negative inf')

    group.add_argument('--alibi-fusion-attn-type',
                    type=int,
                    help='alibi pse type, support for 0,2,3')

    group.add_argument('--alibi-diagonal-opposite',
                       action='store_true',
                       default=False,
                       help='make alibi diagonal opposite')

    return parser


def _add_ndmm_args(parser):
    group = parser.add_argument_group(title='ndmm')
    group.add_argument('--use-nd-matmul', action='store_true', default=False,
                       help='use use-nd-matmul to replace megatron-style tensor parallel')
    group.add_argument('--nd1-dim1-size', type=int, default=1,
                       help='Dim1 of the first nd matmul when use-3d-matmul is True')
    group.add_argument('--nd2-dim1-size', type=int, default=1,
                       help='Dim1 of the second nd matmul when use-3d-matmul is True')
    return parser


def _add_auto_parallel_args(parser):
    group = parser.add_argument_group(title='auto_parallel')
    group.add_argument('--auto-parallel', action='store_true', 
                       help='enable automatic parallelism with auto-parallel')
    group.add_argument('--nnodes', type=int, default=1, help='the number of node in the cluster')
    group.add_argument('--nproc-per-node', type=int, default=8, help='the number of NPU on each node')
    group.add_argument('--master-addr', type=str, default=None, help='the ip-address of master node')
    group.add_argument('--master-port', type=str, default=None, help='the ip-port of master node')
    group.add_argument('--node-rank', type=int, default=0, 
                       help='the rank of nodes in the cluster, starting from 0 and increment by 1')
    group.add_argument('--profile-operator', action='store_true', help='')
    group.add_argument('--profile-memory', action='store_true', help='')
    group.add_argument('--prof-file', type=str, default=None, help='')
    return parser


def _add_auto_parallel_mm_args(parser):
    group = parser.add_argument_group(title='auto_parallel_mm')
    group.add_argument('--auto-parallel-mm', action='store_true', default=False,
                       help='enable multimode automated parallel policy search')
    group.add_argument('--auto-parallel-profile', action='store_true', default=False, 
                       help='multimode performance sampling')

    return parser


def _add_2d_tp_args(parser):
    group = parser.add_argument_group(title='2d-tp')
    group.add_argument('--tp-2d', action='store_true', default=False,
                       help='use use-2d-tp to replace megatron-style tensor parallel')
    group.add_argument('--tp-x', type=int, default=1,
                       help='the fist dim tensor parallel size for Linear')
    group.add_argument('--tp-y', type=int, default=1,
                       help='the second dim tensor parallel size for Linear')
    group.add_argument('--enable-overlap-ag-with-matmul', action='store_true', default=False,
                       help='use enable-overlap-ag-with-matmul to overlap all-gather with matmul')
    group.add_argument('--enable-overlap-matmul-with-rs', action='store_true', default=False,
                       help='use enable-overlap-matmul-with-rs to overlap matmul with reduce-scatter')
    group.add_argument('--enable-backward-overlap-ag-with-matmul', action='store_true', default=False,
                       help='use enable-backward-overlap-ag-with-matmul to overlap all-gather  with matmul in backward')
    return parser


def _add_hccl_group_buffer_args(parser):
    group = parser.add_argument_group(title='hccl-group-buffer')
    group.add_argument('--hccl-group-buffer', type=str, default=None,
                       help='the hccl buffer for group')
    group.add_argument('--hccl-group-buffer-adaptive', action='store_true', default=False,
                       help='the hccl buffer for group adaptively')
    group.add_argument('--hccl-ep-group-buffer-adaptive-factor', type=float, default=-1.0,
                       help='the ep group buffer factor')
    return parser 
