from typing import List

from mindspeed.deprecate import AutoExecuteFunction

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
from mindspeed.features_manager.context_parallel.adaptive_context_parallel import AdaptiveContextParallelFeature
from mindspeed.features_manager.context_parallel.context_parallel_feature import ContextParallelFeature
from mindspeed.features_manager.context_parallel.context_parallel_kv_cache import ContextParallelKvCacheFeature
from mindspeed.features_manager.context_parallel.ulysses_context_parallel import UlyssesContextParallelFeature
from mindspeed.features_manager.data_parallel.async_log_allreduce import AsyncLogAllreduceFeature
from mindspeed.features_manager.functional.profile import ProfileFeature
from mindspeed.features_manager.functional.profiler_default import ProfilerDefaultFeature
from mindspeed.features_manager.functional.npu_deterministic import NPUDeterministicFeature
from mindspeed.features_manager.functional.tflops_calculate import TflopsCalculateFeature

from mindspeed.features_manager.fusions.grouped_matmul import GroupedMatmulFeature
from mindspeed.features_manager.fusions.fused_bias_swiglu import FusedSwigluFeature
from mindspeed.features_manager.fusions.fused_softmax import FusedSoftmaxFeature
from mindspeed.features_manager.fusions.fused_rope import FusedRoPEFeature
from mindspeed.features_manager.fusions.fused_moe_permute import FusedMoEPermuteFeature
from mindspeed.features_manager.affinity.affinity import AffinityFeature

from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature
from mindspeed.features_manager.megatron_basic.transformer_engine_basic import TransformerEngineBasicFeature
from mindspeed.features_manager.pipeline_parallel import NoopLayersFeature
from mindspeed.features_manager.pipeline_parallel.optimize_p2p_comm import OptimizeP2PCommFeature
from mindspeed.features_manager.pipeline_parallel.ripipe_schedules_feature import RiPipeSchedulesBubbleFeature, \
    RiPipeSchedulesAdvanceFeature

from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from mindspeed.features_manager.pipeline_parallel.unaligned_pineline_feature import UnalignedPipelineFeature

from mindspeed.features_manager.llava.llava_multimodal import LlavaModel
from mindspeed.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature
from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed.features_manager.recompute.activation_function import RecomputeActivationFeature
from mindspeed.features_manager.recompute.norm_function import RecomputeNormFeature
from mindspeed.features_manager.recompute.enable_recompute_layers_per_pp_rank import EnableRecomputeLayersPerPPRank
from mindspeed.features_manager.recompute.recompute_method import RecomputeMethodFeature
from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature

from mindspeed.features_manager.tensor_parallel.mc2 import MC2Feature
from mindspeed.features_manager.tensor_parallel.locl_coc import CoCFeature

from mindspeed.features_manager.moe.tp_extend_ep import MoETpExtendEpFeature
from mindspeed.features_manager.moe.gmm import MoEGmmFeature
from mindspeed.features_manager.moe.shared_expert import MoESharedExpertsFeature
from mindspeed.features_manager.moe.fb_overlap import MoEFwdBwdOverlapFeature
from mindspeed.features_manager.moe.balanced_moe import BalancedMoEFeature
from mindspeed.features_manager.moe.moe_allgather_overlap import MoEAllGatherOverLapFeature
from mindspeed.features_manager.moe.moe_alltoall_overlap import MoEAlltoAllOverLapFeature
from mindspeed.features_manager.moe.moe_alltoall_mc2 import MoEAlltoAllMC2Feature
from mindspeed.features_manager.moe.moe_zero_memory import MoEZeroMemoryFeature
from mindspeed.features_manager.moe.experts_placement import ExpertsPlacementFeature
from mindspeed.features_manager.moe.moe_fix_router import MoEFixRouterFeature

from mindspeed.features_manager.hccl_buffer.hccl_buffer_adaptive import HcclBufferAdaptiveFeature
from mindspeed.features_manager.hccl_buffer.hccl_buffer_set import HcclBufferSetFeature

from mindspeed.features_manager.optimizer.fused_ema_adamw_feature import FusedEmaAdamwFeature
from mindspeed.features_manager.optimizer.virtual_optimizer import VirtualOptimizerFeature
from mindspeed.features_manager.transformer.flash_attention.alibi_feature import AlibiFeature
from mindspeed.features_manager.transformer.flash_attention.fusion_attention_v1_feature import FusionAttentionFeature
from mindspeed.features_manager.transformer.flash_attention.fusion_attention_v2_feature import FusionAttentionV2Feature
from mindspeed.features_manager.transformer.flash_attention.generate_mask_feature import GenerateMaskFeature
from mindspeed.features_manager.transformer.flash_attention.reset_attention_mask_feature import ResetAttentionMaskFeature
from mindspeed.features_manager.transformer.module_rearrange.mcore_rearrange import MegatronMcoreRearrangeFeature
from mindspeed.features_manager.pipeline_parallel.variable_seq_length import VariableSequenceLengthFeature
from mindspeed.features_manager.pipeline_parallel.multi_parameter import MultiParameterFeature
from mindspeed.features_manager.pipeline_parallel.optimize_send_recv_comm import OptimizeSendRecvCommFeature
from mindspeed.features_manager.pipeline_parallel.dualpipev_feature import DualpipeVFeature
from mindspeed.features_manager.memory.reuse_fp32_param import ReuseFP32Param
from mindspeed.features_manager.memory.smart_swap import SmartSwapFeature

from mindspeed.features_manager.dist_train.dist_train_feature import DistTrainFeature

from mindspeed.features_manager.tokenizer.build_tokenizer import BuildTokenizerFeature
from mindspeed.features_manager.distributed.buffer_pad import BufferPadFeature
from mindspeed.features_manager.distributed.torch_fully_sharded_data_parallel import TorchFullyShardedDataParallelFeature
from mindspeed.features_manager.distributed.reset_bucket_group_order_feature import ResetBucketGroupOrderFeature

from mindspeed.features_manager.custom_fsdp.custom_fsdp_feature import CustomFSDPFeature
from mindspeed.features_manager.tensor_parallel.tp_2d import TP2dFeature
from mindspeed.features_manager.compress_dense.compress_dense import AnsCompressTensorFeature
from mindspeed.features_manager.compress.compress import CompressActivationFeature, CompressOptimizerFeature
from mindspeed.features_manager.disable_gloo_group.disable_gloo_group_feature import DisableGlooGroupFeature

from mindspeed.features_manager.memory.swap_attention import SwapAttentionFeature
from mindspeed.features_manager.transformer.multi_head_latent_attention.mla_feature import MLAFeature

from mindspeed.features_manager.tensor_parallel.vocab_parallel import ReplaceIndexPutFeature
from mindspeed.features_manager.distributed.layerzero import LayerZeroFeature
from mindspeed.features_manager.optimizer.swap_optimizer_feature import SwapOptimizerFeature
from mindspeed.features_manager.optimizer.low_precision_optimizer_feature import LowPrecisionOptimizerFeature

from mindspeed.features_manager.ai_framework.ms_patch_feature import MindSporePatchFeature

from mindspeed.features_manager.auto_settings.auto_settings_feature import AutoSettingsFeature
from mindspeed.features_manager.ckpt_acceleration.ckpt_acceleration import CkptAccelerationFeature


FEATURES_LIST = [
    # Functional features
    ProfilerDefaultFeature(),
    # Virtaul Optimizer
    VirtualOptimizerFeature(),
    LowPrecisionOptimizerFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # llava-multimodal
    LlavaModel(),
    UnalignedPipelineFeature()
]


# this list is for reconstruction of mindspeed
def add_megatron_basic_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RequirementsBasicFeature(),
        MegatronBasicFeature(),
        TransformerEngineBasicFeature()
    ])


def add_fusions_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        GroupedMatmulFeature(),
        FusedSwigluFeature(),
        FusedSoftmaxFeature(),
        FusedRoPEFeature(),
        FusedMoEPermuteFeature(),
    ])


def add_affinity_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        AffinityFeature(),
    ])


def add_functional_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ProfilerDefaultFeature(),
        NPUDeterministicFeature(),
        TflopsCalculateFeature(),
        ProfileFeature(),
    ])


def add_recompute_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RecomputeActivationFeature(),
        RecomputeNormFeature(),
        EnableRecomputeLayersPerPPRank(),
        RecomputeMethodFeature()
    ])


def add_context_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ContextParallelFeature(),
        UlyssesContextParallelFeature(),
        ContextParallelKvCacheFeature(),
        AdaptiveContextParallelFeature()
    ])


def add_data_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        AsyncLogAllreduceFeature(),
    ])


def add_tensor_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        UnalignedLinearFeature(),
        MC2Feature(),
        CoCFeature(),
        TP2dFeature(),
        ReplaceIndexPutFeature()
    ])


def add_pipeline_parallel_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        RiPipeSchedulesBubbleFeature(),
        RiPipeSchedulesAdvanceFeature(),
        NoopLayersFeature(),
        OptimizeP2PCommFeature(),
        VariableSequenceLengthFeature(),
        MultiParameterFeature(),
        OptimizeSendRecvCommFeature(),
        UnalignedPipelineFeature(),
        DualpipeVFeature(),
    ])


def add_transformer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        FusionAttentionFeature(),
        FusionAttentionV2Feature(),
        AlibiFeature(),
        GenerateMaskFeature(),
        ResetAttentionMaskFeature(),
        MLAFeature(),
        MegatronMcoreRearrangeFeature()
    ])


def add_moe_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        MoEGmmFeature(),
        MoETpExtendEpFeature(),
        MoESharedExpertsFeature(),
        MoEAllGatherOverLapFeature(),
        MoEAlltoAllOverLapFeature(),
        MoEZeroMemoryFeature(),
        MoEFwdBwdOverlapFeature(),
        BalancedMoEFeature(),
        ExpertsPlacementFeature(),
        MoEAlltoAllMC2Feature(),
        MoEFixRouterFeature(),
    ])


def add_hccl_buffer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        HcclBufferSetFeature(),
        HcclBufferAdaptiveFeature(),
    ])


def add_optimizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        # Optimizer features: fused-ema-adamw
        FusedEmaAdamwFeature(),
        VirtualOptimizerFeature(),
        LowPrecisionOptimizerFeature(),
    ])


def add_llava_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        LlavaModel()
    ])


def add_dist_train_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        DistTrainFeature()
    ])


def add_tokenizer_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        BuildTokenizerFeature()
    ])


def add_distributed_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        BufferPadFeature(),
        LayerZeroFeature(),
        TorchFullyShardedDataParallelFeature(),
        CustomFSDPFeature(),
        ResetBucketGroupOrderFeature()
    ])


def add_reuse_param_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        ReuseFP32Param()
    ])


def add_swap_manage_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        SmartSwapFeature(),
        SwapAttentionFeature()
    ])


def add_compress_dense_features(features_list: List[MindSpeedFeature]):
    features_list.extend([
        AnsCompressTensorFeature()
    ])


def add_compress_memory_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        CompressActivationFeature(),
        CompressOptimizerFeature()
    ])


def add_disable_gloo_group_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        DisableGlooGroupFeature()
    ])


def add_swap_optimizer_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        SwapOptimizerFeature(),
    ])


def add_ai_framework_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        MindSporePatchFeature()
    ])


def add_auto_settings_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        AutoSettingsFeature(),
    ])


def add_ckpt_acceleration_feature(features_list: List[MindSpeedFeature]):
    features_list.extend([
        CkptAccelerationFeature(),
    ])


def create_features_list():
    features_list = []
    add_megatron_basic_features(features_list)
    add_context_parallel_features(features_list)
    add_data_parallel_features(features_list)
    add_fusions_features(features_list)
    add_affinity_features(features_list)
    add_functional_features(features_list)
    add_recompute_features(features_list)
    add_tensor_parallel_features(features_list)
    add_pipeline_parallel_features(features_list)
    add_moe_features(features_list)
    add_hccl_buffer_features(features_list)
    add_optimizer_features(features_list)
    add_llava_features(features_list)
    add_dist_train_features(features_list)
    add_tokenizer_features(features_list)
    add_distributed_features(features_list)
    add_reuse_param_features(features_list)
    add_swap_manage_features(features_list)
    add_compress_dense_features(features_list)
    add_compress_memory_feature(features_list)
    add_disable_gloo_group_feature(features_list)
    add_swap_optimizer_feature(features_list)
    add_transformer_features(features_list)
    add_ai_framework_feature(features_list)
    add_auto_settings_feature(features_list)
    add_ckpt_acceleration_feature(features_list)
    return features_list


@AutoExecuteFunction
def set_default_features_list():
    MindSpeedFeaturesManager.set_features_list(create_features_list())


set_default_features_list()
