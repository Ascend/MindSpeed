from mindspeed.features_manager.functional.profiler_default import ProfilerDefaultFeature
from mindspeed.features_manager.functional.npu_deterministic import NPUDeterministicFeature
from mindspeed.features_manager.functional.tflops_calculate import TflopsCalculateFeature
from mindspeed.features_manager.functional.profile import ProfileFeature

from mindspeed.features_manager.fusions.grouped_matmul import GroupedMatmulFeature
from mindspeed.features_manager.fusions.fused_bias_swiglu import FusedSwigluFeature

from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature

from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from mindspeed.features_manager.pipeline_parallel.unaligned_pineline_feature import UnalignedPipelineFeature

from mindspeed.features_manager.llava.llava_multimodal import LlavaModel
from mindspeed.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature
from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed.features_manager.recompute.activation_function import RecomputeActivationImpl
from mindspeed.features_manager.recompute.norm_function import RecomputeNormImpl
from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature

from mindspeed.features_manager.moe.tp_extend_ep import MoETpExtendEpFeature
from mindspeed.features_manager.moe.gmm import MoEGmmFeature
from mindspeed.features_manager.moe.shared_expert import MoESharedExpertsFeature
from mindspeed.features_manager.optimizer.virtual_optimizer import VirtualOptimizerFeature

FEATURES_LIST = [
    # Functional features
    ProfilerDefaultFeature(),
    # Virtaul Optimizer
    VirtualOptimizerFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # llava-multimodal
    LlavaModel(),
    UnalignedPipelineFeature()
]

# this list is for reconstruction of mindspeed
FEATURES_LIST_V2 = (
    # recompute
    RecomputeActivationImpl(),
    RecomputeNormImpl(),

    # Functional features
    ProfilerDefaultFeature(),
    NPUDeterministicFeature(),
    TflopsCalculateFeature(),
    ProfileFeature(),

    # Fusions features
    GroupedMatmulFeature(),
    FusedSwigluFeature(),

    # Megatron basic
    RequirementsBasicFeature(),
    MegatronBasicFeature(),

    # Tensor parallel features
    UnalignedLinearFeature(),

    # llava-multimodal
    LlavaModel(),

    # MoeExperts use gemm
    MoEGmmFeature(),
    # MoeTp2EpFeature
    MoETpExtendEpFeature(),
    # MoeSharedExpertsFeature
    MoESharedExpertsFeature()
)
