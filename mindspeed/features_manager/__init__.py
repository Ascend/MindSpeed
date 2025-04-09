from mindspeed.features_manager.functional.profiler_default import ProfilerDefaultFeature
from mindspeed.features_manager.functional.npu_deterministic import NPUDeterministicFeature

from mindspeed.features_manager.fusions.grouped_matmul import GroupedMatmulFeature
from mindspeed.features_manager.fusions.fused_bias_swiglu import FusedSwigluFeature

from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature
from mindspeed.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature

from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from mindspeed.features_manager.pipeline_parallel.unaligned_pineline_feature import UnalignedPipelineFeature

from mindspeed.features_manager.llava.llava_multimodal import LlavaModel

FEATURES_LIST = [
    # Functional features
    ProfilerDefaultFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # llava-multimodal
    LlavaModel(),
    UnalignedPipelineFeature()
]

# this list is for reconstruction of mindspeed
FEATURES_LIST_V2 = (
    # Functional features
    ProfilerDefaultFeature(),
    NPUDeterministicFeature(),

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
    # unaligned pipeline
    UnalignedPipelineFeature()
)
