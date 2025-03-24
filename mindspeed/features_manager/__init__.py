from .tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from .functional.profiler_default_feature import ProfilerDefaultFeature
from .llava.llava_multimodal import LlavaModel

FEATURES_LIST = [
    # Functional features
    ProfilerDefaultFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # llava-multimodal
    LlavaModel()
]


FEATURES = (
    # Functional features
    ProfilerDefaultFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # llava-multimodal
    LlavaModel(),
)
