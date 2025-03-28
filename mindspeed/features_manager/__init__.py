from .tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from .functional.profiler_default_feature import ProfilerDefaultFeature
from .optimizer.swap_optimizer_feature import SwapOptimizerFeature

FEATURES_LIST = [
    SwapOptimizerFeature(),
    # Functional features
    ProfilerDefaultFeature(),
    # Tensor parallel features
    UnalignedLinearFeature()
]
