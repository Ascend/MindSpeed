from .tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from .functional.profiler_default_feature import ProfilerDefaultFeature
from .pipeline_parallel.fb_overlap_feature import FwdBwdOverlapFeature
from .optimizer.swap_optimizer_feature import SwapOptimizerFeature
from .pipeline_parallel.fb_overlap_feature import FwdBwdOverlapFeature
from .pipeline_parallel.dualpipev_feature import DualpipeVFeature
from .optimizer.virtual_optimizer import VirtualOptimizerFeature
from .moe.moe_zerc_feature import MoeZeRCFeature

FEATURES_LIST = [
    SwapOptimizerFeature(),
    VirtualOptimizerFeature(),
    # Functional features
    ProfilerDefaultFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # Pipeline Parallel features
    FwdBwdOverlapFeature(),
    DualpipeVFeature(),
    # MoE features
    MoeZeRCFeature()
]
