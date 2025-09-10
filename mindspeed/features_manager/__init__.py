from .tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature
from .functional.profiler_default_feature import ProfilerDefaultFeature
from .pipeline_parallel.fb_overlap_feature import FwdBwdOverlapFeature
from .optimizer.swap_optimizer_feature import SwapOptimizerFeature
from .pipeline_parallel.fb_overlap_feature import FwdBwdOverlapFeature
from .pipeline_parallel.dualpipev_feature import DualpipeVFeature
from .pipeline_parallel.chimera_feature import ChimeraFeature
from .optimizer.virtual_optimizer import VirtualOptimizerFeature
from .compress_dense.compress_dense import AnsCompressTensorFeature
from .moe.moe_zerc_feature import MoeZeRCFeature

FEATURES_LIST = [
    SwapOptimizerFeature(),
    VirtualOptimizerFeature(),
    AnsCompressTensorFeature(),
    # Functional features
    ProfilerDefaultFeature(),
    # Tensor parallel features
    UnalignedLinearFeature(),
    # Pipeline Parallel features
    FwdBwdOverlapFeature(),
    DualpipeVFeature(),
    ChimeraFeature(),
    # MoE features
    MoeZeRCFeature()
]
