from megatron.training import get_args
from mindspeed.core.memory.adaptive_memory.adaptive_memory_swap_manager import SwapManager as AdaptiveMemorySwapManager
from mindspeed.core.memory.adaptive_recomputing.swap_manager import SwapManager as AdaptiveRecomputingSwapManager


def swap_out_by_size(size):
    args = get_args()
    if args.adaptive_memory_optimization:
        return AdaptiveMemorySwapManager().swap_out_by_size(size)
    else:
        return AdaptiveRecomputingSwapManager().swap_out_by_size(size)
