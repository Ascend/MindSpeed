from functools import wraps
from megatron.training import get_args
from mindspeed.model.transformer import generate_attention_mask
import mindspeed.model.transformer


def get_attention_mask_wrapper(get_attention_mask):
    @wraps(get_attention_mask)
    def wrapper(*args, **kwargs):
        argument = get_args()
        automated_pipeline_perf = argument.automated_pipeline_perf and argument.optimized_mbs_list
        if automated_pipeline_perf:
            generate_attention_mask()
        else:
            get_attention_mask(*args, **kwargs)
        return mindspeed.model.transformer._GLOBAL_ATTN_MASK

    return wrapper
