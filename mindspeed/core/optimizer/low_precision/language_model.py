from functools import wraps
import torch
from megatron.training import get_args


def transformer_language_model_init_wrapper(init_func):
    @wraps(init_func)
    def transformer_language_model_init_func(self, *args, **kwargs):
        init_func(self, *args, **kwargs)
        args = get_args()
        quant_enabled = bool(getattr(args, "quant_states", None))
        if not quant_enabled:
            return
        for name, param in self.named_parameters():
            if "output_layer" in name or "embedding" in name:
                setattr(param, "keep_fp32", True)

    return transformer_language_model_init_func