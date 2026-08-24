from functools import wraps
import torch
from mindspeed.args_utils import get_full_args


def quant_grad_param_and_grad_buffer_init_wrapper(init_func):
    @wraps(init_func)
    def quant_grad_param_and_grad_buffer_init(self, ddp_config, param_dtype, grad_dtype, *args, **kwargs):
        quant_args = get_full_args()
        quant_grads_enabled = getattr(quant_args, 'quant_grads', False)
        if quant_grads_enabled:
            # Keep the FP16 DDP grad buffer so both memory and communication stay
            # in FP16. We deliberately do not attach per-tensor quantization metadata
            # here: gradients are accumulated natively in FP16 and only copied to the
            # optimizer in FP16. Re-quantizing on every micro-batch is what caused the
            # precision divergence.
            grad_dtype = torch.float16

        init_func(self, ddp_config, param_dtype, grad_dtype, *args, **kwargs)

        if not quant_grads_enabled:
            return

        # Default NaN/Inf checks use the unquantized bucket values; disable them and rely on
        # higher-level AMP/non-finite handling when quant grads are enabled.
        self.ddp_config.check_for_nan_in_grad = False
        self.ddp_config.check_for_large_grads = False

        # No per-gradient ScaleMeta / bucket.scales are needed for the FP16-native
        # path. Keeping them would force repeated dequant/requant during micro-batch
        # accumulation and add a scale-sync step before DDP communication.
        return

    return quant_grad_param_and_grad_buffer_init


def quant_grad_start_grad_sync_wrapper(start_grad_sync):
    @wraps(start_grad_sync)
    def quant_start_grad_sync(self):
        return start_grad_sync(self)

    return quant_start_grad_sync
