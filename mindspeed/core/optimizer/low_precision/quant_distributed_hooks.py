# Copyright (c) Microsoft Corporation.
# Copyright (c) 2024; NVIDIA CORPORATION.

from functools import wraps
import torch

from megatron.training import get_args


def collect_main_grad_data_for_unscaling_quant(self):
    main_grads = []
    seen_ids = set()
    for group in self.optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                if id(param.grad.data) not in seen_ids:
                    main_grads.append(param.grad.data)
                    seen_ids.add(id(param.grad.data))
            quant_grad = getattr(param, "quant_grad", None)
            if quant_grad is not None and id(quant_grad.data) not in seen_ids:
                main_grads.append(quant_grad.data)
                seen_ids.add(id(quant_grad.data))

    return main_grads


def copy_model_grads_to_main_grads_quant(self):
    args = get_args()

    def copy_group_grads(model_groups, shard_main_groups):
        for model_group, shard_main_group in zip(model_groups, shard_main_groups):
            for model_param, shard_main_param in zip(model_group, shard_main_group):
                param_range_map = self._get_model_param_range_map(model_param)
                param_range = param_range_map["param"]
                assert param_range.size == shard_main_param.nelement()

                model_grad = model_param.main_grad
                shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                if args.quant_grads:
                    # The DDP buffer is already FP16; keep the shard in FP16 for the
                    # optimizer without re-quantizing it. Reuse the buffer view to avoid
                    # an extra FP16 copy; this matches the original --quant-grads memory
                    # footprint while keeping FP16-native accumulation.
                    shard_main_param.quant_grad = shard_model_grad
                    shard_main_param.grad = None
                else:
                    shard_main_param.grad = shard_model_grad.float()

    # Copy model groups to shard groups.
    copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
    copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)


def ddp_make_backward_post_hook_wrapper(make_hook_func):
    @wraps(make_hook_func)
    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        # Use Megatron's native FP16 main_grad accumulation. This avoids the
        # dequantize -> add -> requantize round-trip on every micro-batch.
        return make_hook_func(self, param)

    return _make_backward_post_hook
