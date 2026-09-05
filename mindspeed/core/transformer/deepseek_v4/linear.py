# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from megatron.training import get_args


class LinearNoTP(torch.nn.Linear):
    """Replicated linear used by DeepSeek-V4 compressor and indexer modules.

    Initialization is intentionally isolated from the global RNG stream:
    every layer is Xavier-filled from a fixed seed, then the caller's RNG is
    restored.  CP=1 and CP=2 construct the same LinearNoTP stack with different
    data-parallel sizes; consuming ``config.init_method`` from the global stream
    desynchronizes later Megatron layers and breaks first-step CP alignment.
    """

    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__(
            input_size,
            output_size,
            bias=kwargs.get("bias", True),
            dtype=config.params_dtype,
        )
        self.config = config
        current_seed = torch.random.initial_seed()
        torch.manual_seed(123)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.random.manual_seed(current_seed)
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)
        setattr(self.weight, "all_reduce", True)
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.pop(f"{prefix}_extra_state", None)
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        args = get_args()
        if getattr(args, "fp8", None) and input_.dtype != torch.float32:
            from mindspeed.te.pytorch.fp8.recipes import matmul_fp8

            output = matmul_fp8(input_, self.weight)
        else:
            output = torch.matmul(input_, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is not None:
            for key, value in quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + key] = value if keep_vars else value.detach()
