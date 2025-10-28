from __future__ import annotations

import math
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.adamw import AdamW as TorchAdamW
from torch.optim.optimizer import Optimizer

from mindspeed.core.optimizer.low_precision.fp8_simulator import (
    cast_with_fp8_guard,
    get_torch_npu,
)

torch_npu = get_torch_npu()


_FP8_QTYPE_MAX = {
    'fp8_e4m3': 448,
    'fp8_e5m2': 57344,
    'hif8_15': 15,
    'hif8_224': 224,
    'mxfp8': None,
}


class ScaleMeta:
    """Metadata container for precision-aware FP8 state tensors."""

    def __init__(self, qtype: str, state: Tensor, block_size: Optional[int] = None) -> None:
        if qtype not in _FP8_QTYPE_MAX:
            raise ValueError(f"Unsupported quantization type: {qtype}")

        self.qtype = qtype
        self.device = state.device
        self.fp8_max = _FP8_QTYPE_MAX[qtype]
        self.block_size = None
        self.scale: Optional[Tensor] = None
        self.scale_inv: Optional[Tensor] = None

        if qtype == 'mxfp8':
            # mx quant receives scale tensors from hardware runtime
            self.block_size = block_size
        else:
            if block_size is not None and block_size >= 16:
                self.block_size = block_size
                scale_len = math.ceil(state.numel() / block_size)
            else:
                self.block_size = None
                scale_len = 1
            self.scale = torch.ones(scale_len, device=self.device, dtype=torch.float32)
            self.scale_inv = 1 / self.scale

    def _clamp_to_quant_range(self, tensor: Tensor) -> Tensor:
        if self.fp8_max is None:
            return tensor
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        finfo = torch.finfo(tensor.dtype)
        max_val = torch.tensor(float(self.fp8_max), dtype=torch.float32, device=tensor.device)
        safe_max = (max_val * (1.0 - finfo.eps)).to(tensor.dtype)
        safe_min = -safe_max
        return torch.clamp(tensor, min=safe_min, max=safe_max)

    def quantization(self, fp32_tensor: Tensor) -> Tensor:
        if self.qtype == 'mxfp8':
            quant_tensor, sf = torch_npu.npu_dynamic_mx_quant(
                fp32_tensor.to(torch.bfloat16),
                block_size=self.block_size,
                dst_type=torch.float8_e4m3fn,
            )
            self.scale = sf
            self.scale_inv = 1 / self.scale
            return quant_tensor

        if fp32_tensor.dtype != torch.float32:
            fp32_tensor = fp32_tensor.to(torch.float32)
        amax_value = self.compute_amax(fp32_tensor)
        self.update_scale(amax_value)
        scaled = self.block_scaling(fp32_tensor, self.scale)
        scaled = self._clamp_to_quant_range(scaled)
        if self.qtype == 'fp8_e4m3':
            return cast_with_fp8_guard(scaled, torch.float8_e4m3fn, fallback_dtype=torch.float16)
        if self.qtype == 'fp8_e5m2':
            return cast_with_fp8_guard(scaled, torch.float8_e5m2, fallback_dtype=torch.float16)
        if self.qtype in {'hif8_15', 'hif8_224'}:
            return torch_npu.HiFloat8Tensor.to_hifloat8(scaled)
        raise ValueError(f"Unsupported quantization type: {self.qtype}")

    def dequantization(self, quant_tensor: Tensor) -> Tensor:
        if self.qtype == 'mxfp8':
            self.mxfp8_scale_convert()
        if self.qtype in {'mxfp8', 'fp8_e4m3', 'fp8_e5m2', 'hif8_15', 'hif8_224'}:
            dequant_tensor = quant_tensor.float()
        else:
            raise ValueError(f"Unsupported quantization type: {self.qtype}")

        if self.scale_inv is None:
            return dequant_tensor
        return self.block_scaling(dequant_tensor, self.scale_inv)

    def mxfp8_scale_convert(self) -> None:
        if self.scale_inv is None:
            return
        self.scale_inv = 2 ** (self.scale_inv - 127)
        self.scale_inv = self.scale_inv.view(-1)

    def block_scaling(self, inputs: Tensor, scale: Optional[Tensor]) -> Tensor:
        if scale is None:
            return inputs
        if self.block_size is None:
            return inputs * scale

        if inputs.numel() % self.block_size != 0:
            num_blocks = inputs.numel() // self.block_size
            large_num = num_blocks * self.block_size
            inputs_flat = inputs.view(-1)
            l_tensor, s_tensor = torch.split(
                inputs_flat,
                [large_num, inputs_flat.numel() - large_num],
                dim=0,
            )
            l_tensor = (l_tensor.view(-1, self.block_size) * scale[:-1].unsqueeze(1)).view(-1)
            s_tensor = s_tensor * scale[-1]
            inputs_flat = torch.cat([l_tensor, s_tensor])
        else:
            inputs_flat = inputs.view(-1, self.block_size) * scale.unsqueeze(1)
        return inputs_flat.view_as(inputs)

    def update_scale(self, amax: Tensor) -> None:
        if self.scale is None:
            return
        sf = self.fp8_max / amax
        sf = torch.where(amax > 0.0, sf, self.scale)
        sf = torch.where(torch.isfinite(amax), sf, self.scale)
        sf = torch.where(torch.isinf(sf), torch.full_like(sf, torch.finfo(amax.dtype).max), sf)
        self.scale.copy_(sf)
        self.scale_inv = 1 / self.scale

    def compute_amax(self, fp32_tensor: Tensor) -> Tensor:
        if self.block_size is not None:
            return fp32_tensor.view(-1, self.block_size).abs().max(dim=1).values
        return fp32_tensor.abs().max()

    def to_device(self, device: torch.device) -> None:
        if self.scale is not None:
            self.scale = self.scale.to(device)
        if self.scale_inv is not None:
            self.scale_inv = self.scale_inv.to(device)
        self.device = device




class HalfPrecisionMeta:
    """Metadata container for fp16/bf16 optimizer states with scaling support."""

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported half precision dtype: {dtype}")
        self.dtype = dtype
        self.device = device
        self.scale = torch.ones(1, dtype=torch.float32, device=device)
        self.scale_inv = torch.ones(1, dtype=torch.float32, device=device)
        self.max_range = None
        if dtype == torch.float16:
            self.max_range = torch.full((1,), torch.finfo(torch.float16).max / 2.0, dtype=torch.float32, device=device)

    def quantization(self, fp32_tensor: Tensor) -> Tensor:
        if self.dtype == torch.float16:
            absmax = fp32_tensor.detach().abs().max()
            scale = absmax / self.max_range
            scale = scale.to(torch.float32)
            scale = torch.where(scale > 0.0, scale, torch.zeros_like(scale))
            scale = scale.reshape(-1)
            self.scale = scale
            self.scale_inv = torch.where(scale > 0.0, scale.reciprocal(), torch.zeros_like(scale))
            scaled = fp32_tensor * self.scale_inv
            return scaled.to(torch.float16)
        self.scale.fill_(1.0)
        self.scale_inv.fill_(1.0)
        return fp32_tensor.to(torch.bfloat16)

    def dequantization(self, quant_tensor: Tensor) -> Tensor:
        tensor = quant_tensor.to(torch.float32)
        if self.dtype == torch.float16:
            return tensor * self.scale
        return tensor

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.scale = self.scale.to(device)
        self.scale_inv = self.scale_inv.to(device)
        if self.max_range is not None:
            self.max_range = self.max_range.to(device)

def cal_hcf(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x


def _allocate_state_tensor(spec: Union[str, torch.dtype], reference: Tensor, block_size: Optional[int]) -> Tensor:
    if isinstance(spec, torch.dtype):
        return torch.zeros_like(reference, dtype=spec, device=reference.device)

    if spec in {'fp16', 'bf16'}:
        meta = HalfPrecisionMeta(torch.float16 if spec == 'fp16' else torch.bfloat16, reference.device)
        base = torch.zeros_like(reference, dtype=torch.float32, device=reference.device)
        quant = meta.quantization(base)
        setattr(quant, 'meta', meta)
        return quant

    if block_size is None:
        block_size = 32 if spec == 'mxfp8' else cal_hcf(reference.numel(), 128)

    scale_meta = ScaleMeta(spec, reference, block_size)
    base = torch.zeros_like(reference, dtype=torch.float32, device=reference.device)
    quant = scale_meta.quantization(base)
    setattr(quant, 'meta', scale_meta)
    return quant


def _dequantize_tensor(tensor: Optional[Tensor]) -> Optional[Tensor]:
    if tensor is None:
        return None
    meta = getattr(tensor, 'meta', None)
    if meta is None:
        return tensor.to(torch.float32)
    return meta.dequantization(tensor)


def _requantize_tensor(storage_tensor: Optional[Tensor], tensor_fp32: Tensor) -> None:
    if storage_tensor is None:
        return
    meta = getattr(storage_tensor, 'meta', None)
    if meta is not None:
        storage_tensor.copy_(meta.quantization(tensor_fp32))
    else:
        storage_tensor.copy_(tensor_fp32.to(dtype=storage_tensor.dtype))


def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    step_tensor: Tensor,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    for i, param in enumerate(params):
        grad_tensor = grads[i]
        exp_avg_tensor = exp_avgs[i]
        exp_avg_sq_tensor = exp_avg_sqs[i]
        max_exp_avg_sq_tensor = max_exp_avg_sqs[i] if amsgrad else None

        grad_fp32 = _dequantize_tensor(grad_tensor)
        exp_avg_fp32 = _dequantize_tensor(exp_avg_tensor)
        exp_avg_sq_fp32 = _dequantize_tensor(exp_avg_sq_tensor)
        max_exp_avg_sq_fp32 = (
            _dequantize_tensor(max_exp_avg_sq_tensor)
            if amsgrad and max_exp_avg_sq_tensor is not None
            else None
        )
        master_fp32 = _dequantize_tensor(param)

        torch._fused_adamw_(
            [master_fp32],
            [grad_fp32],
            [exp_avg_fp32],
            [exp_avg_sq_fp32],
            [max_exp_avg_sq_fp32] if amsgrad and max_exp_avg_sq_fp32 is not None else [],
            [step_tensor],
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
        )

        _requantize_tensor(param, master_fp32)
        _requantize_tensor(exp_avg_tensor, exp_avg_fp32)
        _requantize_tensor(exp_avg_sq_tensor, exp_avg_sq_fp32)
        if amsgrad and max_exp_avg_sq_tensor is not None:
            _requantize_tensor(max_exp_avg_sq_tensor, max_exp_avg_sq_fp32)


class FusedTorchAdamW(TorchAdamW):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=False,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=True,
        )


class AdamW(Optimizer):
    _DTYPE_ALIAS = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp8': torch.uint8,
    }

    _EXP_AVG_ALLOWED = {'fp32', 'fp16', 'bf16', 'fp8', 'mxfp8', 'fp8_e4m3', 'hif8_15'}
    _EXP_AVG_SQ_ALLOWED = {'fp32', 'fp16', 'bf16', 'fp8', 'mxfp8', 'fp8_e4m3', 'fp8_e5m2', 'hif8_224'}

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(params, defaults)

        from megatron.training import get_args

        self.args = get_args()

    def _resolve_dtype(self, token: Union[str, torch.dtype], default: torch.dtype = torch.float32) -> torch.dtype:
        if isinstance(token, torch.dtype):
            return token
        if isinstance(token, str):
            return self._DTYPE_ALIAS.get(token.lower(), default)
        return default

    def _resolve_state_spec(self, token: Union[str, torch.dtype], is_exp_avg_sq: bool = False):
        if isinstance(token, torch.dtype):
            if token == torch.uint8:
                return 'fp8_e4m3'
            if token == torch.float16:
                return 'fp16'
            if token == torch.bfloat16:
                return 'bf16'
            return token
        if isinstance(token, str):
            token_lower = token.lower()
            if token_lower in self._DTYPE_ALIAS:
                resolved = self._DTYPE_ALIAS[token_lower]
                if resolved == torch.uint8:
                    return 'fp8_e4m3'
                if resolved in (torch.float16, torch.bfloat16):
                    return token_lower
                return resolved
            allowed = self._EXP_AVG_SQ_ALLOWED if is_exp_avg_sq else self._EXP_AVG_ALLOWED
            if token_lower in allowed:
                if token_lower == 'fp8':
                    return 'fp8_e4m3'
                return token_lower
        return torch.float32

    def _state_block_size(self, key: str) -> Optional[int]:
        return getattr(self.args, f'{key}_block_size', None)

    def _init_param_state(self, param: torch.nn.Parameter, amsgrad: bool = False):
        state = self.state[param]
        if state:
            return state

        master_dtype = self._resolve_dtype(getattr(self.args, 'main_params_dtype', torch.float32))
        master_param = param.detach().to(device=param.device, dtype=master_dtype).clone()
        state['master_param'] = master_param

        exp_avg_spec = self._resolve_state_spec(getattr(self.args, 'exp_avg_dtype', torch.float32))
        exp_avg_sq_spec = self._resolve_state_spec(
            getattr(self.args, 'exp_avg_sq_dtype', torch.float32),
            is_exp_avg_sq=True,
        )

        exp_avg_block = self._state_block_size('exp_avg')
        exp_avg_sq_block = self._state_block_size('exp_avg_sq')

        state['exp_avg'] = _allocate_state_tensor(exp_avg_spec, master_param, exp_avg_block)
        state['exp_avg_sq'] = _allocate_state_tensor(exp_avg_sq_spec, master_param, exp_avg_sq_block)
        if amsgrad:
            state['max_exp_avg_sq'] = torch.zeros_like(master_param, dtype=torch.float32, device=param.device)
        return state

    def initialize_state(self, param: torch.nn.Parameter):
        group = next((g for g in self.param_groups if param in g['params']), None)
        amsgrad = group['amsgrad'] if group is not None else False
        return self._init_param_state(param, amsgrad=amsgrad)

    def get_unscaled_state(self, param: torch.nn.Parameter, state_name: str) -> Tensor:
        state = self.initialize_state(param)
        if state_name == 'master_param':
            return state['master_param']
        tensor = state.get(state_name)
        if tensor is None:
            raise KeyError(f"State '{state_name}' not initialized for parameter")
        return _dequantize_tensor(tensor)

    def set_scaled_state(self, param: torch.nn.Parameter, state_name: str, tensor: Tensor) -> None:
        state = self.initialize_state(param)
        if state_name == 'master_param':
            state['master_param'].copy_(tensor.to(dtype=state['master_param'].dtype, device=param.device))
            return
        if state_name not in state:
            raise KeyError(f"State '{state_name}' not initialized for parameter")
        storage = state[state_name]
        tensor = tensor.to(torch.float32, device=param.device)
        _requantize_tensor(storage, tensor)

    def load_state_dict(self, state_dict):
        saved_groups = state_dict['param_groups']
        id_map = {
            id(param): param
            for param in chain.from_iterable(g['params'] for g in self.param_groups)
        }

        state = defaultdict(dict)
        for key, value in state_dict['state'].items():
            if key in id_map:
                param = id_map[key]
                group = next((g for g in self.param_groups if param in g['params']), None)
                amsgrad = group['amsgrad'] if group is not None else False
                self._init_param_state(param, amsgrad=amsgrad)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        self.set_scaled_state(param, sub_key, sub_value)
                    else:
                        self.state[param][sub_key] = sub_value
                state[param] = self.state[param]
            else:
                state[key] = value

        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(self.param_groups, saved_groups)]
        super().load_state_dict({'state': state, 'param_groups': param_groups})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            master_params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            if 'step' in group:
                group['step'] += 1
                if hasattr(group['step'], 'is_cpu') and group['step'].is_cpu:
                    group['step'] = group['step'].cuda()
            else:
                group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.cuda.current_device())

            for param in group['params']:
                grad_tensor = getattr(param, 'decoupled_grad', None)
                if grad_tensor is None:
                    grad_tensor = param.grad
                if grad_tensor is None:
                    continue
                if grad_tensor.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                params.append(param)
                grads.append(grad_tensor)

                state = self._init_param_state(param, amsgrad)
                master_params.append(state['master_param'])
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            if master_params:
                adamw(
                    master_params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    group['step'],
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                )

                for model_param, master_param in zip(params, master_params):
                    model_param.data.copy_(master_param.to(dtype=model_param.dtype))

        return loss

