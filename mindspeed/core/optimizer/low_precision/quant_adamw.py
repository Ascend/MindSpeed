from typing import Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from copy import deepcopy
from itertools import chain
import math

import torch
import torch_npu
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW as TorchAdamW


def _infer_block_size(state: torch.Tensor, qtype: str) -> Optional[int]:
    if qtype == "mxfp8":
        return 32
    if state.ndim == 0:
        return None
    min_block = 32
    trailing = state.shape[-1]
    if trailing >= min_block:
        for candidate in (128, 96, 64, 48, 32):
            if trailing % candidate == 0:
                return candidate
        return trailing
    total = state.numel()
    if total >= min_block:
        return min_block
    return None


class ScaleMeta:
    def __init__(self, qtype, state, block_size=None):
        if qtype == "e4m3":
            self.fp8_max = 448
            self.qtype = 1
        elif qtype == "e5m2":
            self.fp8_max = 57344
            self.qtype = 2
        elif qtype == "hif8_15":
            self.fp8_max = 15
            self.qtype = 3
        elif qtype == "hif8_224":
            self.fp8_max = 224
            self.qtype = 3
        elif qtype == "mxfp8":
            self.fp8_max = None
            self.qtype = 4
        elif qtype == "fp16":
            self.fp8_max = 65503
            self.qtype = 5
        elif qtype == "bf16":
            self.fp8_max = torch.finfo(torch.bfloat16).max
            self.qtype = 6
        else:
            raise ValueError(f"Unsupported quantization type: {qtype}")
        if block_size is not None and block_size < 16:
            block_size = None
        self.block_size = block_size
        self.scale_codes = None
        self._scale_is_code = False
        if self.qtype == 4 and self.block_size is None:
            self.block_size = 32
        if self.block_size is not None:
            scale_len = math.ceil(state.numel() / self.block_size)
        else:
            scale_len = 1
        self.scale = torch.ones(scale_len, device=state.device, dtype=torch.float32)
        self.scale_inv = torch.empty_like(self.scale)
        self._refresh_scale_inverse()

    @staticmethod
    def _decode_mxfp8_codes(code_tensor: torch.Tensor):
        codes = torch.clamp(code_tensor.to(torch.float32).view(-1), min=0.0, max=254.0)
        scale_inv = torch.pow(2.0, codes - 127.0)
        scale_inv = torch.where(
            torch.isfinite(scale_inv) & (scale_inv > 0),
            scale_inv,
            torch.ones_like(scale_inv),
        )
        finfo = torch.finfo(scale_inv.dtype)
        max_safe_scale_inv = finfo.max / 512.0
        scale_inv = torch.clamp(scale_inv, max=max_safe_scale_inv)
        scale = torch.where(
            scale_inv > 0,
            1.0 / scale_inv,
            torch.ones_like(scale_inv),
        )
        scale = scale.view_as(code_tensor)
        scale_inv = scale_inv.view_as(code_tensor)
        return scale, scale_inv

    def quantization(self, fp32_tensor: torch.Tensor):
        if self.qtype == 4:
            # (scale_inv = 2 ** (code - 127)). Avoid heuristic candidates.
            quant_tensor, sf = torch_npu.npu_dynamic_mx_quant(
                fp32_tensor.to(torch.bfloat16),
                block_size=self.block_size if self.block_size is not None else 32,
                dst_type=torch.float8_e4m3fn,
            )
            device = fp32_tensor.device
            codes = sf.to(torch.float32)
            codes = torch.clamp(torch.round(codes), min=0.0, max=254.0)
            self.scale_codes = codes.to(device=device, dtype=torch.float32)
            self._scale_is_code = True
            scale, scale_inv = self._decode_mxfp8_codes(self.scale_codes)
            finfo = torch.finfo(scale_inv.dtype)
            max_safe_scale_inv = finfo.max / 512.0
            scale_inv = torch.clamp(scale_inv, max=max_safe_scale_inv)
            self.scale = scale.to(device=device, dtype=torch.float32)
            self.scale_inv = scale_inv.to(device=device, dtype=torch.float32)
            if self.block_size is None:
                self.block_size = 32
        else:
            if self.qtype in (5, 6):
                # BF16 has wide dynamic range; skip per-block scaling to avoid blowing up the scale.
                if isinstance(self.scale, torch.Tensor):
                    self.scale.fill_(1.0)
                else:
                    self.scale = 1.0
                if isinstance(self.scale_inv, torch.Tensor):
                    self.scale_inv.fill_(1.0)
                else:
                    self.scale_inv = 1.0
                target_dtype = torch.float16 if self.qtype == 5 else torch.bfloat16
                return fp32_tensor.to(target_dtype)
            amax_value = self.compute_amax(fp32_tensor)
            self.update_scale(amax=amax_value)
            if self.qtype == 3:
                if isinstance(self.scale, torch.Tensor):
                    self.scale.div_(self.fp8_max)
                else:
                    self.scale = self.scale / self.fp8_max
                if isinstance(self.scale_inv, torch.Tensor):
                    self.scale_inv.mul_(self.fp8_max)
                else:
                    self.scale_inv = self.scale_inv * self.fp8_max
            scaled_tensor = self.block_scaling(fp32_tensor, self.scale)
            self._refresh_scale_inverse()
            if self.qtype == 1:
                quant_tensor = scaled_tensor.to(torch.float8_e4m3fn)
            elif self.qtype == 2:
                quant_tensor = scaled_tensor.to(torch.float8_e5m2)
            elif self.qtype == 3:
                quant_tensor = torch_npu.HiFloat8Tensor.to_hifloat8(scaled_tensor.contiguous())
            elif self.qtype == 5:
                quant_tensor = scaled_tensor.to(torch.float16)
            elif self.qtype == 6:
                quant_tensor = scaled_tensor.to(torch.bfloat16)
            else:
                raise ValueError(f"Unsupported quantization type: {self.qtype}")
        return quant_tensor

    def dequantization(self, quant_tensor: torch.Tensor):
        dequant_tensor = quant_tensor.float()
        if self.scale_inv is None:
            return dequant_tensor
        return self.block_scaling(dequant_tensor, self.scale_inv)

    def mxfp8_scale_convert(self):
        if not self._scale_is_code or self.scale_codes is None:
            return
        device = self.scale.device if self.scale is not None else (
            self.scale_inv.device if self.scale_inv is not None else self.scale_codes.device
        )
        scale, scale_inv = self._decode_mxfp8_codes(self.scale_codes.to(device=device))
        self.scale = scale.to(device=device, dtype=torch.float32)
        self.scale_inv = scale_inv.to(device=device, dtype=torch.float32)

    def _refresh_scale_inverse(self):
        if self.qtype == 4:
            if self._scale_is_code:
                if self.scale_codes is None:
                    return
                device = self.scale.device if self.scale is not None else (
                    self.scale_inv.device if self.scale_inv is not None else self.scale_codes.device
                )
                scale, scale_inv = self._decode_mxfp8_codes(self.scale_codes.to(device=device))
                self.scale = scale.to(device=device, dtype=torch.float32)
                self.scale_inv = scale_inv.to(device=device, dtype=torch.float32)
            else:
                target_device = None
                if self.scale_inv is not None:
                    target_device = self.scale_inv.device
                elif self.scale is not None:
                    target_device = self.scale.device
                if self.scale is not None:
                    self.scale = self.scale.to(
                        device=target_device if target_device is not None else self.scale.device,
                        dtype=torch.float32,
                    )
                if self.scale_inv is not None:
                    self.scale_inv = self.scale_inv.to(
                        device=target_device if target_device is not None else self.scale_inv.device,
                        dtype=torch.float32,
                    )
            return
        if self.scale is None:
            self.scale_inv = None
            return
        if self.scale_inv is None or self.scale_inv.shape != self.scale.shape or self.scale_inv.device != self.scale.device:
            self.scale_inv = torch.empty_like(self.scale)
        safe_scale = torch.where(self.scale == 0, torch.ones_like(self.scale), self.scale)
        torch.reciprocal(safe_scale, out=self.scale_inv)
        self.scale_inv = torch.where(self.scale == 0, torch.zeros_like(self.scale), self.scale_inv)

    def block_scaling(self, inputs: torch.Tensor, scale: torch.Tensor):
        if scale is None:
            return inputs
        if self.block_size is not None:
            scale_flat = scale.reshape(-1)
            if scale_flat.numel() == 0:
                return inputs
            if inputs.numel() % self.block_size != 0:
                num_blocks = inputs.numel() // self.block_size
                large_num = num_blocks * self.block_size
                inputs_flatten = inputs.view(-1)
                l_tensor, s_tensor = torch.split(
                    inputs_flatten, [large_num, inputs_flatten.numel() - large_num], dim=0
                )
                l_tensor = (l_tensor.view(-1, self.block_size) * scale_flat[:-1].unsqueeze(1)).view(-1)
                s_tensor = s_tensor * scale_flat[-1]
                inputs_flatten = torch.cat([l_tensor, s_tensor])
            else:
                inputs_flatten = inputs.view(-1, self.block_size) * scale_flat.unsqueeze(1)
            inputs = inputs_flatten.view(inputs.shape)
        else:
            inputs = inputs * scale
        return inputs

    def update_scale(self, amax: torch.Tensor):
        sf = self.fp8_max / amax
        sf = torch.where(amax > 0.0, sf, self.scale)
        sf = torch.where(torch.isfinite(amax), sf, self.scale)
        sf = torch.where(torch.isinf(sf), torch.full_like(sf, torch.finfo(amax.dtype).max), sf)
        self.scale.copy_(sf)
        self._refresh_scale_inverse()

    def compute_amax(self, fp32_tensor: torch.Tensor):
        if self.block_size is not None:
            amax_value = fp32_tensor.view(-1, self.block_size).abs().max(dim=1).values
        else:
            amax_value = fp32_tensor.abs().max()
        return amax_value

    def to_device(self, device):
        if self.scale is not None:
            self.scale = self.scale.to(device)
        if self.scale_inv is not None:
            self.scale_inv = self.scale_inv.to(device)
        if self.scale_codes is not None:
            self.scale_codes = self.scale_codes.to(device)


def _dequantize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "meta") and tensor.meta is not None:
        return tensor.meta.dequantization(tensor.data)
    if tensor.dtype != torch.float32:
        return tensor.to(torch.float32)
    return tensor


def _requantize_tensor(storage_tensor: torch.Tensor, tensor_fp32: torch.Tensor):
    if hasattr(storage_tensor, "meta") and storage_tensor.meta is not None:
        storage_tensor.data.copy_(storage_tensor.meta.quantization(tensor_fp32.data))
    else:
        if storage_tensor.dtype != tensor_fp32.dtype:
            storage_tensor.copy_(tensor_fp32.to(dtype=storage_tensor.dtype))
        else:
            storage_tensor.copy_(tensor_fp32)


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
):
    for i, param in enumerate(params):
        grad_tensor = grads[i]
        exp_avg_tensor = exp_avgs[i]
        exp_avg_sq_tensor = exp_avg_sqs[i]
        max_exp_avg_sq_tensor = max_exp_avg_sqs[i] if amsgrad else None

        grad_fp32 = _dequantize_tensor(grad_tensor)
        exp_avg_fp32 = _dequantize_tensor(exp_avg_tensor)
        exp_avg_sq_fp32 = _dequantize_tensor(exp_avg_sq_tensor)
        max_exp_avg_sq_fp32 = None
        if amsgrad and max_exp_avg_sq_tensor is not None:
            max_exp_avg_sq_fp32 = _dequantize_tensor(max_exp_avg_sq_tensor)

        torch._fused_adamw_(
            [param],
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
    ):
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
    }

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
            **kwargs,
    ):
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

    def _resolve_dtype(self, dtype_value):
        if isinstance(dtype_value, torch.dtype):
            return dtype_value
        if isinstance(dtype_value, str):
            return self._DTYPE_ALIAS.get(dtype_value, torch.float32)
        return torch.float32

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    def _get_state_tensor(self, state: torch.Tensor, qtype: str):
        if qtype != "fp32":
            block_size = _infer_block_size(state, qtype)
            scale_meta = ScaleMeta(qtype, state, block_size)
            quant_state = scale_meta.quantization(state.data)
            quant_state.meta = scale_meta
            return quant_state
        return state

    def _get_state_qtype(self, param: torch.nn.Parameter):
        if hasattr(param, "keep_fp32"):
            return "fp32", "fp32"
        if self.args.quant_states == "fp8":
            return "e4m3", "e5m2"
        if self.args.quant_states == "hif8":
            return "hif8_15", "hif8_224"
        if self.args.quant_states == "mxfp8":
            return "mxfp8", "mxfp8"
        if self.args.quant_states == "fp16":
            return "fp16", "fp16"
        return "fp32", "fp32"

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        groups = self.param_groups
        saved_groups = deepcopy(state_dict['param_groups'])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group that doesn't match optimizer")

        id_map = dict(
            zip(chain.from_iterable(g['params'] for g in saved_groups),
                chain.from_iterable(g['params'] for g in groups))
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            if isinstance(value, torch.Tensor):
                if hasattr(value, "meta") and value.meta is not None:
                    if not self.args.quant_states:
                        value = value.meta.dequantization(value.data)
                    value_device = value.to(device=param.device)
                    if self.args.quant_states:
                        value_device.meta = value.meta
                        value_device.meta.to_device(param.device)
                else:
                    value_device = value.to(device=param.device)
                    if key == "master_param":
                        master_dtype = self._resolve_dtype(getattr(self.args, "main_params_dtype", torch.float32))
                        if value_device.dtype != master_dtype:
                            value_device = value_device.to(master_dtype)
                    exp_avg_qtype, exp_avg_sq_qtype = self._get_state_qtype(param)
                    if key == "exp_avg":
                        value_device = self._get_state_tensor(value_device, exp_avg_qtype)
                    if key == "exp_avg_sq":
                        value_device = self._get_state_tensor(value_device, exp_avg_sq_qtype)
                return value_device
            if isinstance(value, dict):
                return {
                    k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k)
                    for k, v in value.items()
                }
            if isinstance(value, Iterable):
                return type(value)(
                    _cast(param, v, param_id=param_id, param_groups=param_groups)
                    for v in value
                )
            return value

        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(param, v, param_id=k, param_groups=state_dict['param_groups'])
            else:
                state[k] = v

        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            model_params = []
            master_params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            if 'step' in group:
                group['step'] += 1
                if hasattr(group['step'], "is_cpu") and group['step'].is_cpu:
                    group['step'] = group['step'].cuda()
            else:
                group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.cuda.current_device())

            for p in group['params']:
                grad_tensor = None
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grad_tensor = p.grad
                elif hasattr(p, "quant_grad") and p.quant_grad is not None:
                    if p.quant_grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grad_tensor = p.quant_grad
                elif hasattr(p, "decoupled_grad") and p.decoupled_grad is not None:
                    if p.decoupled_grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grad_tensor = p.decoupled_grad
                if grad_tensor is None:
                    continue

                model_params.append(p)
                grads.append(grad_tensor)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    exp_avg_qtype, exp_avg_sq_qtype = self._get_state_qtype(p)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = self._get_state_tensor(
                        torch.zeros_like(p, memory_format=torch.preserve_format), exp_avg_qtype)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = self._get_state_tensor(
                        torch.zeros_like(p, memory_format=torch.preserve_format), exp_avg_sq_qtype)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            # if master_params:
            adamw(
                model_params,
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

        return loss