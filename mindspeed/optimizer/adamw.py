from typing import List, Optional, Tuple, Union, Iterable
from copy import deepcopy
from collections import defaultdict
from itertools import chain
import math
import torch
import torch_npu
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW as TorchAdamW


class ScaleMeta:
    def __init__(self, qtype, state, block_size):
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
        else:
            raise ValueError("Unsupported quantization type: {}".format(qtype))
        if block_size < 16:
            block_size = state.numel()
        scale_len = math.ceil(state.numel() / block_size)
        if self.qtype != 4:
            self.scale = torch.ones(scale_len, device=state.device)
            self.scale_inv = 1 / self.scale
        else:
            self.scale = None
            self.scale_inv = None
        self.block_size = block_size

    def quantization(self, fp32_tensor):
        if self.qtype == 4:
            quant_tensor, sf = torch_npu.npu_dynamic_mx_quant(fp32_tensor, block_size=self.block_size,
                                                              dst_type=torch.float8_e4m3fn)
            self.scale = sf
            self.scale_inv = 1 / self.scale
        else:
            amax_value = self.compute_amax(fp32_tensor)
            self.update_scale(amax=amax_value)
            fp32_tensor = self.block_scaling(fp32_tensor, self.scale)

            if self.qtype == 1:
                quant_tensor = fp32_tensor.to(torch.float8_e4m3fn)
            elif self.qtype == 2:
                quant_tensor = fp32_tensor.to(torch.float8_e5m2)
            elif self.qtype == 3:
                quant_tensor = torch_npu.HiFloat8Tensor.to_hifloat8(fp32_tensor)
            elif self.qtype == 5:
                quant_tensor = fp32_tensor.to(torch.float16)
            else:
                raise ValueError("Unsupported quantization type: {}".format(self.qtype))
        return quant_tensor

    def dequantization(self, quant_tensor):
        if self.qtype == 4:
            self.mxfp8_scale_convert()
        dequant_tensor = quant_tensor.float()
        dequant_tensor = self.block_scaling(dequant_tensor, self.scale_inv)
        return dequant_tensor

    def mxfp8_scale_convert(self):
        self.scale = 2 ** (self.scale - 127)
        self.scale = self.scale.view(-1)

    def block_scaling(self, inputs, scale):
        inputs_flatten = inputs.view(-1, self.block_size) * scale.unsqueeze(1)
        inputs = inputs_flatten.view(inputs.shape)
        return inputs

    def update_scale(self, amax=None):
        sf = self.fp8_max / amax
        sf = torch.where(amax > 0.0, sf, self.scale)
        sf = torch.where(torch.isfinite(amax), sf, self.scale)
        sf = torch.where(torch.isinf(sf), torch.full_like(sf, torch.finfo(amax.dtype).max), sf)
        self.scale.copy_(sf)
        self.scale_inv = 1 / self.scale

    def compute_amax(self, fp32_tensor: torch.Tensor):
        amax_value = fp32_tensor.view(-1, self.block_size).abs().max(dim=1).values
        return amax_value

    def to_device(self, device):
        self.scale = self.scale.to(device)
        self.scale_inv = self.scale_inv.to(device)


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          step: int,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        if hasattr(grads[i], "meta"):
            grad = grads[i].meta.dequantization(grads[i].data)
        else:
            grad = grads[i]
        if hasattr(exp_avgs[i], "meta"):
            exp_avg = exp_avgs[i].meta.dequantization(exp_avgs[i].data)
        else:
            exp_avg = exp_avgs[i]
        if hasattr(exp_avg_sqs[i], "meta"):
            exp_avg_sq = exp_avg_sqs[i].meta.dequantization(exp_avg_sqs[i].data)
        else:
            exp_avg_sq = exp_avg_sqs[i]

        # Perform stepweight decay
        bias_correction1 = beta1 ** (step - 1)
        bias_correction2 = beta2 ** (step - 1)

        param.data, exp_avg, exp_avg_sq = torch_npu.npu_apply_adam_w(
            bias_correction1,
            bias_correction2,
            lr,
            weight_decay,
            beta1,
            beta2,
            eps,
            grad,
            None,
            amsgrad,
            maximize,
            out=(param.data, exp_avg, exp_avg_sq)
        )

        if hasattr(exp_avgs[i], "meta"):
            exp_avgs[i].data.copy_(exp_avgs[i].meta.quantization(exp_avg.data))
        if hasattr(exp_avg_sqs[i], "meta"):
            exp_avg_sqs[i].data.copy_(exp_avg_sqs[i].meta.quantization(exp_avg_sq.data))


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
    ):
        super().__init__(params, 
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                foreach=False,
                maximize=maximize,
                capturable=False,
                differentiable=False,
                fused=True,)


def cal_hcf(x, y):
    """calculate the highest common factor"""
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller + 1):
        if ((x % i == 0) and (y % i == 0)):
            res = i
    return res


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(AdamW, self).__init__(params, defaults)
        from megatron.training import get_args
        self.args = get_args()

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    def _get_state_tensor(self, state, qtype):
        if qtype != "fp32":
            if qtype == "mxfp8":
                block_size = 32
            else:
                block_size = cal_hcf(state.numel(), 128)
            scale_meta = ScaleMeta(qtype, state, block_size)
            state = scale_meta.quantization(state.data)
            state.meta = scale_meta
        return state

    def _get_state_qtype(self, param):
        """get quantization type of state1 and state2."""
        if hasattr(param, "keep_fp32"):
            return "fp32", "fp32"
        if self.args.quant_states == "fp8":
            return "e4m3", "e5m2"
        elif self.args.quant_states == "hif8":
            return "hif8_15", "hif8_224"
        elif self.args.quant_states == "mxfp8":
            return "mxfp8", "mxfp8"
        else:
            return "fp32", "fp32"

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict['param_groups'])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = dict(zip(chain.from_iterable(g['params'] for g in saved_groups),
                      chain.from_iterable(g['params'] for g in groups)))

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                if hasattr(value, "meta"):
                    if not self.args.quant_states:
                        value = value.meta.dequantization(value.data)  # remove meta
                    value_device = value.to(device=param.device)
                    if self.args.quant_state:
                        value_device.meta = value.meta
                        value_device.meta.to_device(param.device)  # meta to device
                else:
                    value_device = value.to(device=param.device)
                    exp_avg_qtype, exp_avg_sq_qtype = self._get_state_qtype(param)
                    if key == "exp_avg":
                        value_device = self._get_state_tensor(value_device, exp_avg_qtype)
                    if key == "exp_avg_sq":
                        value_device = self._get_state_tensor(value_device, exp_avg_sq_qtype)
                return value_device
            elif isinstance(value, dict):
                return {k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k) for k, v in value.items()}
            elif isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)  # type: ignore[call-arg]
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(param, v, param_id=k, param_groups=state_dict['param_groups'])
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group['params']:
                if p.grad is None and not hasattr(p, "quant_grad"):
                    continue
                params_with_grad.append(p)
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)
                else:
                    if p.quant_grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.quant_grad)

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

            adamw(params_with_grad,
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
                  maximize=group['maximize'])

        return loss
