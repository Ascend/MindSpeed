import logging
from collections import abc, defaultdict
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.cuda import FloatTensor
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup


log = logging.getLogger(__name__)


def _refresh_per_optimizer_state():
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _check_supported_device(tensor: torch.Tensor):
    if tensor.is_cuda or tensor.device.type in ("xla", "cpu", "npu"):
        return
    raise AssertionError(f"Unsupported device: {tensor.device}")


class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    """
    Lazily serves tensor to request device. This class extends
    _MultiDeviceReplicator to allow support for "cpu" as a device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        _check_supported_device(master_tensor)
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}


class ShardedGradScaler(GradScaler):

    def __init__(
        self,
        init_scale: float = 2.0**16,
        min_scale: float = 1.,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        hysteresis: int = 2,
        enabled: bool = True,
        process_group: Optional[ProcessGroup] = dist.group.WORLD,
    ):
        if init_scale is None:
            init_scale = 1.0
        super().__init__(
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(
                _refresh_per_optimizer_state)
        self.device = torch.device("cuda")
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    @property
    def loss_scale(self) -> torch.Tensor:
        '''
        The scaler's scale is lazily initialized, or None if _lazy_init_scale_growth_tracker is not used
        Initialization is only done when scale() is called for the first time

        But megatronOptimizer doesn't scale directly, but manually scales loss
        '''
        if not self._enabled:
            return torch.tensor([1.0], dtype=torch.float32, device=self.device)
        elif self._scale is None:
            self._lazy_init_scale_growth_tracker(self.device)
            self._check_none_scale()
        return self._scale

    def scale(
        self, outputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self._enabled:
            return outputs

        if isinstance(outputs, torch.Tensor):
            _check_supported_device(outputs)
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            self._check_none_scale()
            scaled_output = outputs * self._scale.to(
                device=outputs.device, non_blocking=True
            )
            # Here we ensure the return dtype is the same as the outputs dtype.
            # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
            # format (fp16, bf16) and so the scaled loss should be of the same dtype.
            return scaled_output.type(outputs.dtype)

        stash: List[_GeneralMultiDeviceReplicator] = []

        def apply_scale(
            val: Union[torch.Tensor, abc.Iterable]
        ) -> Union[torch.Tensor, abc.Iterable]:
            if isinstance(val, torch.Tensor):
                _check_supported_device(val)
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    self._check_none_scale()
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                scaled_val = val * stash[0].get(val.device)

                return scaled_val.type(val.dtype)
            elif isinstance(val, abc.Iterable):
                iterator = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterator)
                else:
                    return iterator
            else:
                raise ValueError(
                    "outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)  # type: ignore[return-value]

    def _foreach_non_finite_check_and_unscale_cpu_(
        self, grads: List, found_inf: torch.Tensor, inv_scale: torch.Tensor
    ) -> None:
        if len(grads) == 0:
            return
        if inv_scale.numel() != 1:
            raise ValueError("inv_scale must be a 1-element tensor.")
        if found_inf.numel() != 1:
            raise ValueError("found_inf must be a 1-element tensor.")

        for grad in grads:
            if grad.device.type != "cpu":
                log.error(
                    "tensor device is %s but was expected to be ``cpu``",
                    grad.device,
                )
                raise ValueError(
                    "Gradients were found on a non-CPU device when"
                    " expected to be on CPU."
                )
            if (
                torch.isinf(grad).any().item() is True
                or torch.isnan(grad).any().item() is True
            ):
                found_inf.data = torch.tensor([1.0])
                break
            else:
                grad.data *= inv_scale.item()

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    ) -> Dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
        per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)

        per_device_and_dtype_grads = defaultdict(
            lambda: defaultdict(list))
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError(
                            "Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        if param.grad.dtype is torch.float16:
                            # coalesce is not supported in torch.float16
                            param_grad_fp32 = param.grad.type(
                                torch.float32).coalesce()
                            param.grad = param_grad_fp32.type(torch.float16)
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    if grads[0].device.type == "cpu":
                        self._foreach_non_finite_check_and_unscale_cpu_(
                            grads,
                            per_device_found_inf.get(device),
                            per_device_inv_scale.get(device),
                        )
                    else:
                        torch._amp_foreach_non_finite_check_and_unscale_(
                            grads,
                            per_device_found_inf.get(device),
                            per_device_inv_scale.get(device),
                        )
        # There exist contexts (e.g. w/ `use_orig_params=True`) wherein some
        # ranks may have no (non-zero sized) parameter shards, necessitating the
        # initialization of `per_device_found_inf._per_device_tensors` here
        if not per_device_found_inf._per_device_tensors:
            self._check_none_scale()
            per_device_found_inf.get(self._scale.device)
        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._enabled:
            return False

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        self._check_none_scale()
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full(
            (1,), 0.0, dtype=torch.float32, device=self._scale.device
        )

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, True
        )
        optimizer_state["stage"] = OptState.UNSCALED

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        future_handles = []

        for v in optimizer_state["found_inf_per_device"].values():
            if v.device.type == "cpu":
                v_on_cuda = v.cuda()
                future_handles.append(
                    dist.all_reduce(
                        v_on_cuda, async_op=True, group=self.process_group
                    ).get_future()
                )
                v.copy_(v_on_cuda.cpu())
            else:
                future_handles.append(
                    dist.all_reduce(
                        v, async_op=True, group=self.process_group
                    ).get_future()
                )

        # Make sure that the calls are done before moving out.
        if future_handles:
            torch.futures.wait_all(future_handles)

        if (
            len(optimizer_state["found_inf_per_device"]) == 0
        ):
            raise AssertionError("No inf checks were recorded for this optimizer.")

        found_inf = sum(v.item()
                        for v in optimizer_state["found_inf_per_device"].values())
        return found_inf > 0.

    def step(
        self, optimizer: torch.optim.Optimizer, *args, **kwargs
    ) -> Optional[float]:
        return super().step(optimizer, *args, **kwargs)

    def _update_scale(self, found_inf) -> None:
        """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
        if found_inf.item() >= 1.0:
            self._scale *= self._backoff_factor  # type: ignore[arg-type]
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(
                    self._scale * self.backoff_factor, self.min_scale)
        else:
            successful = self._growth_tracker + 1  # type: ignore[operator]
            if successful == self._growth_interval:  # type: ignore[arg-type]
                self._scale *= self._growth_factor  # type: ignore[arg-type]
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
            else:
                self._growth_tracker = successful

    def update(self, new_scale: Optional[Union[float, FloatTensor]] = None) -> None:
        """
        Updates the scale factor.
        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.
        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)
        Args:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.
        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """

        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker(
            "update")  # type: ignore[var-annotated]

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                if not (isinstance(new_scale, torch.cuda.FloatTensor) and (new_scale.numel() == 1) and not new_scale.requires_grad):
                    raise AssertionError("new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False.")
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            if len(found_infs) == 0:
                raise AssertionError("No inf checks were recorded prior to update.")

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            self._update_scale(found_inf_combined)

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def _meg_step(self, optimizer, *args, **kwargs):
        '''Split the optional step with unscale for adapted with megatron
        In between we can insert other operations like clip grad
        '''
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        retval = self._maybe_opt_step(
            optimizer, optimizer_state, *args, **kwargs)
        optimizer_state["stage"] = OptState.STEPPED
        return retval

    def state_dict(self):
        state_dict = {}
        state_dict['scale'] = self._scale
        state_dict['growth_tracker'] = self._growth_tracker
        state_dict['hysteresis_tracker'] = self._hysteresis_tracker
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']
        
    def _check_none_scale(self):
        if self._scale is None:
            raise AssertionError("Got none scale")