import contextlib
import functools
import logging
import os
import warnings
from itertools import accumulate, chain
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    no_type_check,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _DeviceHandle,
    _named_parameters_with_duplicates,
    _no_dispatch_record_stream,
    _set_zero3_flattened,
    HandleTrainingState,
)

__all__ = [
    "FlatParameter",
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "ParamInfo",
    "SharedParamInfo",
    "HandleShardingStrategy",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


"""
[Note: Fully Sharded Module]
We define the "fully sharded module" to be the original ``nn.Module`` that owns
a ``FlatParamHandle``. It is the *single* module logically responsible for the
*single* unshard/reshard pair for the handle's ``FlatParameter`` for a given
forward or backward pass. The fully sharded module should be passed to the
``FlatParamHandle`` constructor.

For the wrapper code path:
- The ``FullyShardedDataParallel`` module wrapping the fully sharded module
runs the unshard/reshard on behalf of the ful+ly sharded module by overriding
``nn.Module.forward``.
- The fully sharded module is exactly the module passed to the
``FullyShardedDataParallel`` constructor's ``module`` argument.

For the non-wrapper code path:
- Hooks registered on the fully sharded module run the unshard/reshard.
- The fully sharded module may either be the direct argument to ``fully_shard``
or a submodule chosen by the provided wrapping policy.
"""

# We should use 'safe' by default since it respects method overrides, but for
# special cases such as for high CPU overhead or for intentionally bypassing
# checks in the overrides, we may use 'unsafe'.
_FSDP_USE_UNSAFE_SETATTR = "FSDP_USE_UNSAFE_SETATTR"

# Some value to set padding in tensors to for debuggability
_FLAT_PARAM_PADDING_VALUE = 42


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class SharedParamInfo(NamedTuple):
    """
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    """

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str
    prim_param_name: str  # unprefixed
    prim_module: nn.Module
    prim_module_name: str


class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""

    in_shard: bool
    # Use to index into the sharded flat parameter, e.g.
    # `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: Optional[int]
    numel_in_shard: Optional[int]
    # Use to get part of the parameter in the local shard from a flattened
    # version of the unsharded parameter, e.g.
    # `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: Optional[int]
    intra_param_end_idx: Optional[int]  # inclusive
    # `unshard_data [flat_param_start_idx : flat_param_end_idx]`
    flat_param_start_idx: Optional[int] = None
    flat_param_end_idx: Optional[int] = None # inclusive


class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flat parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original parameter.
    """

    param_names: Tuple[str, ...]
    param_shapes: Tuple[torch.Size, ...]
    param_numels: Tuple[int, ...]
    param_offsets: Tuple[Tuple[int, int], ...]


class _FlatParameterMeta(_ParameterMeta):
    # Make `isinstance(t, FlatParameter)` return True for custom tensor
    # instances that have the _is_flat_param flag for BC
    def __instancecheck__(self, instance):
        # NB: do NOT test the super implementation
        return isinstance(instance, torch.Tensor) and getattr(
            instance, "_is_flat_param", False
        )


class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: Tuple[ParamInfo, ...]
    _shapes: Tuple[torch.Size, ...]
    _fqns: Tuple[str, ...]
    _numels_with_padding: Tuple[int, ...]
    _numels: Tuple[int, ...]
    _shard_param_infos: Tuple[_ShardParamInfo, ...]
    _shared_param_infos: Tuple[SharedParamInfo, ...]
    _modules: Set[nn.Module]
    _shard_numel_padded: int
    _zero1_shard: Tensor
    _zero3_shard: Tensor
    _full_param_padded: Tensor
    _full_grad_padded: Tensor
    _full_prec_grad_padded: Tensor
    _post_backward_hook_state: Tuple[Any, Any]
    _saved_grad: Tensor
    _params: Optional[List[nn.Parameter]]
    _shared_params: Optional[List[nn.Parameter]]
    _tensors: Optional[List[Optional[Tensor]]]
    _is_grad_none_mask: Optional[List[bool]]
    _is_padding_mask: List[bool]
    _cpu_grad: Tensor = None

    def __new__(cls, data=None, requires_grad=True):
        if cls is not FlatParameter:
            raise ValueError("subclasses FlatParameter not supported")
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)  # type: ignore[call-arg]
        r._is_flat_param = True  # type: ignore[attr-defined]
        return r

    # NB: This is not a regular method, because FlatParameters are not actually
    # instances of this class (see __new__ above).  So you must indirectly
    # call this directly through the classmethod.
    @classmethod
    def _init_metadata(
        cls,
        self,
        param_infos: List[ParamInfo],
        numels: List[int],
        shapes: List[torch.Size],
        fqns: List[str],
        shared_param_infos: List[SharedParamInfo],
        params: Optional[List[nn.Parameter]],
        shared_params: Optional[List[nn.Parameter]],
        is_padding_mask: List[bool],
    ) -> None:
        """
        Initializes attributes holding metadata about the original parameters
        comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.

        Args:
            See the Attributes in the class docstring.
        """
        if len(param_infos) != len(shapes) or len(param_infos) != len(fqns):
            raise ValueError("Incorrect number of param_infos")
            
        self._num_params = len(param_infos)
        self._param_infos = param_infos
        self._shapes = shapes
        self._fqns = fqns
        self._is_padding_mask = is_padding_mask

        numels_without_padding: List[int] = []
        for numel, is_padding in zip(numels, is_padding_mask):
            if not is_padding:
                numels_without_padding.append(numel)
        self._numels = tuple(numels_without_padding)
        self._numels_with_padding = tuple(numels)
        if len(self._numels) != self._num_params:
            raise AssertionError("self._numels do not match num_param")

        self._shared_param_infos = tuple(shared_param_infos)
        self._modules = {pi.module for pi in self._param_infos}.union(
            {spi.module for spi in self._shared_param_infos}
        )
        if (params is None) != (shared_params is None):
            raise AssertionError("Param and Shared_param should be both None or non-None")
        if params is not None:
            if len(shared_params) != len(shared_param_infos):
                raise AssertionError("shared_params do not match shared_param_infos")
            self._params = []
            for param, is_padding in zip(params, is_padding_mask):
                if not is_padding:
                    self._params.append(param)
            self._shared_params = shared_params
            # Mark the original parameters to avoid flattening them into
            # another `FlatParameter` during recursive construction
            for param in chain(self._params, self._shared_params):
                _set_zero3_flattened(param)
            self._is_grad_none_mask = [False for _ in range(self._num_params)]
            self._tensors = [None for _ in range(self._num_params)]
        else:
            self._params = None
            self._shared_params = None
            self._is_grad_none_mask = None
            self._tensors = None
        self._unpadded_unsharded_size = self.size()
        _set_zero3_flattened(self)
        # Tracks whether the `FlatParameter`'s post-backward hook has been
        # called to modify the behavior of the post-backward callback
        self._post_backward_called = False


class FlatParamHandle:
    ##################
    # INITIALIZATION #
    ##################
    def __init__(
        self,
        params: Sequence[Union[nn.Parameter, Tensor]],
        zero3_module: nn.Module,
        device: torch.device,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
        zero3_process_group: dist.ProcessGroup,
        zero1_process_group: dist.ProcessGroup,
        offload_grads: bool = False
    ):
        self.initialize(params,
                        zero3_module,
                        device=device,
                        mp_param_dtype=mp_param_dtype,
                        mp_reduce_dtype=mp_reduce_dtype,
                        zero3_process_group=zero3_process_group,
                        zero1_process_group=zero1_process_group,
                        offload_grads=offload_grads
                        )
        self._init_flat_param_and_metadata(
            params, zero3_module, self._aligned_numel, self.zero1_world_size  # type: ignore[arg-type]
        )
        self._use_unsharded_views(as_params=False)

        
    def initialize(
        self,
        params: Sequence[Union[nn.Parameter, Tensor]],
        zero3_module: nn.Module,
        device: torch.device,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
        zero3_process_group: dist.ProcessGroup,
        zero1_process_group: dist.ProcessGroup,
        offload_grads: bool = False
    ):
        params = list(params)
        if len(params) == 0:
            raise ValueError(
                f"Cannot construct a {self.__class__.__name__} with an empty parameter list"
            )
        self._init_setattr_fns()
        align_addresses = True
        self._init_get_unflat_views_fn(align_addresses)
        self.device = device
        self._device_handle = _DeviceHandle.from_device(self.device)
        self.zero3_process_group = zero3_process_group
        self.zero1_process_group = zero1_process_group
        self.zero1_world_size = zero1_process_group.size()
        self.zero1_group_rank = zero1_process_group.rank()
        self.zero3_group_rank = zero3_process_group.rank()
        self.zero3_group_size = zero3_process_group.size()
        self._training_state = HandleTrainingState.IDLE
        self._debug_level = dist.get_debug_level()
        self._zero3_module = zero3_module
        # For strategies that do not free after forward, we skip using sharded
        # views after forward since the unsharded data exists. We still switch
        # `self.flat_param` to point to the sharded flat parameter since what
        # it points to parameterizes behavior. We use the following attribute
        # to track which tensor data the parameters are unsharded views into.
        self._unsharded_flat_param_for_skipped_views: Optional[Tensor] = None
        # The index in the state's `all_handles`, which must be the
        # same across ranks for the execution order validation to work
        self._handle_index: Optional[int] = None
        # Index in handles_to_pre_forward_order
        self._pre_forward_order_index: Optional[int] = None
        # Index in `handles_post_forward_order`
        self._post_forward_index: Optional[int] = None
        # Used for guarding against mistargeted forward prefetches
        self._needs_pre_forward_unshard = False
        # Used for guarding against mistargeted backward prefetches
        self._needs_pre_backward_unshard = False
        # Was the handle prefetched? Set on successful _prefetch_handle and unshard
        self._prefetched = False
        self._ran_pre_backward_hook = False
        self._ran_post_backward_hook = False
        #!==================== add support for zero1 param & grad sync state=========================
        self._needs_param_sync = True
        self._param_synced = False
        self._grad_synced = False
        self.enter_backward = False
        #!===================================================================================
        self._offload_grads = offload_grads
        self.prev_iter_synced = True
        # Optimistically assume a valid input `params` and set dtype attributes
        # before `_init_flat_param()`, which performs the actual validation
        self._orig_param_dtype = params[0].dtype
        self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
        self._aligned_numel = (
            _get_aligned_numel(unsharded_dtype=self._fwd_bwd_param_dtype)
            if align_addresses
            else 0
        )
        if self.zero1_world_size % self.zero3_group_size != 0:
            raise ValueError(f"The dp {self.zero1_world_size=} is not multiply of {self.zero3_group_size=}")

    @property
    def full_prec_dtype(self):
        return torch.float32
    
    @property
    def param_dtype(self):
        return self._fwd_bwd_param_dtype
    
    @property
    def grad_dtype(self):
        return self._reduce_dtype
    
    def _init_setattr_fns(self):
        use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, "") == "1"
        self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

    def _init_get_unflat_views_fn(self, align_addresses: bool):
        self._get_unflat_views = (
            self._get_unflat_views_aligned
            if align_addresses
            else self._get_unflat_views_unaligned
        )

    def _init_flat_param_and_metadata(
        self,
        params: List[Union[Tensor, nn.Parameter]],
        module: nn.Module,
        aligned_numel: int,
        div: int
    ) -> None:
        """
        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
        if len(params) == 0:
            raise ValueError("Expects non-empty `params`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        (
            dtype,
            flat_param_requires_grad,
            device,
        ) = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        # For alignment padding, only `numels` gets strictly non-`None`
        # elements, and all other lists get `None` elements for padding.
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        fqns: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[
            Union[Tensor, nn.Parameter], Tuple[nn.Module, str, str]
        ] = {}
        params_to_flatten: List[Union[Tensor, nn.Parameter]] = []
        shared_params: List[Union[Tensor, nn.Parameter]] = []
        is_padding_mask: List[bool] = []
        total_numel = total_numel_without_padding = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param not in params_set:
                    continue
                if param in shared_param_memo:  # shared reference
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
                    shared_params.append(param)
                    shared_param_infos.append(
                        SharedParamInfo(
                            param_name,
                            submodule,
                            submodule_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                else:
                    if aligned_numel > 0:
                        numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                        if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                            padding_tensor = _construct_padding_tensor(
                                numel_to_pad, dtype, False, device
                            )
                            params_to_flatten.append(padding_tensor)
                            is_padding_mask.append(True)
                            numels.append(numel_to_pad)
                            total_numel += numel_to_pad
                    param = cast(nn.Parameter, param)
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    is_padding_mask.append(False)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    fqn = (
                        submodule_name + "." + param_name
                        if submodule_name
                        else param_name
                    )
                    fqns.append(fqn)
                    total_numel += param.numel()
                    total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {params}\nmodule: {module}"
            )
        if (
            self.zero1_group_rank == 0
            and aligned_numel > 0
            and total_numel != total_numel_without_padding
        ):
            logger.info(
                "ZeRo3 FlatParameter address alignment created "
                "%s numel of padding (%s vs. %s)",
                total_numel - total_numel_without_padding,
                total_numel,
                total_numel_without_padding,
            )
        # if aligned_numel > 0:
            # Pad to be divisible by world size to avoid a copy for the
            # post-backward reduce-scatter
        numel_to_pad = div - (total_numel % div)
        if numel_to_pad > 0 and numel_to_pad < div:
            if self.zero1_group_rank == 0:
                logger.info(
                    "ZeRO3 FlatParameter world size divisibility created "
                    "%s numel of padding",
                    numel_to_pad,
                )
            padding_tensor = _construct_padding_tensor(
                numel_to_pad, dtype, False, device
            )
            params_to_flatten.append(padding_tensor)
            is_padding_mask.append(True)
            numels.append(numel_to_pad)
            total_numel += numel_to_pad
        # Pass `aligned_numel=0` since we already included padding tensors
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
            params_to_flatten,
            aligned_numel=0,
            requires_grad=flat_param_requires_grad,
            div=div
        )
        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            fqns,
            shared_param_infos,
            _convert_to_params(params_to_flatten),
            _convert_to_params(shared_params),
            is_padding_mask,
        )

    def _validate_tensors_to_flatten(
        self, tensors: List[Union[Tensor, nn.Parameter]]
    ) -> Tuple:
        """
        Validates the tensors to flatten and returns any necessary metadata.
        """
        dtype: Optional[torch.dtype] = None
        # Return as the logical OR over each tensor's value
        flat_param_requires_grad: Optional[bool] = None
        device: Optional[torch.device] = None
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if dtype is None and not tensor.is_floating_point():
                raise ValueError("Cannot flatten integer dtype tensors")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {dtype} "
                    f"and {tensor.dtype}"
                )
            if device is not None and tensor.device != device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{device} and {tensor.device}"
                )
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        return dtype, flat_param_requires_grad, device

    def flatten_tensors(
        self,
        tensors: List[Tensor],
        aligned_numel: int,
        div: int
    ) -> Tensor:
        """
        Flattens ``tensors`` into a single flat tensor optionally including
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        div: The total tensor numel is a multipy of div to avoid different size among rank
        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        if len(tensors) == 0:
            raise ValueError("Expects non-empty `tensors`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        flat_tensors: List[Tensor] = []
        if aligned_numel > 0:
            total_numel = 0
            for tensor in tensors:
                numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                    padding_tensor = _construct_padding_tensor(
                        numel_to_pad, dtype, False, device
                    )
                    flat_tensors.append(padding_tensor)
                    total_numel += numel_to_pad
                flat_tensors.append(torch.flatten(_detach_if_needed(tensor)))
                total_numel += tensor.numel()
            numel_to_pad = div - (total_numel % div)
            if numel_to_pad > 0 and numel_to_pad < div:
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
        else:
            flat_tensors = [
                torch.flatten(_detach_if_needed(tensor)) for tensor in tensors
            ]
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(
        self,
        tensors: List[Tensor],
        aligned_numel: int,
        requires_grad: bool,
        div: int
    ) -> FlatParameter:
        flat_param_data = self.flatten_tensors(tensors, aligned_numel, div)
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_param_reduce_dtypes(
        self,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
    ) -> None:
        """
        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
        # Save whether these dtypes were specified so that we permit the
        # parameter dtype to change up until the lazy initialization
        self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
        self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        if self._fwd_bwd_param_dtype is None or self._reduce_dtype is None:
            raise ValueError(f"Runtime dtype not set")

    ###################################
    # SHARD INITIALIZATION & METADATA #
    ###################################
    @torch.no_grad()
    def shard(self):
        """
        Shards the handle's ``FlatParameter``. This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        _p_assert(
            flat_param.storage_offset() == 0,
            "The `FlatParameter` is not the sole occupant of its storage",
        )
        orig_storage = flat_param._typed_storage()
        #! _get_shard returns a clone of original parameter
        zero1_flat_param, zero1_padded = FlatParamHandle._get_shard(
            flat_param, self.zero1_group_rank, self.zero1_world_size
        )
        zero1_flat_param = zero1_flat_param.to(self.full_prec_dtype)       
        flat_param._zero1_shard = zero1_flat_param
        flat_param.data = zero1_flat_param  # type: ignore[call-overload]
        
        start_idx = zero1_flat_param.numel() * self.zero1_group_rank
        end_idx = zero1_flat_param.numel() * (self.zero1_group_rank + 1) - 1  # inclusive
        
        self._init_shard_metadata(zero1_padded, start_idx, end_idx)
        if orig_storage._size() > 0:
            orig_storage._resize_(0)
        self._use_sharded_views()   

    def _init_shard_metadata(
        self,
        numel_padded: int,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> None:
        """
        Initializes shard-related metadata for this rank's shard of the flat
        parameter: ``_sharded_size``, ``_shard_param_infos``, and
        ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()  # type: ignore[attr-defined]
        sharded_flat_param_numel = flat_param.numel()  # includes `numel_padded`
        _p_assert(
            unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx,
            f"unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}",
        )
        _p_assert(
            numel_padded <= sharded_flat_param_numel,
            f"numel_padded: {numel_padded} "
            f"sharded_flat_param_numel: {sharded_flat_param_numel}",
        )
        shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )
        _p_assert(
            len(shard_param_infos) == flat_param._num_params,
            f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
        )
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]
        flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> Tuple[_ShardParamInfo, ...]:
        """
        Computes the shard metadata based on ``unsharded_start_idx`` and
        ``unsharded_end_idx`` (inclusive), which give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        _p_assert(len(flat_param_offsets) == len(
            self.flat_param._numels_with_padding
        ), f"Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}"
        )
        shard_param_infos: List[_ShardParamInfo] = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        # `unsharded_param_start_idx` and `unsharded_param_end_idx` are indices
        # into the unsharded flat parameter (inclusive) of the given parameter
        for i, (
            (unsharded_param_start_idx, unsharded_param_end_idx),
            is_padding,
        ) in enumerate(zip(flat_param_offsets, self.flat_param._is_padding_mask)):
            if is_padding:
                continue
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None, unsharded_param_start_idx, unsharded_param_end_idx)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # This branch can only happen once since the rank's
                    # unsharded start index can only intersect one parameter
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                if not (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ):
                    raise ValueError(
                        f"Invalid `offset_in_shard` of {offset_in_shard} for "
                        f"sharded flat parameter with {sharded_flat_param_numel} numel"
                    )
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                    unsharded_param_start_idx,
                    unsharded_param_end_idx,
                )
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    @staticmethod
    def _get_unpadded_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` without any padding for the given
        ``rank`` and ``world_size`` and the numel to pad for that shard.

        If ``tensor`` is already flattened or may be viewed in the flattened
        shape (which is true in the expected usage), then this method does not
        allocate any new tensor memory.
        """
        if rank >= world_size:
            raise ValueError(f"Shard rank should be small than shard world size, got {rank} and {world_size}")
        chunks = torch.flatten(tensor).chunk(world_size)
        if len(chunks) < (rank + 1):
            # This rank gets an empty chunk fully padded with zeros since there
            # are not enough chunks across ranks
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[rank]
        numel_to_pad = chunks[0].numel() - chunk.numel()
        return chunk, numel_to_pad

    @staticmethod
    def _get_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` with padding for the given ``rank`` and
        ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
        chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        shard = chunk.clone()
        if numel_to_pad > 0:
            shard = F.pad(shard, [0, numel_to_pad])
        return shard, numel_to_pad

    @staticmethod
    def _get_shard_from_padded_unshard_tensor(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` with padding for the given ``rank`` and
        ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
        chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        shard = chunk.clone()
        _p_assert(numel_to_pad == 0, f"The padded unshard flat param should be dividable with {world_size=}")
        return shard
    
    def _get_flat_param_offsets(self) -> List[Tuple[int, int]]:
        """
        Returns [start, end] offsets of each original parameter's flattened
        data in the unsharded flat parameter (without padding).
        NOTE: The returned list includes elements for alignment padding.
        """
        cumulative_sum = list(accumulate(self.flat_param._numels_with_padding))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        param_offsets = list(zip(starts, ends))
        return param_offsets

    @no_type_check
    @torch.no_grad()
    def init_flat_param_attributes(self) -> None:
        """
        This initializes some attributes on the handle's ``FlatParameter``.
        This should be called during lazy initialization since it requires the
        parameter to be on the compute device if not offloading to CPU and we
        want to give users the chance to move the parameter appropriately after
        the FSDP constructor.

        For each tensor attribute on the ``FlatParameter``, see the unshard and
        reshard methods in this class for the allocation and free pattern.
        """
        flat_param = self.flat_param
        self._check_on_compute_device(self.flat_param)
        # We maintain a padded unsharded tensor that serves as the
        # all-gather destination and owns the original parameter storages.
        padded_unsharded_numel = flat_param.numel() * self.zero1_world_size
        flat_param._full_param_padded = torch.empty(
            padded_unsharded_numel,
            device=self.device,
            dtype=self._fwd_bwd_param_dtype,
        )
        flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
        _free_storage(flat_param._full_param_padded)
        #! add support for grad saving
        flat_param._full_grad_padded = torch.empty(
            padded_unsharded_numel,
            device=self.device,
            dtype=self._fwd_bwd_param_dtype,
        )
        _free_storage(flat_param._full_grad_padded)
        #! grad accumulation support
        flat_param._full_prec_grad_padded = torch.empty(
            padded_unsharded_numel,
            device=self.device,
            dtype=self.full_prec_dtype,
        )
        _free_storage(flat_param._full_prec_grad_padded)
        if self._offload_grads:
            cpu_device = torch.device("cpu")
            flat_param._cpu_grad = torch.zeros(
                    padded_unsharded_numel,
                    device=cpu_device,
                    dtype=self.full_prec_dtype,
                ).pin_memory(device=self.device)
    ###################
    # UNSHARD/RESHARD #
    ###################
    
    def pre_unshard(self) -> bool:
        """
        Returns: ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. 
        """
        if (
            self._training_state in [HandleTrainingState.SUMMON_FULL_PARAMS, HandleTrainingState.SYNC_PARAMS]
            and self._skipped_use_sharded_views
        ):
            self._use_sharded_views()
        self._check_on_compute_device(self.flat_param)
        if self.needs_unshard():
            self._alloc_padded_unsharded_flat_tensor()

    def unshard(self):
        padded_unsharded_flat_param = self._get_padded_unsharded_flat_tensor(param=True, free=False)
        padded_unsharded_flat_param = self._all_gather_flat_param(padded_unsharded_flat_param)
        self._use_unpadded_unsharded_flat_param(padded_unsharded_flat_param)
            
    def needs_unshard(self) -> bool:
        """Returns if the handle's flat parameter needs to be unsharded."""
        padded_unsharded_flat_param = self._get_padded_unsharded_flat_tensor(free=False)
        already_unsharded = (
            padded_unsharded_flat_param._typed_storage()._size()
            == padded_unsharded_flat_param.numel()
        )
        return not already_unsharded

    def _alloc_padded_unsharded_flat_tensor(self, param: bool = True):
        flat_param = self.flat_param
        unsharded_flat_tensor = self._get_padded_unsharded_flat_tensor(param)
        self._check_storage_freed(unsharded_flat_tensor)
        _alloc_storage(unsharded_flat_tensor,
                       flat_param._padded_unsharded_size)
        return unsharded_flat_tensor

    def _get_padded_unsharded_flat_tensor(self, param: bool = True, free: bool = True) -> torch.Tensor:
        """
        Returns a reference to the padded unsharded flat parameter depending on
        the calling context. This should only be called if using a sharded
        strategy.
        """
        flat_param = self.flat_param
        if param:
            padded_unsharded_flat_tensor = flat_param._full_param_padded
            dtype = self._fwd_bwd_param_dtype
        else:
            padded_unsharded_flat_tensor = flat_param._full_grad_padded
            dtype = self._fwd_bwd_param_dtype
        _p_assert(
            padded_unsharded_flat_tensor.dtype == dtype,
            f"Expects same precision but got {padded_unsharded_flat_tensor.dtype}  vs {dtype}",
        )

        if free and padded_unsharded_flat_tensor.untyped_storage().size() > 0:
            _free_storage(padded_unsharded_flat_tensor)
        return padded_unsharded_flat_tensor

    def _all_gather_flat_param(
        self,
        padded_unsharded_flat_param: Tensor,
    ) -> Tensor:
        """
        All-gathers the handle's flat parameter to the destination
        ``padded_unsharded_flat_param``, and switches to using the all-gathered
        tensor.
        """
        _p_assert(
            hasattr(self, "zero3_process_group") and hasattr(self, "zero3_group_size"),
            "Expects a process group and world size to have been set via `shard()`",
        )
        #! cast zero1 param to zero3 param
        #! be careful of recompute
        if self._needs_param_sync and not self._param_synced:
            sharded_flat_param = self.flat_param._zero1_shard.to(self._fwd_bwd_param_dtype)
            expected_numel = sharded_flat_param.numel() * self.zero1_world_size
            process_group = self.zero1_process_group
            source = "zero1 shard"
        else:
            sharded_flat_param = self.flat_param._zero3_shard.to(self._fwd_bwd_param_dtype)
            expected_numel = sharded_flat_param.numel() * self.zero3_group_size
            process_group = self.zero3_process_group
            source = "zero3 shard"
            
        _p_assert(
            padded_unsharded_flat_param.numel() == expected_numel,
            f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}")
        log0(f"All gather into full parameter from {source} with {process_group.size()=}")
        dist.all_gather_into_tensor(
            padded_unsharded_flat_param,
            sharded_flat_param,
            process_group,
        )
        return padded_unsharded_flat_param

    def _use_unpadded_unsharded_flat_param(
        self,
        padded_unsharded_flat_param: torch.Tensor,
    ) -> None:
        """
        Switches to using the *unpadded* unsharded flat parameter, which is a
        view into the *padded* unsharded flat parameter.
        """
        unsharded_size = self.flat_param._unpadded_unsharded_size
        self.flat_param.data = padded_unsharded_flat_param[:unsharded_size.numel()].view(unsharded_size)
        # this `.view()` is not autograd visible
        in_forward = self._training_state == HandleTrainingState.FORWARD
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        if in_forward or in_pre_backward:
            self._use_unsharded_views(as_params=False)
        else:
            self._use_unsharded_views(as_params=True)
            
    def _use_unpadded_unsharded_flat_grad(
        self,
        padded_unsharded_flat_grad: torch.Tensor,
    ) -> None:
        """
        Switches to using the *unpadded* unsharded flat parameter, which is a
        view into the *padded* unsharded flat parameter.
        """
        unsharded_size = self.flat_param._unpadded_unsharded_size
        self.flat_param.grad.data = padded_unsharded_flat_grad[:unsharded_size.numel()].view(unsharded_size)
        self._use_unsharded_grad_views()
        
    def post_unshard(self):
        """
        Runs the post-unshard logic. This includes freeing the low precision
        shard if needed.
        """
        self._check_on_compute_device(self.flat_param)

    @torch.no_grad()
    def unshard_grad(self):
        """
        Unshard the handle's ``FlatParameter``'s gradient.

        If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
        flat_param = self.flat_param
        self._check_unsharded(flat_param)

        # Check if all ranks have a `None` gradient
        num_grad_none = torch.zeros(1, dtype=torch.int32, device=self.device)
        num_grad_none[0] = flat_param.grad is None
        dist.all_reduce(num_grad_none, group=self.zero1_process_group)
        if num_grad_none[0] == self.zero1_world_size:
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            self._use_unsharded_grad_views()
            return
        if flat_param.grad is None:
            # In the case that only some ranks have `None` gradient, we use
            # zeros to approximate as a best effort attempt
            if self._debug_level == dist.DebugLevel.INFO:
                warnings.warn(
                    f"[Rank {self.rank}] Only some but not all ranks have a "
                    "`None` `FlatParameter` gradient, so FSDP is using zeros to "
                    "approximate those ranks' sharded gradients being `None`"
                )
            flat_param._saved_grad = None  # type: ignore[assignment]
            sharded_grad = torch.zeros(flat_param._sharded_size, device=self.device, dtype=self._fwd_bwd_param_dtype)  # type: ignore[attr-defined]
        #如果该rank上有梯度,保存在flat_param._saved_grad中
        else:
            self._check_sharded(flat_param.grad)
            # flat_param._saved_grad = flat_param.grad  # type: ignore[attr-defined]
            sharded_grad = flat_param.grad.to(self._fwd_bwd_param_dtype)  # type: ignore[attr-defined]
        # 分配内存,全聚合
        padded_unsharded_grad = torch.zeros(
            flat_param._padded_unsharded_size,  # type: ignore[attr-defined]
            device=self.device,
            dtype=self._fwd_bwd_param_dtype,
        )
        dist.all_gather_into_tensor(
            padded_unsharded_grad, sharded_grad, self.zero1_process_group
        )
        # 使用非分片的梯度视图
        unsharded_size = self.flat_param._unpadded_unsharded_size
        flat_param.grad = padded_unsharded_grad[: unsharded_size.numel()].view(
            unsharded_size
        )
        self._use_unsharded_grad_views()

    def reshard_grad(self):
        self.flat_param.grad = self.flat_param._saved_grad  # type: ignore[attr-defined]
        self._use_sharded_grad_views()
        delattr(self.flat_param, "_saved_grad")
        
    def offload_grad(self):
        if not self._offload_grads:
            warnings.warn(f"Call offload grad when offload grads is False")
            return
        cpu_tensor = self.flat_param._cpu_grad
        gpu_tensor = self.flat_param._full_prec_grad_padded
        self._check_on_cpu(cpu_tensor)
        self._check_on_compute_device(gpu_tensor)
        self._check_padded_unsharded(gpu_tensor)
        cpu_tensor.untyped_storage().copy_(gpu_tensor.untyped_storage(), non_blocking=True)
        
    def alloc_full_prec_grad(self):
        if not self.already_load_full_prec_grad():
            flat_param = self.flat_param
            full_prec_grad = flat_param._full_prec_grad_padded
            self._check_storage_freed(full_prec_grad)
            _alloc_storage(full_prec_grad, flat_param._padded_unsharded_size) 
            full_prec_grad.zero_()
            return 
                
    def reload_full_prec_grad(self):
        if not self._offload_grads:
            return
        with torch.no_grad():
            gpu_tensor = self.flat_param._full_prec_grad_padded
            self._check_padded_unsharded(gpu_tensor)
            self._check_on_compute_device(gpu_tensor)
            cpu_tensor = self.flat_param._cpu_grad
            self._check_on_cpu(cpu_tensor)
            gpu_tensor.untyped_storage().copy_(cpu_tensor.untyped_storage(), non_blocking=True)
    
    def already_load_full_prec_grad(self):
        gpu_tensor = self.flat_param._full_prec_grad_padded
        return gpu_tensor.device == self.device and gpu_tensor.untyped_storage().size() > 0
    
    def free_full_prec_grad(self):
        full_prec_grad = self.flat_param._full_prec_grad_padded
        self._check_on_compute_device(full_prec_grad)
        _free_storage(full_prec_grad)
        
    def accumulate_grad(self):
        '''
        Precondition:
        runtime_grad: _full_grad_padded finished grad compute
        
        Postcondition:
        grad is accumulated to full_prec_grad
        '''
        full_prec_grad = self.flat_param._full_prec_grad_padded
        runtime_grad = self.flat_param._full_grad_padded
        self._check_padded_unsharded(full_prec_grad)
        self._check_padded_unsharded(runtime_grad)
        self._check_on_compute_device(full_prec_grad)
        self._check_on_compute_device(runtime_grad)
        full_prec_grad.add_(runtime_grad)
        return 
        
    def prepare_gradient_for_backward(self):
        """
        Prepares the gradient for the backward computation by saving and
        clearing any existing sharded gradient in ``.grad`` to enable computing
        a new unsharded gradient.
        
        #! optimize this logic:
        1. if grad is not freed, Then last iter must not synced grad, then we use use_unshard_grad_view to accumulate grad

        2. if grad is freed, Then last iter must synced grad. alloc memeory for grad.
            2.1 alloc memory for grad computation
            2.2 set grad views
            
        PostCondition:
            flat_param.grad is the padded_unshard_grad
            return the views of grad in correct position
        """

        _p_assert(
            self._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.IDLE),
            "Expects to be in `BACKWARD_PRE` or `IDLE` (if prefetching)",
        )
      
        flat_param = self.flat_param
        if not flat_param.requires_grad:
            return
        _p_assert(flat_param._full_grad_padded is not None, f"{self} got a None _full_grad_padded tensor for unshard flat parameters...")
        self._check_on_compute_device(flat_param)
        self._check_unsharded(flat_param.data)
        #! 1. alloc memory if needed
        padded_unsharded_flat_grad = flat_param._full_grad_padded
        if self._is_storage_freed(padded_unsharded_flat_grad):
            #! alloc memory
            self._alloc_padded_unsharded_flat_tensor(param=False)
            padded_unsharded_flat_grad.zero_()
        else:
            self._check_padded_unsharded(padded_unsharded_flat_grad)
        #! 2.  point grad to the reference tensor set proper view and grad view
        flat_param.grad = flat_param._full_grad_padded  
        self._use_unpadded_unsharded_flat_grad(padded_unsharded_flat_grad)

    def set_shard_grad(self, shard_grad):
        flat_param = self.flat_param
        _p_assert(not self._grad_synced, "A parameter should only sync its grad only once during one grad sync cycle")
        flat_param._saved_grad = shard_grad 
        self._grad_synced = True
    
    def free_runtime_unshard_grad(self):
        self._free_unsharded_flat_tensor(param=False)
        
    def prepare_gradient_for_zero1(self):
        """
        Prepares the gradient for optimizer computation by moving the sharded
        gradient to the ``.grad`` attribute for the convienience of later reduce op
        Precondition : saved_grad is the sharded grad 
        
        Postcondition: storage of saved_grad is freed
        
        Post Condition:
        ``.grad`` contains only the ``shard grad`` : Note : unshard grad storage free is done after zero1 grad sync
        the full unsharded grad storage is freed
        """
        self._use_sharded_views()
        self._use_sharded_grad_views() 
        del self.flat_param._saved_grad

    def _get_reduce_scatter_tensors(self):
        tensor = self.flat_param._full_prec_grad_padded
        _p_assert(tensor.dtype == self.full_prec_dtype, "full_prec grad is not full prec.")
        self._check_padded_unsharded(tensor)
        self._check_on_compute_device(tensor)
        chunks = tensor.chunk(self.zero1_world_size)
        new_tensor = torch.empty_like(chunks[0])
        return tensor, new_tensor
    
    def _get_reduce_scatter_group(self):
        return self.zero1_process_group
    
    def reshard(self, free_unsharded_flat_param: bool):
        """
        Runs the reshard logic. This includes freeing the unsharded flat
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flat parameter.
        """
        if self._needs_param_sync and not self._param_synced:
            zero3_shard = FlatParamHandle._get_shard_from_padded_unshard_tensor(self.flat_param.data, self.zero3_group_rank, self.zero3_group_size)
            self.flat_param._zero3_shard = zero3_shard
            self._param_synced = True
            
        if free_unsharded_flat_param:
            self._use_sharded_flat_param()
            self._free_unsharded_flat_tensor()


    def post_reshard(self):
        """
        Runs the post-reshard logic. 
        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flat parameter.
        """
        pass

    def _free_unsharded_flat_tensor(self, param: bool = True):
        """
        Frees the padded unsharded flat parameter. The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        msg = "Parameter" if param else "Gradient"
        log0(f"Freeing {msg} memory on handle {self}, {self._pre_forward_order_index=} {self._post_forward_index=}")
        
        unsharded_flat_tensor = self._get_padded_unsharded_flat_tensor(param)
        self._check_on_compute_device(unsharded_flat_tensor)
        # Do not free the memory until all ops in the current stream finish
        _no_dispatch_record_stream(
            unsharded_flat_tensor, self._device_handle.current_stream()
        )
        _free_storage(unsharded_flat_tensor)

    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flat parameter."""
        flat_param = self.flat_param
        flat_param.data = flat_param._zero1_shard  # type: ignore[attr-defined]
        self._use_sharded_views()
    #########
    # VIEWS #
    #########
    
    @no_type_check
    def _get_unflat_views_unaligned(
        self,
        tensor: Optional[torch.Tensor] = None,
    ) -> Iterator[Tensor]:
        """
        Returns unflattened ``Tensor`` views into ``tensor`` if it is not
        ``None`` or ``flat_param`` otherwise, where the unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param

        views = (
            subtensor.view(shape)
            for (subtensor, shape) in zip(
                torch.split(tensor, flat_param._numels, dim=0),
                flat_param._shapes,
            )
        )
        return views

    @no_type_check
    def _get_unflat_views_aligned(
        self,
        tensor: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        This has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        splits: List[Tensor] = torch.split(
            tensor, flat_param._numels_with_padding, dim=0
        )
        idx = 0
        views: List[Tensor] = []
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            if is_padding:
                continue
            views.append(
                split.view(flat_param._shapes[idx])
            )
            idx += 1
        return views

    @no_type_check
    @torch.enable_grad()
    def _use_unsharded_views(self, as_params: bool) -> None:
        """
        Unflattens the unsharded flat parameter by setting the original
        parameter variables to be views into it.
        
        unsharded unpadded and restore original parameter views

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.
        """
        log0(f"Change to unsharded Parameter View on {self._pre_forward_order_index=} {self._post_forward_index=}")
        
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        views = self._get_unflat_views()

        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, flat_param._param_infos)
        ):
            if as_params:
                param = self.flat_param._params[i]
                self._setattr_param(module, param_name, param)
                param.data = view
            else:  # `as_params=False`
                param_var: Tensor = view
                if self.flat_param._tensors[i] is None:
                    # Save the `Tensor` for the pre-backward
                    self.flat_param._tensors[i] = view  # save for pre-backward
                else:
                    # Use the saved `Tensor` variable from the forward to
                    # preserve the autograd graph so that the post-backward
                    # hook fires (e.g. for reentrant AC)
                    tensor = self.flat_param._tensors[i]
                    tensor.data = view
                    param_var = tensor
                self._setattr_tensor(module, param_name, param_var)
                if self._training_state == HandleTrainingState.FORWARD:
                    module._parameters[param_name] = param_var
        for i, (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(self.flat_param._shared_param_infos):
            prim_param: Union[Tensor, nn.Parameter] = getattr(
                prim_module, prim_param_name
            )
            _p_assert(
                not as_params or isinstance(prim_param, nn.Parameter),
                f"as_params={as_params} type(prim_param)={type(prim_param)}",
            )
            if as_params:
                shared_param = self.flat_param._shared_params[i]
                self._setattr_param(module, param_name, shared_param)
                shared_param.data = prim_param
            else:
                self._setattr_tensor(module, param_name, prim_param)
                if self._training_state == HandleTrainingState.FORWARD:
                    module._parameters[param_name] = prim_param

    @no_type_check
    def _use_unsharded_grad_views(self) -> None:
        """
        Unflattens the unsharded flat parameter's gradient by setting the
        original parameter variables' gradients to be views into it.
        
        From the unpadded unshard grad to set parameter grad views at corresponing position relative to param
        SO basically this is a similiar function to use_unsharded_param_views
        """
        log0(f"Change to unsharded Gradient View on {self._pre_forward_order_index=} {self._post_forward_index=}")
        
        if self.flat_param.grad is None:
            for param in chain(self.flat_param._params, self.flat_param._shared_params):
                param.grad = None
            return
        # Expects the gradient to be in `flat_param.grad`
        self._check_unsharded(self.flat_param.grad)

        views = self._get_unflat_views(self.flat_param.grad)
        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, self.flat_param._param_infos)
        ):
            _p_assert(
                hasattr(module, param_name),
                f"{self.flat_param._fqns[i]} is missing",
            )
            param = getattr(module, param_name)
            if (
                param.shape != view.shape
                or param.dtype != view.dtype
                or param.device != view.device
            ):
                # NOTE: This is a hack using `.data` to side step the check
                # that parameter/gradient sizes/dtypes/devices match. From
                # calling `reshard()`, `param` has the sharded size, has the
                # full precision dtype, and if CPU offloading is enabled, is on
                # CPU. Thus, one or more of the following cases can hold when
                # in `no_sync()`, where `view` is the original parameter's
                # gradient:
                # 1. `view` can have the unsharded size.
                # 2. `view` can have the parameter low precision dtype.
                # 3. `view` can be on GPU.
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = view
            else:
                param.grad = view
        for i, (
            param_name,
            module,
            module_name,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(self.flat_param._shared_param_infos):
            _p_assert(
                hasattr(module, param_name),
                f"{module_name + '.' + param_name if module_name else param_name} is missing",
            )  # did not save FQN info in `_shared_param_infos`
            param = getattr(module, param_name)
            prim_param = getattr(prim_module, prim_param_name)
            if (
                param.shape != prim_param.grad.shape
                or param.dtype != prim_param.grad.dtype
                or param.device != prim_param.grad.device
            ):
                # NOTE: This is the same hack to use `.data` to side step the
                # size check.
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = prim_param.grad
            else:
                param.grad = prim_param.grad

    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Assumes the flat parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flat parameter, and after the context, restores the original parameters
        as ``Tensor`` views into the flat parameter.
        """
        self._use_unsharded_views(as_params=True)
        try:
            yield
        finally:
            self._use_unsharded_views(as_params=False)

    @no_type_check
    @torch.no_grad()
    def _use_sharded_views(self) -> None:
        """
        Sets the original parameter variables' data to be flattened views into
        the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
        log0(f"Change to sharded Parameter View on {self._pre_forward_order_index=} {self._post_forward_index=}")
        self._unsharded_flat_param_for_skipped_views = None
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        # Construct once and reuse for all parameters not in the local shard
        size_0_empty_tensor = torch.empty(
            0,
            dtype=self.flat_param.dtype,  # in case `flat_param` changed dtype
            device=self.flat_param.device,
            requires_grad=False,
        )
        for param, shard_param_info, (param_name, module, _) in zip(
            flat_param._params, 
            flat_param._shard_param_infos, 
            flat_param._param_infos
        ):
            self._setattr_param(module, param_name, param)
            if not shard_param_info.in_shard:
                # Allow the original data to be freed via garbage collection
                param.data = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                param.data = flat_param[offset : offset + numel_in_shard]
        for i, (
            param,
            (param_name, module, _, prim_param_name, prim_module, _),
        ) in enumerate(
            zip(self.flat_param._shared_params, self.flat_param._shared_param_infos)
        ):
            self._setattr_param(module, param_name, param)
            prim_param = getattr(prim_module, prim_param_name)
            param.data = prim_param  # could be both empty and non-empty
        if self._training_state == HandleTrainingState.BACKWARD_POST:
            # Clear the saved `Tensor`s since they are unneeded now
            for i in range(len(self.flat_param._tensors)):
                self.flat_param._tensors[i] = None
                
    @no_type_check
    @torch.no_grad()
    def _use_sharded_grad_views(self) -> None:
        """
        Set the original parameter variables' gradients to be flattened views into the sharded flat parameter's gradient.

        This is a no-op if there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
        log0(f"Change to sharded Gradient View on {self._pre_forward_order_index=} {self._post_forward_index=}")
        
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        grad = self.sharded_grad
        if grad is None:
            for param in chain(flat_param._params, flat_param._shared_params):
                param.grad = None
            return
        self._check_sharded(grad)
        for param, shard_param_info, is_grad_none in zip(
            flat_param._params,
            flat_param._shard_param_infos,
            flat_param._is_grad_none_mask,
        ):
            if not shard_param_info.in_shard:
                param.grad = None
            else:
                numel_in_shard = shard_param_info.numel_in_shard
                
                if param.requires_grad and not is_grad_none:
                    offset = shard_param_info.offset_in_shard
                    if param.dtype != grad.dtype:
                        if param.grad is None:
                            # `.grad` must have the same shape as `param`
                            param.grad = torch.empty_like(param)
                        param.grad.data = grad[
                            offset : offset + numel_in_shard
                        ]
                    else:
                        param.grad = grad[
                            offset : offset + numel_in_shard
                        ]
                    
                else:
                    param.grad = None

        for i, (param, (_, _, _, prim_param_name, prim_module, _)) in enumerate(
            zip(flat_param._shared_params, flat_param._shared_param_infos)
        ):
            in_sharded_flat_param = hasattr(prim_module, prim_param_name)
            if in_sharded_flat_param and param.requires_grad:
                prim_param = getattr(prim_module, prim_param_name)
                param.grad = prim_param.grad 
            else:
                param.grad = None

    def _reset_flat_param_grad_info_if_needed(self):
        """
    
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
        flat_param = self.flat_param
        all_grad_none = True
        requires_grad = False
        for param in flat_param._params:
            all_grad_none &= param.grad is None
            requires_grad |= param.requires_grad
        if all_grad_none:
            flat_param.grad = None
        # As long as one parameter requires gradient, then the flat parameter
        # must require gradient
        flat_param.requires_grad = requires_grad

    def _deregister_orig_params(self):
        for param_info in self.flat_param._param_infos:
            param_name, module, _ = param_info
            if hasattr(module, param_name):
                delattr(module, param_name)
        for param_name, module, _, _, _, _ in self.flat_param._shared_param_infos:
            if hasattr(module, param_name):
                delattr(module, param_name)

    ###########
    # HELPERS #
    ###########
    def _get_modules(self) -> Set[nn.Module]:
        """
        Returns a :class:`set` of the modules whose parameters are included
        in this handle's flat parameter.
        """
        return {pi.module for pi in self.flat_param._param_infos}.union(
            {spi.module for spi in self.flat_param._shared_param_infos}
        )

    def is_sharded(self, tensor: Tensor) -> bool:
        """
        Returns if ``tensor`` is *currently* sharded. For ``NO_SHARD``, we
        choose to have this always return ``False`` for clarity.
        """
        if (
            not hasattr(self.flat_param, "_sharded_size")
        ):
            # `_sharded_size` is defined iff `handle.shard()` has been called
            return False
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        return tensor.size() == sharded_size

    def param_module_names(self) -> Iterator[Tuple[str, str]]:
        shared_param_infos = [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,
                _,
            ) in self.flat_param._shared_param_infos
        ]
        for param_info in chain(self.flat_param._param_infos, shared_param_infos):
            param_name, _, module_name = param_info
            yield (param_name, module_name)

    def shared_param_module_names(self) -> Iterator[Tuple[str, str]]:
        for param_name, _, module_name in [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,
                _,
            ) in self.flat_param._shared_param_infos
        ]:
            yield (param_name, module_name)

    @property
    def _fqns_in_shard(self) -> List[str]:
        """Returns the FQNs of the parameters present in this rank's shard."""
        fqns_in_shard: List[str] = []
        for fqn, shard_param_info in zip(
            self.flat_param._fqns, self.flat_param._shard_param_infos
        ):
            if shard_param_info.in_shard:
                fqns_in_shard.append(fqn)
        return fqns_in_shard

    @property
    def sharded_grad(self) -> Optional[Tensor]:
        """Returns the handle's sharded gradient."""
        flat_param = self.flat_param
        grad: Optional[Tensor]
        
        if hasattr(flat_param, "_saved_grad"):
            # In the post-backward hook, the sharded gradient is still in
            # `_saved_grad_shard`.
            grad = flat_param._saved_grad.to(self.full_prec_dtype)
        else:
            # If in IDLE or in FORWARD states, then there may be an
            # (accumulated) gradient. If accessed in IDLE, then this should
            # be due to re-registering the original parameters (e.g. in state
            # dict load).
            _p_assert(
                flat_param.grad is None
                or self._training_state
                in (HandleTrainingState.FORWARD, HandleTrainingState.IDLE),
                "Sharded strategies should use `_cpu_grad` or `_saved_grad_shard` "
                "unless in IDLE or FORWARD",
            )
            grad = None
        return grad

    #######################
    # CHECKS & INVARIANTS #
    #######################
    def _check_on_compute_device(self, tensor: Tensor):
        _p_assert(
            tensor.device == self.device,
            f"Expects tensor to be on the compute device {self.device}",
        )

    def _check_on_cpu(self, tensor: Tensor):
        _p_assert(
            tensor.device == torch.device("cpu"),
            f"Expects tensor to be on CPU but got {tensor.device}",
        )

    @staticmethod
    def _check_storage_freed(tensor: Tensor):
        storage_size: int = tensor._typed_storage()._size()
        _p_assert(
            storage_size == 0,
            f"Expects storage to be freed but got storage with size {storage_size}",
        )
        
    @staticmethod
    def _is_storage_freed(tensor: Tensor) -> bool:
        return tensor is not None and tensor._typed_storage()._size() == 0

    @staticmethod
    def _check_storage_allocated(tensor: Tensor):
        storage_size: int = tensor._typed_storage()._size()
        _p_assert(storage_size > 0, "Expects storage to be allocated")

    def _check_unsharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be unsharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        unsharded_size = self.flat_param._unpadded_unsharded_size
        _p_assert(
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()} with storage {tensor.untyped_storage().size()}",
        )
        
    def _check_padded_unsharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be unsharded and padded"
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        unsharded_size = self.flat_param._padded_unsharded_size
        _p_assert(
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()} with storage {tensor.untyped_storage().size()}",
        )

    def _check_sharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be sharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        _p_assert(
            tensor.size() == sharded_size,
            msg_prefix + f"with size {sharded_size} but got {tensor.size()} with storage {tensor.untyped_storage().size()}",
        )

    ##############
    # PROPERTIES #
    ##############
    
    @property
    def _skipped_use_sharded_views(self) -> bool:
        return self._unsharded_flat_param_for_skipped_views is not None
    #================== debug =========================
    
    def _named_module_parameters(self):
        #! 获取模型的parameter, 动态重建的参数
        for i, (param_name, module, module_name) in enumerate(
            self.flat_param._param_infos
        ):
            _p_assert(
                hasattr(module, param_name),
                f"{self.flat_param._fqns[i]} is missing",
            )
            param = getattr(module, param_name)
            yield f"{module_name}.{param_name}", param
            
    def _get_orig_param_by_name(self, total_name):
        flat_param = self.flat_param
        for param, (param_name, _, module_name) in zip(
            flat_param._params, flat_param._param_infos
        ):            
            if total_name == f"{module_name}.{param_name}":
                return param
        return None
            
    def _get_module_param_by_name(self, total_name):
        flat_param = self.flat_param
        for param_name, module, module_name in flat_param._param_infos:
            if total_name == f"{module_name}{param_name}":
                return getattr(module, param_name)
        return None

    def __param_list(self):
        self._use_unsharded_grad_views()
        for param in self.flat_param._params:
            yield param
            yield param
            
            yield param  
            
    
    def _shard_grad_list(self):
        for param in self.flat_param._params:
            yield param.grad


def _unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    module._parameters[param_name] = param
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, param)


def _unsafe_setattr_tensor(module: nn.Module, param_name: str, tensor: Tensor) -> None:
    module._parameters.pop(param_name, None)
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, tensor)


def _safe_setattr_tensor_or_param(
    module: nn.Module, param_name: str, tensor_or_param: Union[Tensor, nn.Parameter]
):
    # Call `delattr()` and `setattr()` to go through `nn.Module` checks
    if hasattr(module, param_name):
        delattr(module, param_name)
    setattr(module, param_name, tensor_or_param)


def _convert_to_params(
    tensors: List[Union[torch.Tensor, nn.Parameter]]
) -> List[nn.Parameter]:
    return [t if isinstance(t, nn.Parameter) else nn.Parameter(t, requires_grad=t.requires_grad) for t in tensors]


def _detach_if_needed(param_or_tensor: Union[nn.Parameter, Tensor]) -> Tensor:
    return (
        param_or_tensor.detach()
        if isinstance(param_or_tensor, nn.Parameter)
        else param_or_tensor
    )


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # NOTE: This alignment constraint comes from TorchInductor.
    ALIGNMENT = 16  # bytes
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@functools.lru_cache(8)
def _get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    return (
        # torch.ones(
        torch.zeros(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
    )


def log0(msg):
    if dist.get_rank() == 0:
        logger.info(msg)
        