"""Define common classes for multi-parameter pipeline parallelism.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch


@dataclass
class InterleavingSchedulerArgs:
    """Define arguments for interleaving scheduler."""

    get_model_config: Callable
    """Get the model config."""

    get_pipeline_model_parallel_world_size: Callable
    """"Get the pipeline model parallel world size."""

    get_pipeline_model_parallel_rank: Callable
    """Get the pipeline model parallel rank."""

    set_virtual_pipeline_model_parallel_rank: Callable
    """Set the virtual pipeline model parallel rank."""

    get_args: Callable
    """Get the args."""

    is_encoder_and_decoder: bool
    """If the model is encoder and decoder."""

    is_pipeline_first_stage: Callable
    """If the model is pipeline first stage."""

    is_pipeline_last_stage: Callable
    """If the model is pipeline last stage."""

    deallocate_output_tensor: Callable
    """Deallocate the output tensor."""

    send_forward_backward_recv_forward_backward: Callable
    """Send forward and backward and receive forward and backward."""

    forward_step: Callable
    """"Forward step."""

    check_first_val_step: Callable
    """Check first validation step."""

    backward_step: Callable
    """Backward step."""

    recv_forward: Callable
    """Receive forward."""

    recv_backward: Callable
    """Receive backward."""

    send_forward_recv_forward: Callable

    """Send forward and receive forward."""

    send_backward_recv_backward: Callable
    """Send backward and receive backward."""


class Config(Protocol):
    """Define a protocol for configuration."""

    timers: Callable
    """Timers object to call for various timing functions. 
    See megatron.core.timers.Timers
    """

    grad_scale_func: Callable
    """If using loss scaling, this function should take the loss 
    and return the scaled loss. If None, no function is called on the loss.
    """

    pipeline_dtype: torch.dtype
    """dtype used in p2p communication, usually params_dtype"""

    overlap_p2p_comm: bool
    """When True some of the peer to peer communication
    for pipeline parallelism will overlap with computation.
    Must be False if batch_p2p_comm is true.
    """

    batch_p2p_comm: bool
    """Use batch_isend_irecv instead of individual isend/irecv calls. 
    Must be False if overlap_p2p_comm is True.
    """

    barrier_with_L1_time: bool
    """If true, use barrier with level 1 time measurements.
        It is up to the user to make sure
       calling barrier with their timers will not result in hangs.
       This can happen if for example
       the user adds a level 1 timer that is not called by all ranks.
    """

    no_sync_func: Callable
    """Function that creates a context that suppresses asynchronous
       data-parallel communication. If
       the model is an instance of core.distributed.DistributedDataParallel,
       the default is to use core.distributed.DistributedDataParallel.no_sync.
    """

    grad_sync_func: Callable
    """Function that launches asynchronous gradient reductions
        (e.g. distributed optimizer gradient reduce-scatters).
        The function should take one argument: an iterable of parameters whose
       gradients are to be synchronized.
    """

    param_sync_func: Callable
    """Function that launches asynchronous parameter synchronizations
    (e.g. distributed optimizer parameter all-gathers).
    The function should take one argument: an iterable of parameters to
       be synchronized.
    """

    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
    """If int, set the number of microbatches where not all of the layers
        will be checkpointed and recomputed.
        The rest of the microbatches within the window of maximum outstanding
       microbatches will recompute all layers
       (either full recompute or selective recompute). If
       None, the checkpoint and recompute
       will be left up to the forward_step function.
    """

    deallocate_pipeline_outputs: bool
    """If True, output data is deallocated after the tensor
        is sent to the next pipeline stage.
       Helps with saving memory,
       does nothing when pipeline parallel is not used.
    """

    finalize_model_grads_func: Optional[Callable] = None
    """Function that finalizes gradients on all workers.
        Could include ensuring that grads are
       all-reduced across data parallelism, pipeline parallelism,
       and sequence parallelism dimensions.
    """

    calculate_per_token_loss: bool
    """Whether cross entropy loss is calculated over
    the actual number of non-padded tokens in the global batch,
    versus the default behavior of assuming all tokens are non-padded.
    """
