# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass
from typing import ClassVar, Dict  # codecheck_ignore
from enum import Enum
import math

import torch
import torch_npu


class TensorState(Enum):
    """ Define activation tensor status.
    """
    NORMAL = "normal"
    COMPRESS = "compress"


class ListNode:
    """ Utilize the linked list data structure to record 
    the dependencies between transformer layer computations.
    """
    def __init__(
        self, 
        order_layer_uuid, 
        prev_layer_node=None, 
        next_layer_node=None
    ) -> None:
        self.order_layer_uuid = order_layer_uuid
        self.next_layer_node = next_layer_node
        self.prev_layer_node = prev_layer_node
    
    def set_next_layer_node(self, next_layer_node) -> None:
        self.next_layer_node = next_layer_node
    
    def next(self):
        return self.next_layer_node
    
    def prev(self):
        return self.prev_layer_node


class ShareMemory:
    """ Class for managing shared swap tensor and shared PDF tensor.
    """
    def __init__(self, numel: int, dtype: torch.dtype) -> None:
        self.numel = numel
        self.dtype = dtype
        self.min_host_size = 2 * 1024 * 1024
        device = torch.empty([], device=torch.cuda.current_device()).device
        self.virtual_tensor = get_swap_tensor(numel, device, dtype)
        self.can_be_used = True
        self.pdf = torch.zeros(256, dtype=torch.int32, device=device)


def get_swap_tensor(ts_numel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ Return the swap tensor through the input attributes.

    Args:
        ts_numel: Swap tensor numel.
        device: Swap tensor device.
        dtype: Swap tensor dtype.

    Returns:
        swap_tensor: Return swap tensor.
    """
    if not hasattr(torch_npu, "empty_with_swapped_memory"): 
        raise ModuleNotFoundError("PTA dose not support this func, please update to latest version.")
    size = torch.Size([ts_numel])
    swap_tensor = torch_npu.empty_with_swapped_memory(size, dtype=dtype, device=device)
    swap_tensor.zero_()
    return swap_tensor


class TensorManager:
    """ Manages tensor compression/decompression with NPU-accelerated operator npu_hans_encode/npu_hans_decode.

    Core Responsibilities:
        - Memory allocation for compressed representations
        - State tracking for tensor lifecycle
        - NPU hardware acceleration for encode/decode operations
    """
    def __init__(self, tensor: torch.Tensor, compress_ratio: float = 0.5) -> None:
        self.tensor = tensor
        self.fixed_numel = (math.ceil(tensor.numel() * compress_ratio) // tensor.element_size() + 1) // 2 * 2
        self.mantissa_numel = tensor.numel() * (tensor.element_size() - 1) // self.tensor.element_size()
        self.storage_size = self.tensor.numel() * self.tensor.element_size()
        self.var = None
        self.fixed = None
        self.mantissa = None
        self.state = TensorState.NORMAL
        self.statistic = True

    def malloc(self, var: ShareMemory, statistic: bool) -> None:
        """ Allocate the required memory before executing the compression operator.

        Args:
            var: Share memory.
        """
        self.var = var
        self.statistic = statistic
        self.fixed = torch.zeros(
            self.fixed_numel, dtype=self.tensor.dtype, device=self.tensor.device)
        self.mantissa = torch.zeros(
            self.mantissa_numel, dtype=self.tensor.dtype, device=self.tensor.device)

    def encode(self) -> None:
        """ Asynchronous execution of compression task.
        """
        self.var.pdf, self.mantissa, self.fixed, self.var.virtual_tensor = torch_npu.npu_hans_encode(\
                        self.tensor, self.statistic, False, \
                        out=(self.var.pdf, self.mantissa, self.fixed, self.var.virtual_tensor))

    def encode_wait(self) -> None:
        """ Wait for the asynchronous task to complete compression, 
        then release the memory of the original activation values.
        """
        self.state = TensorState.COMPRESS
        self.tensor.untyped_storage().resize_(0)

    def pre_decode(self) -> None:
        """ Reapply for activation memory before decompression task.
        """
        self.tensor.untyped_storage().resize_(self.storage_size)

    def decode(self) -> None:
        """ Asynchronous execution of decompression task.
        """
        self.tensor = torch_npu.npu_hans_decode(self.mantissa, \
                    self.fixed, self.var.virtual_tensor, self.var.pdf, False, out=self.tensor)

    def release(self) -> None:
        """ After decompression, release all allocated memory.
        """
        if hasattr(self.var, "can_be_used"):
            self.var.can_be_used = True
        self.fixed = None
        self.mantissa = None
        self.var = None
        self.state = TensorState.NORMAL

    def recover(self) -> None:
        """ Synchronize and restore all activation, and release any excess memory.
        """
        self.pre_decode()
        self.decode()
        self.release()
    

@dataclass
class SimulationHyperParams:
    """ Hyperparameters for Time-Consuming Theoretical Modeling.
    """
    allgather_throughput: Dict[str, float]
    all2all_throughput: Dict[str, float]

    TFLOPS: ClassVar[int] = 10**12
    GIGABYTE: ClassVar[int] = 1024 ** 3
    MAX_BANDWIDTH: ClassVar[int] = 1000 * GIGABYTE
    encode_throughput: float = 100.0 * GIGABYTE
    decode_throughput: float = 111.0 * GIGABYTE
    cube_tflops: float = 280.0 * TFLOPS 


class SimulationBase:
    """ Used for modeling various asynchronous operators.
    """
    def __init__(self, simulation_config: SimulationHyperParams) -> None:
        self.simulation_config = simulation_config

    def time_cost(self, op_name: str, *args, **kwargs) -> float:
        if op_name == "matmul":
            return self._matmul(*args, **kwargs) 
        elif op_name == "all2all":
            return self._all2all(*args, **kwargs)
        elif op_name == "allgather":
            return self._allgather(*args, **kwargs)
        else:
            return 0
            
    def _matmul(self, *args, **kwargs) -> float:
        """ Matmul time cost.
        """
        output_shape = infer_matmul_shape(args[0], args[1])
        total_flop = 2 * args[0].shape[-1]
        for dim in output_shape:
            total_flop *= dim
        return total_flop / self.simulation_config.cube_tflops

    def _all2all(self, *args, **kwargs) -> float:
        """ All2All time cost.
        """
        if not kwargs.get("group", False):
            return 0
        group_size = torch.distributed.get_world_size(kwargs["group"])
        simulation_bandwidth = self.simulation_config.all2all_throughput.get(
            str(group_size), self.simulation_config.MAX_BANDWIDTH)
        return args[0].numel() * args[0].element_size() / simulation_bandwidth

    def _allgather(self, *args, **kwargs) -> float:
        """ AllGather time cost.
        """            
        if not kwargs.get("group", False):
            return 0
        group_size = torch.distributed.get_world_size(kwargs["group"])
        simulation_bandwidth = self.simulation_config.allgather_throughput.get(
            str(group_size), self.simulation_config.MAX_BANDWIDTH)
        return args[0].numel() * args[0].element_size() / simulation_bandwidth

    def _reducescatter(self, *args, **kwargs) -> float:
        raise NotImplementedError
    
    def encode_max_numel(self, estimated_time) -> int:
        return int(self.simulation_config.encode_throughput * estimated_time / 2)
    
    def decode_max_numel(self, estimated_time) -> int:
        return int(self.simulation_config.decode_throughput * estimated_time / 2)


class SimulationA2(SimulationBase):
    def __init__(self):
        cfg = SimulationHyperParams(
            allgather_throughput={
                "2": 36 * SimulationHyperParams.GIGABYTE,
                "4": 73 * SimulationHyperParams.GIGABYTE,
                "8": 147 * SimulationHyperParams.GIGABYTE,
                "16": 138 * SimulationHyperParams.GIGABYTE,
            },
            all2all_throughput={
                "2": 37 * SimulationHyperParams.GIGABYTE,
                "4": 69 * SimulationHyperParams.GIGABYTE,
                "8": 119 * SimulationHyperParams.GIGABYTE,
                "16": 40.1 * SimulationHyperParams.GIGABYTE,
                "32": 30.2 * SimulationHyperParams.GIGABYTE,
                "64": 27.0 * SimulationHyperParams.GIGABYTE,
            }
        )
        super().__init__(cfg)


class SimulationA3(SimulationBase):
    def __init__(self):
        cfg = SimulationHyperParams(
            allgather_throughput={
                "2": 350.0 * SimulationHyperParams.GIGABYTE,
                "4": 324.9 * SimulationHyperParams.GIGABYTE,
                "8": 298.8 * SimulationHyperParams.GIGABYTE,
                "16": 283.6 * SimulationHyperParams.GIGABYTE,
            },
            all2all_throughput={
                "2": 0 * SimulationHyperParams.GIGABYTE,
                "4": 229.4 * SimulationHyperParams.GIGABYTE,
                "8": 173.9 * SimulationHyperParams.GIGABYTE,
                "16": 154.0 * SimulationHyperParams.GIGABYTE,
                "32": 143.9 * SimulationHyperParams.GIGABYTE,
                "64": 137.3 * SimulationHyperParams.GIGABYTE,
            }
        )
        super().__init__(cfg)


def infer_matmul_shape(A: torch.Tensor, B: torch.Tensor):
    a_shape = list(A.shape)
    b_shape = list(B.shape)

    a_was_1d = False
    b_was_1d = False

    if A.dim() == 1:
        a_shape = [1, a_shape[0]]
        a_was_1d = True
    if B.dim() == 1:
        b_shape = [b_shape[0], 1]
        b_was_1d = True

    if a_shape[-1] != b_shape[-2]:
        raise ValueError(f"Incompatible shapes: {A.shape} @ {B.shape}")

    batch_a = a_shape[:-2]
    batch_b = b_shape[:-2]
    try:
        broadcast_batch = torch.broadcast_shapes(tuple(batch_a), tuple(batch_b))
    except RuntimeError as e:
        raise ValueError(f"Cannot broadcast batch dimensions: {batch_a} vs {batch_b}") from e

    m = a_shape[-2]
    n = b_shape[-1]

    out_shape = list(broadcast_batch) + [m, n]
    if a_was_1d and b_was_1d:
        return tuple(out_shape[:-2])  # scalar output
    elif a_was_1d:
        return tuple(out_shape[:-2] + [n])
    elif b_was_1d:
        return tuple(out_shape[:-2] + [m])
    else:
        return tuple(out_shape)
