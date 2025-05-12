# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Optional, List, Callable, Union
import argparse
import math
from functools import partial
import torch
import torch_npu


def get_swap_tensor(ts_numel: int, device: torch.device, dtype: torch.dtype):
    if not hasattr(torch_npu, "empty_with_swapped_memory"): 
        raise ModuleNotFoundError("PTA dose not support this func, please update to latest version.")
    size = torch.Size([ts_numel])
    swap_tensor = torch_npu.empty_with_swapped_memory(size, dtype=dtype, device=device)
    swap_tensor.zero_()
    return swap_tensor


class ShareMemory:
    """Get virtual memory.""" 
    def __init__(self, numel: int, dtype: torch.dtype) -> None:
        self.numel = numel
        self.dtype = dtype
        self.min_host_size = 2 * 1024 * 1024
        device = torch.empty([], device=torch.cuda.current_device()).device
        self.virtual_tensor = get_swap_tensor(numel, device, dtype)
        self.can_be_used = True


class CompressTensor:
    """Compression and decompression for tensors.
    """    
    def __init__(self, tensor: torch.Tensor, pdf: torch.Tensor, 
                compress_ratio: float, stream: torch.Stream, idx: int) -> None:
        self.tensor = tensor
        self.pdf = pdf
        self.fixed_numel = (math.ceil(tensor.numel() * compress_ratio) // tensor.element_size() + 1) // 2 * 2
        self.storage_size = self.tensor.numel() * self.tensor.element_size()
        self.shape = tensor.shape
        self.hans_stream = stream
        self.default_stream = torch.cuda.default_stream()
        self.encode_event = torch.npu.Event()
        self.decode_event = torch.npu.Event()
        self.fixed = None
        self.idx = idx


    def alloc_init(self, tensor: torch.Tensor, mantissa: torch.Tensor, var: torch.Tensor, statistic: bool):
        self.tensor = tensor
        self.mantissa = mantissa
        self.var = var
        self.statistic = statistic
        self.fixed = torch.zeros(self.fixed_numel, dtype=tensor.dtype, device=tensor.device)


    def encode(self):
        self.hans_stream.wait_stream(self.default_stream)
        with torch.npu.stream(self.hans_stream):
            if hasattr(self.mantissa, "virtual_tensor"):
                self.pdf, self.mantissa.virtual_tensor, self.fixed, self.var.virtual_tensor = torch_npu.npu_hans_encode(\
                        self.tensor, self.statistic, False, \
                        out=(self.pdf, self.mantissa.virtual_tensor, self.fixed, self.var.virtual_tensor))
            else:
                self.pdf, self.mantissa, self.fixed, self.var.virtual_tensor = torch_npu.npu_hans_encode(\
                        self.tensor, self.statistic, False, \
                        out=(self.pdf, self.mantissa, self.fixed, self.var.virtual_tensor))
            self.encode_event.record()

    def wait_encode(self):
        self.default_stream.wait_event(self.encode_event)
        self.tensor.untyped_storage().resize_(0)

    def decode(self):
        self.hans_stream.wait_stream(self.default_stream)
        self.tensor.untyped_storage().resize_(self.storage_size)
        with torch.npu.stream(self.hans_stream):
            if hasattr(self.mantissa, "virtual_tensor"):
                self.tensor = torch_npu.npu_hans_decode(self.mantissa.virtual_tensor, \
                    self.fixed, self.var.virtual_tensor, self.pdf, False, out=self.tensor)
            else:
                self.tensor = torch_npu.npu_hans_decode(self.mantissa, \
                    self.fixed, self.var.virtual_tensor, self.pdf, False, out=self.tensor)
            self.decode_event.record()

    def wait_decode(self):
        self.default_stream.wait_event(self.decode_event)

        if hasattr(self.mantissa, "can_be_used"):
            self.mantissa.can_be_used = True
        else:
            self.mantissa = None
        self.var.can_be_used = True
        self.fixed = None
        self.tensor = None


class CTM:
    """Compress tensor object management: shared memory reuse management and compression stream control.
    """    
    def __init__(self, train_args: argparse.Namespace, swap_mantissa: bool = True):
        self.activations = []
        self.act_length = None
        self.fwd_order = -1
        self.bwd_index = -1
        self.bwd_order_chain = []
        self.accumulate_flag = False
        self.cts = []
        self.share_memorys = []
        self.hans_stream = torch.npu.Stream()
        self.pdf = torch.zeros(256, dtype=torch.int32, device=torch.cuda.current_device())
        self.statisticed = False
        self.compress_ratio = 0.5
        self.debug_mode = False
        self.swap_mantissa = swap_mantissa
        self.train_args = train_args

    @property
    def step(self):
        return self.train_args.curr_iteration - self.train_args.iteration

    def fwd_order_update(self):
        self.fwd_order += 1
        if not self.accumulate_flag and self.step == 1:
            self.act_length = self.fwd_order
            self.accumulate_flag = True
        if self.act_length is not None and self.fwd_order == self.act_length:
            self.fwd_order = 0

    def bwd_order_update(self):
        self.bwd_index += 1
        if self.act_length is not None and self.bwd_index == self.act_length:
            self.bwd_index = 0

    def save_activation(self, act: torch.Tensor):
        if self.step == 0:
            self.activations.append(act)
            self._compress(self.fwd_order)
        else:
            self.activations[self.fwd_order] = act
        return self.fwd_order

    def pop_activation(self, order: int, _grad):
        self.bwd_order_update()
        if self.step == 0:
            self.bwd_order_chain.append(order)
            self._decompress(order)
        else:
            cur_idx = self.bwd_order_chain[self.bwd_index]
            if self.activations[cur_idx] is not None and \
                self.activations[cur_idx].untyped_storage().size() == 0:
                self._decompress(cur_idx, async_op=False)
            self.activations[cur_idx] = None

    def decompress_normal_hook(self, _grad):
        if self.step == 0:
            return
        order_index = self.bwd_index + 1
        if order_index > len(self.bwd_order_chain) - 1:
            return
        order = self.bwd_order_chain[order_index]
        if self.cts[order] is None or self.cts[order].fixed is None:
            return
        self._decompress(order, async_op=True)

    def decompress_wait_hook(self, _grad):
        if self.step == 0:
            return
        order_index = self.bwd_index + 1
        if order_index > len(self.bwd_order_chain) - 1:
            return
        order = self.bwd_order_chain[order_index]
        if self.cts[order] is None or self.cts[order].fixed is None:
            return
        self.cts[order].wait_decode()
        self.activations[order] = None

    def compress(self):
        self.fwd_order_update()
        if self.step == 0:
            return
        last_order = self.fwd_order - 1
        if last_order < 0 or self.activations[last_order] is None:
            return
        self._compress(last_order, async_op=True)

    def compress_wait(self):
        if self.step == 0:
            return
        last_order = self.fwd_order - 1
        if last_order < 0 or self.activations[last_order] is None:
            return
        self.cts[last_order].wait_encode()

    def _decompress(self, order: int, async_op: bool = False):
        ct = self.cts[order]
        ct.decode()
        if not async_op:
            ct.wait_decode()
            self.activations[order] = None

    def _compress(self, order: int, async_op: bool = False):
        act = self.activations[order]
        if self.step == 0:
            ct = CompressTensor(act, self.pdf, self.compress_ratio, self.hans_stream, len(self.cts))
            self.cts.append(ct)
        else:
            ct = self.cts[order]
        ct.alloc_init(
            act,
            self.get_mantissa_sm(act),
            self.get_var_sm(act),
            self.get_statistic()
        )
        ct.encode()
        if not async_op:
            ct.wait_encode()

    def get_statistic(self):
        if not self.statisticed:
            self.statisticed = True
            return True
        else:
            return False

    def get_mantissa_sm(self, act: torch.Tensor):
        mantissa_numel = act.numel() * (act.element_size() - 1)
        if self.swap_mantissa:
            return self.get_sm(mantissa_numel // act.element_size(), act.dtype)
        else:
            return torch.zeros(mantissa_numel // act.element_size(), dtype=act.dtype, device=act.device)

    def get_var_sm(self, act: torch.Tensor):
        return self.get_sm(act.numel() // act.element_size(), act.dtype)

    def get_sm(self, numel: int, dtype: torch.dtype):
        for sm in self.share_memorys:
            if sm.can_be_used and numel == sm.numel and dtype == sm.dtype:
                sm.can_be_used = False
                return sm
        sm = ShareMemory(numel, dtype)
        sm.can_be_used = False
        self.share_memorys.append(sm)
        return sm


class ActivationCompress:
    """ Manages asynchronous activation compression/decompression during training.
        Implements a memory optimization with overlaping compression/decompression and matmul operations.
    """    
    def __init__(self, train_args: argparse.Namespace, ctm_name: str, filters: Union[List[bool], None] = None):
        """
        Initializes an ActivationCompress instance.

        Args:
            train_args (argparse.Namespace): Training arguments configuration.
            ctm_name (str): Customize ctm_name to enable compress tensor manager to adapt to different layer types.
            filters (List[bool], optional): Condition filters applied during compression. Defaults to None.
        """
        self.args = train_args
        self.ac_name = ctm_name
        if filters is None:
            filters = []
        self.flag = train_args.compress_dense != "disable" and all(filters)
        if self.flag and not hasattr(train_args, ctm_name):
            swap_mantissa = train_args.compress_dense == "level1"
            setattr(train_args, ctm_name, CTM(train_args, swap_mantissa))
        self.ctm = getattr(self.args, self.ac_name, None)
        self.target_tensor = None
    
    def compress_and_wait_decompress_async_for_previous_layer(self, ts_in: torch.Tensor):
        """
            Asynchronously compress activation of the previous layer during the forward pass and 
            register hooks to trigger wait decompression during backpropagation.

            Args:
                ts_in (torch.Tensor): Input tensor_in for registe decompress_wait_hook.
        """
        if self.flag and self.ctm is not None:
            if isinstance(ts_in, torch.Tensor) and ts_in.requires_grad:
                ts_in.register_hook(self.ctm.decompress_wait_hook)
            elif isinstance(ts_in, tuple) and ts_in[0].requires_grad:
                ts_in[0].register_hook(self.ctm.decompress_wait_hook)
            else:
                raise TypeError("tensor for compress msut be tensor or tuple")
            self.ctm.compress()
        
    def decompress_and_wait_compress_async_for_previous_layer(self, target_ts: torch.Tensor):
        """
            Asynchronously wait compress activation of the previous layer during the forward pass and 
            register hooks to trigger decompression during backpropagation.

            Args:
                target_ts (torch.Tensor): Input activation for compress/decompress.
        """
        if self.flag and self.ctm is not None and target_ts.requires_grad:
            self.target_tensor = target_ts
            self.ctm.compress_wait()
            target_ts.register_hook(self.ctm.decompress_normal_hook)

    
    def order_record(self, ts_out: torch.Tensor):
        """
            In the first training step, record the order of non-asynchronous compression and decompression 
            during both the forward and backward passes to handle complex backward computation graphs.

            Args:
                ts_out (torch.Tensor): tensor_out for registe decompress order record hook.
        """
        if self.flag and self.ctm is not None and ts_out.requires_grad:
            order = self.ctm.save_activation(self.target_tensor)
            order_func = partial(self.ctm.pop_activation, order)
            ts_out.register_hook(order_func)
            self.target_tensor = None
