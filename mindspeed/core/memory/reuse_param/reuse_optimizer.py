import os
import types
from functools import wraps

import torch

TRANSPOSE_BF16_BIT = 32768


def _copy_model_params_to_main_params():
    pass


class ConvertFp32BF16:
    def init_and_reuse_storage_of_tensors(
            self,
            fp32_tensor,
            bf16_fp32_tensor,
            res_tensor,
            bf16_tensor,
            int32_tensor
    ):
        """
        init a list of tensor with length of 2*fp32_tensor.numel() in bf16 to share the same storage.
        Args:
            fp32_tensor: original fp32 tensor.
            bf16_fp32_tensor: a bf16 tensor share the same storage with original list of fp32 tensors.
            res_tensor: a bf16 tensor that store the residual value of fp32 to bf16, shares a half of the
            storage with bf16_fp32_tensor.
            bf16_tensor: a bf16 tensor that store the value from fp32, shares another half of the
            storage with bf16_fp32_tensor.
            int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
        """
        from mindspeed.op_builder import AlgorithmOpBuilder
        reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
        reuse_data_ptr(bf16_fp32_tensor, fp32_tensor, 0)
        reuse_data_ptr(int32_tensor, fp32_tensor, 0)
        self.fp32_tensors_to_bf16_tensors([int32_tensor], [bf16_fp32_tensor])
        reuse_data_ptr(res_tensor, bf16_fp32_tensor, 0)
        reuse_data_ptr(bf16_tensor, bf16_fp32_tensor, res_tensor.numel())

    def fp16_tensor_convert_to_fp32_tensor_deterministic(self, ctx):
        for int32_float32_group, float16_param_group, fp32_from_float16_group in zip(
            ctx.int32_float32_groups, ctx.float16_float32_groups, ctx.fp32_from_float16_groups):
            self.bf16_tensors_to_fp32_tensors_deterministic(int32_float32_group, float16_param_group, fp32_from_float16_group, ctx.optimizer)

    def fp32_tensor_convert_to_fp16_tensor_deterministic(self, ctx):
        for int32_float32_param_group, float16_param_group, fp32_from_float16_group in zip(
            ctx.int32_float32_groups, ctx.float16_float32_groups, ctx.fp32_from_float16_groups):
            self.fp32_tensors_to_bf16_tensors_deterministic(int32_float32_param_group, float16_param_group, fp32_from_float16_group, ctx.optimizer)

    def fp16_tensor_convert_to_fp32_tensor(self, ctx):
        for int32_float32_group, float16_param_group in zip(
                ctx.int32_float32_groups, ctx.float16_float32_groups):
            self.bf16_tensors_to_fp32_tensors(int32_float32_group, float16_param_group)

    def fp32_tensor_convert_to_fp16_tensor(self, ctx):
        for int32_float32_param_group, float16_param_group in zip(
            ctx.int32_float32_groups, ctx.float16_float32_groups):
            self.fp32_tensors_to_bf16_tensors(int32_float32_param_group, float16_param_group)

    def fp32_tensors_to_bf16_tensors(self, int32_tensors, bf16_fp32_tensors):
        """
        fp32(0p0p0p0p) -> bf16(pppp) + res(0000)
        rearrange the storage of bf16_fp32_tensors so that recover the fp32_tensors.
        Args:
            int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
            bf16_fp32_tensors: a list of bf16 tensors share the same storages with original list of fp32 tensors.
        Returns:
            None
        """
        for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return  
            int32_tensor.add_(TRANSPOSE_BF16_BIT)
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())

    def bf16_tensors_to_fp32_tensors(self, int32_tensors, bf16_fp32_tensors):
        """
        res(0000) + bf16(pppp) -> fp32(0p0p0p0p)
        rearrange the storage of bf16_fp32_tensors so that recover the fp32_tensors.
        Args:
            int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
            bf16_fp32_tensors: a list of bf16 tensors share the same storages with original list of fp32 tensors.
        Returns:
            None
        """
        for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
            int32_tensor.sub_(TRANSPOSE_BF16_BIT)

    def fp32_tensors_to_bf16_tensors_deterministic(self, int32_tensors, bf16_fp32_tensors, fp32_tensors, optimizer):
        for int32_tensor, bf16_fp32_tensor, fp32_tensor in zip(int32_tensors, bf16_fp32_tensors, fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return  
            odd_even_tensor = ((int32_tensor & 131071) == TRANSPOSE_BF16_BIT).int()
            int32_tensor.add_(TRANSPOSE_BF16_BIT)
            self.optimizer_exp_avg_save_sign(optimizer, fp32_tensor, int32_tensor, odd_even_tensor)
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())

    def bf16_tensors_to_fp32_tensors_deterministic(self, int32_tensors, bf16_fp32_tensors, fp32_tensors, optimizer):
        for int32_tensor, bf16_fp32_tensor, fp32_tensor in zip(int32_tensors, bf16_fp32_tensors, fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
            self.optimizer_exp_avg_load_sign(optimizer, fp32_tensor, int32_tensor)
            int32_tensor.sub_(TRANSPOSE_BF16_BIT)

    def optimizer_exp_avg_save_sign(self, optimizer, fp32_param, int32_tensor, odd_even_tensor):
        if "exp_avg_sq" in optimizer.state[fp32_param]:
            int32_tensor.sub_(odd_even_tensor)
            sign_tensor = torch.sign(odd_even_tensor - 0.5).reshape(optimizer.state[fp32_param]["exp_avg_sq"].shape)
            optimizer.state[fp32_param]["exp_avg_sq"].mul_(sign_tensor)

    def optimizer_exp_avg_load_sign(self, optimizer, fp32_param, int32_tensor):
        if "exp_avg_sq" in optimizer.state[fp32_param]:
            odd_even_tensor = (torch.sign(optimizer.state[fp32_param]["exp_avg_sq"]) > 0).reshape(-1)
            optimizer.state[fp32_param]["exp_avg_sq"].abs_()
            int32_tensor.add_(odd_even_tensor)