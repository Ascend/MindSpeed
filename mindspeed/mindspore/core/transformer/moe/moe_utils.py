# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from torch.autograd import recompute_instance
import mindspore


def forward_func(func, inputs):
    def detach_tensor(input_):
        if input_.requires_grad and input_.grad_fn is None:
            return input_
        new_input = input_.detach()
        new_input.requires_grad = True
        return new_input

    def deach_helper(tensor):
        return detach_tensor(tensor) if torch.is_floating_point(tensor) else tensor

    def recursive_detach(obj):
        """递归剥离嵌套结构中的浮点张量梯度"""
        if isinstance(obj, tuple):
            # 保持元组类型，递归处理每个元素
            return tuple(recursive_detach(e) for e in obj)
        elif isinstance(obj, list):
            # 保持列表类型，递归处理每个元素
            return [recursive_detach(e) for e in obj]
        elif isinstance(obj, torch.Tensor):
            # 自定义的 detach_tensor 函数
            return deach_helper(obj)
        else:
            # 非张量或非浮点张量直接返回
            return obj

    # 主逻辑：确保输出结构与原代码一致（列表包裹）
    if isinstance(inputs, (tuple, list)):
        # 处理元组/列表输入：递归剥离后转列表（与原逻辑一致）
        detach_inputs = list(recursive_detach(inputs))
    elif isinstance(inputs, torch.Tensor):
        # 处理张量输入：包裹为列表 [detached_tensor]
        detach_inputs = [recursive_detach(inputs)]
    else:
        # 其他类型（如整数、字符串）按需处理，此处包裹为列表保持一致性
        detach_inputs = [inputs]
    if not recompute_instance.recompute:
        with torch.enable_grad():
            output, f_vjp = torch.autograd.vjp(func, *detach_inputs)
    else:
        output = func(*detach_inputs)
        f_vjp = None

    return output, *detach_inputs, f_vjp