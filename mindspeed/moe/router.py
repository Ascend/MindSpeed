# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
import torch


def gating(self, input: torch.Tensor):
    """Forward pass of the router gate.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Logits tensor.
    """
    if self.weight.device.type == 'cpu':
        # move weights to GPU
        self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
    # Convert to specified datatype for routing computation if enabled
    router_dtype = input.dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    logits = torch.nn.functional.linear(input.to(router_dtype), self.weight)
    return logits
