# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
# reworked/refactored some parts to make it run.
import copy
import torch


class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1):
        super(Experts, self).__init__()

        self.experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # All-reduce moe-expert parameters across data modulo expert parallel nodes
        for expert in self.experts:
            for name, param in expert.named_parameters():
                param.allreduce = False

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            chunk = torch.squeeze(chunk, dim=1).contiguous()
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            out = torch.unsqueeze(out, dim=1)
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output