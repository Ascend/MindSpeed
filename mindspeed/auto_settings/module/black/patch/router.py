from functools import wraps

import torch
from megatron.training.global_vars import get_args


def router_forward_decorator(fn):

    @wraps(fn)
    def wrapper(self, input_data):
        prof_file = get_args().prof_file
        if prof_file:
            self.hidden = input_data.shape[-1]

            # Apply input jitter
            input_data = self.apply_input_jitter(input_data)
            logits = self.gating(input_data)
            print(f'self.config.num_moe_experts: {self.config.num_moe_experts}')
            logits = logits.view(-1, self.config.num_moe_experts)

            scores, indices = self.routing(logits)

            scores, full_pattern = force_load_balance(logits, scores, indices)
            print(f'scores shape: {(len(scores[0]), len(scores[1 ]))}')

            return scores, full_pattern

        scores, indices = fn(self, fn)

        return scores, indices

    return wrapper


def force_load_balance(logits, scores, indices):
    # 暂时仅支持aux_loss_load_balancing
    tmp_indices = indices
    if isinstance(indices, tuple):
        # MindSpeed将indices all_gather提前到aux_loss_load_balancing中做了
        tmp_indices, handle = indices
        handle.wait()

    expert_num = logits.shape[1]
    top_k = scores.shape[-1]
    num_total_tokens = tmp_indices.shape[0]

    list1 = [list(range(i, i + top_k)) for i in range(expert_num - top_k + 1)]
    list2 = [list(range(expert_num - top_k + 1, expert_num)) + [0]]
    full_pattern = torch.tensor(list1 + list2, device=logits.device)
    full_pattern = full_pattern.repeat((num_total_tokens // expert_num, 1))

    tokens_per_expert = [0 for _ in range(expert_num)]
    for ele in full_pattern.flatten():
        tokens_per_expert[int(ele.item())] += 1

    return scores, (full_pattern, handle) if isinstance(indices, tuple) else full_pattern