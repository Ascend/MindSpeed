import math
import torch

from megatron.core import parallel_state


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n - closest_power_of_2]


class AlibiForFusionAttnSingleton:
    _alibi_tensor_args = None
    _alibi_tensor = None

    _alibi_slopes_headnum = None
    _alibi_slopes = None

    @classmethod
    def get_alibi_tensor_for_fusion_attn(cls, max_seq_len, num_attention_heads, dtype, neg_diagonal_opposite=False,
                                         last_k=1024):
        if cls._alibi_tensor is None or cls._alibi_tensor_args != (
        max_seq_len, num_attention_heads, neg_diagonal_opposite, last_k):
            if last_k > max_seq_len:
                last_k = max_seq_len

            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            current_head_num = num_attention_heads // tp_world_size
            slopes = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(num_attention_heads)

            position_point = torch.arange(max_seq_len) - max_seq_len + 1
            diag = torch.diag(torch.diag(position_point)).unsqueeze(0).unsqueeze(0)

            position_point = position_point.unsqueeze(0).unsqueeze(0).expand(current_head_num, last_k, -1)
            position_point = position_point - diag.transpose(-1, -2)[:, -last_k:, :].expand(current_head_num, last_k,
                                                                                            max_seq_len)

            alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point.npu()

            if not neg_diagonal_opposite:
                alibi = -torch.abs(alibi)

            alibi = alibi.unsqueeze(0)
            alibi = torch.Tensor(alibi).npu()
            if dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)

            cls._alibi_tensor = alibi
            cls._alibi_tensor_args = (max_seq_len, num_attention_heads, neg_diagonal_opposite, last_k)

        return cls._alibi_tensor

    @classmethod
    def get_alibi_slopes_for_fusion_attn(cls, n):
        if cls._alibi_slopes is None or cls._alibi_slopes_headnum != n:
            slopes = get_slopes(n)

            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_index = parallel_state.get_tensor_model_parallel_rank()

            current_head_num = n // tp_world_size
            slopes = torch.Tensor(slopes[tp_index * current_head_num: tp_index * current_head_num + current_head_num]).npu()

            cls._alibi_slopes = slopes
            cls._alibi_slopes_headnum = n
            return cls._alibi_slopes

        return cls._alibi_slopes
