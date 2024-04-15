import math
import threading
from functools import wraps

import torch
import torch_npu
import torch.nn.functional as F
from megatron import get_args, core
from megatron.core import parallel_state, mpu, tensor_parallel
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.transformer import ParallelMLP

from ascendspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention


try:
    from einops import rearrange
except ImportError:
    rearrange = None


class Alibi:
    _instance = None
    alibi = None
    matmul_result = None
    output_size = None
    lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        else:
            with cls.lock:
                cls._instance = super().__new__(cls)
                return cls._instance


def _get_inverted_mask(attention_mask, alibi):
    inverted_mask = attention_mask.to(alibi.dtype)
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), float("-inf")
    )
    return inverted_mask.to(alibi.device) + alibi.unsqueeze(0)


def _build_alibi_tensor(max_seq_len, num_attention_heads, square_alibi_mask, fill_neg_inf):
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

    def _fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def _buffered_future_mask(maxpos, alibi, attn_heads):
        _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
        _future_mask = _future_mask.unsqueeze(0) + alibi
        return _future_mask[:attn_heads, :maxpos, :maxpos]

    slopes = torch.Tensor(get_slopes(num_attention_heads))
    if square_alibi_mask:
        position_point = torch.arange(max_seq_len) - max_seq_len + 1
        position_point = position_point.unsqueeze(0).unsqueeze(0).expand(num_attention_heads, max_seq_len, -1)
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    else:
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

    # Select the part of the tensor that corresponds to our tensor parallel index.
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_index = parallel_state.get_tensor_model_parallel_rank()
    alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

    if fill_neg_inf:
        return _buffered_future_mask(max_seq_len, alibi, num_attention_heads)

    return alibi


def core_attention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        self.hidden_size_per_partition = self.hidden_size_per_partition // args.context_parallel_size
        self.square_alibi_mask = args.square_alibi_mask
        self.fill_neg_inf = args.fill_neg_inf
        self.beta = 1.0
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        if args.position_embedding_type == 'alibi':
            self.alibi = Alibi()
            alibi = _build_alibi_tensor(args.seq_length,
                                        args.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi
        else:
            self.alibi = None

    return wrapper


def core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                   query_layer.size(2),
                   query_layer.size(0),
                   key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.reshape(output_size[2],
                                      output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                               output_size[0] * output_size[1], -1)

    if self.alibi is None:
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),
            key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0, alpha=(1.0 / self.norm_factor))
    else:
        if self.alibi.matmul_result is None or self.alibi.output_size != output_size:
            args = get_args()
            
            self.alibi.output_size = output_size
            alibi = _build_alibi_tensor(args.seq_length,
                                        args.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi
            
            if self.fill_neg_inf:
                _alibi = self.alibi.alibi[:, :output_size[3], :output_size[3]]
                attention_mask = attention_mask.repeat(output_size[0], 1, 1, 1)[:output_size[0], :, :, :]
                self.alibi.matmul_result = _get_inverted_mask(attention_mask, _alibi).view(-1, output_size[2],
                                                                                           output_size[2]).contiguous()
            else:
                self.alibi.matmul_result = self.alibi.alibi[:, :, :output_size[3]].repeat(output_size[0], 1, 1)

        q_trans = query_layer.transpose(0, 1).contiguous()
        k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
        matmul_result = self.beta * self.alibi.matmul_result + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

        # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    if self.square_alibi_mask:
        attention_scores = torch.max(
            attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
        )
        attention_probs = torch.nn.functional.softmax(attention_scores, -1)
    else:
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                   value_layer.size(2),
                   query_layer.size(0),
                   value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                   output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                           output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
                              (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


def parallel_mlp_init(self, config, is_expert=False):
    super(ParallelMLP, self).__init__()
    args = get_args()

    self.add_bias = config.add_bias_linear

    ffn_hidden_size = config.ffn_hidden_size
    if config.gated_linear_unit:
        ffn_hidden_size *= 2

    # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
    self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
        config.hidden_size,
        ffn_hidden_size,
        config=config,
        init_method=config.init_method,
        bias=self.add_bias,
        gather_output=False,
        skip_bias_add=True,
        is_expert=is_expert,
    )

    self.bias_gelu_fusion = False
    self.activation_func = None
    self.swiglu = args.swiglu

    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu
    elif args.swiglu:
        def swiglu(x):
            if args.use_fused_swiglu:
                return torch_npu.npu_swiglu(x, dim=-1)
            else:
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
        self.activation_func = swiglu
    elif args.squared_relu:
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        self.activation_func = squared_relu
    else:
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu

    # Project back to h.
    self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
        config.ffn_hidden_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=self.add_bias,
        skip_bias_add=True,
        input_is_parallel=True,
        is_expert=is_expert,
    )


def flash_self_attention_forward(self, q, k, v):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
    """
    args = get_args()

    batch_size, seq_length, head_num, head_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

    if not hasattr(self, 'attention_mask'):
        self.attention_mask = (torch.tril(torch.ones([seq_length, seq_length]), diagonal=-(args.pre_tockens + 1)) \
                + torch.triu(torch.ones([seq_length, seq_length]), diagonal=args.next_tockens + 1)).bool().npu()
    if args.shape_order == 'BSH':
        q, k, v = [rearrange(x, 'b s h d -> b s (h d)') for x in [q, k, v]]
    elif args.shape_order == 'SBH':
        q, k, v = [rearrange(x, 'b s h d -> s b (h d)') for x in [q, k, v]]
    elif args.shape_order != 'BSND':
        raise ValueError('Invalid shape-order: {}, shape-order must be SBH or BSH or BSND'.format(args.shape_order))

    try:
        scale = 1.0 / math.sqrt(head_dim) if self.softmax_scale is None else self.softmax_scale
    except Exception as e:
        raise ValueError('Invalid head_dim: {}'.format(head_dim)) from e

    output = torch_npu.npu_fusion_attention( \
        q, k, v, head_num, args.shape_order, \
        pse=None, \
        padding_mask=None, \
        atten_mask=self.attention_mask, \
        scale=scale, \
        pre_tockens=args.pre_tockens, \
        next_tockens=args.next_tockens, \
        keep_prob=1 - self.dropout_p, \
        inner_precise=0
    )[0]

    if args.shape_order == 'BSH':
        output = rearrange(output, 'b s (h d) -> b s h d', h=head_num)
    elif args.shape_order == 'SBH':
        output = rearrange(output, 's b (h d) -> b s h d', h=head_num)
    elif args.shape_order != 'BSND':
        raise ValueError('Invalid shape-order: {}, shape-order must be SBH or BSH or BSND'.format(args.shape_order))

    return output


def ParallelAttention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        config = args[0]
        query_projection_size = config.kv_channels * config.num_attention_heads
        _args = get_args()
        if _args.group_query_attention:
            kv_projection_size = _args.kv_channels * _args.num_query_groups
        else:
            kv_projection_size = _args.kv_channels * _args.num_attention_heads
        # qkv bias
        bias = _args.add_qkv_bias or _args.add_bias_linear
        if args[0].context_parallel_size > 1 and args[0].context_parallel_algo == 'ulysses_cp_algo':
            if self.use_flash_attn:
                self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, mpu.get_context_parallel_group())
            else:
                self.core_attention = UlyssesContextAttention(self.core_attention, mpu.get_context_parallel_group())
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False)
        # dense bias
        bias = _args.add_dense_bias or _args.add_bias_linear
        skip_bias_add = _args.skip_bias_add
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add)
    return wrapper
