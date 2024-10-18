import pytest
import torch
import torch_npu
from mindspeed import megatron_adaptor # noqa

from megatron.training.global_vars import set_args # noqa
from megatron.training.arguments import parse_args # noqa
from megatron.core import mpu # noqa
from megatron.core.transformer.transformer_config import TransformerConfig # noqa
from megatron.core.transformer.dot_product_attention import DotProductAttention # noqa
from megatron.core.tensor_parallel import RowParallelLinear, ColumnParallelLinear

from commons import set_random_seed, initialize_model_parallel # noqa
from unit_tests.common import DistributedTest # noqa
from mindspeed.model.transformer import get_attention_mask, set_attention_mask

from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
from mindspeed.core.training import get_flops_counter

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def set_mc2_args(args):
    args.tensor_model_parallel_size = 2
    args.use_unpad = False
    args.seed = 2024
    args.seq_len = 128
    args.input_size_coeff = 128
    args.output_size_coeff = 128
    args.batch_size = 8
    args.optimize_recomp_communication_level = True
    args.sequence_parallel = 1
    args.use_cp_send_recv_overlap = False
    return args


def run_mc2_tflops(world_size):
    args = parse_args(None, True)
    args = set_mc2_args(args)
    set_args(args)
    args.use_ascend_mc2 = True
    initialize_cfg_from_args(args)
    transformer_config = TransformerConfig(num_layers=1,
                                           hidden_size=12,
                                           num_attention_heads=4,
                                           use_cpu_initialization=True)
    transformer_config.sequence_parallel = args.sequence_parallel
    initialize_model_parallel(args.tensor_model_parallel_size, 1)
    set_random_seed(args.seed)
    input_size = args.input_size_coeff * args.tensor_model_parallel_size
    output_size = args.output_size_coeff * args.tensor_model_parallel_size

    flops_counter = get_flops_counter()
    flops_counter.start()

    linear_layer_col = ColumnParallelLinear(input_size,
                                        output_size,
                                        keep_master_weight_for_test=True,
                                        init_method=transformer_config.init_method,
                                        config=transformer_config).half().npu()

    input_ = torch.rand(args.batch_size, args.seq_len, input_size).half().npu()
    output = linear_layer_col(input_)

    counts_mc2_col = flops_counter.get_flops()
    flops_counter.stop()
    flops_counter.start()

    gather_list = [torch.zeros(input_.shape).half().npu() for _ in range(world_size)]
    torch.distributed.all_gather(gather_list, input_)
    gather_res = torch.concat(gather_list, dim=0)
    output_naive = torch.matmul(gather_res, linear_layer_col.weight.t())
    counts_mm_col = flops_counter.get_flops()
    flops_counter.stop()

    assert counts_mm_col[0] == counts_mc2_col[0]

    args.seq_len = 256
    args.output_size_coeff = 256
    args.input_size_coeff = 256
    input_size = args.input_size_coeff * args.tensor_model_parallel_size
    output_size = args.output_size_coeff * args.tensor_model_parallel_size

    flops_counter.start()
    linear_layer_row = RowParallelLinear(input_size,
                                     output_size,
                                     keep_master_weight_for_test=True,
                                     bias=True, input_is_parallel=True,
                                     skip_bias_add=False,
                                     init_method=transformer_config.init_method,
                                     config=transformer_config).half().npu()
    input_ = torch.rand(args.batch_size, args.seq_len, args.input_size_coeff)
    input_ = input_.half().npu()
    output = linear_layer_row(input_)

    counts_mc2_row = flops_counter.get_flops()
    flops_counter.stop()
    flops_counter.start()

    res = torch.matmul(input_, linear_layer_row.weight.npu().T)
    tensor = torch.empty(args.batch_size // world_size, args.seq_len, output_size)

    counts_mm_row = flops_counter.get_flops()
    flops_counter.stop()

    assert counts_mm_row[0] == counts_mc2_row[0]


def run_fa_tflops(bs, seq_len, dtype, use_fa2):
    from megatron.core.transformer.enums import AttnMaskType
    args = parse_args(None, True)
    args.use_flash_attn = True
    args.micro_batch_size = bs
    args.seq_length = seq_len
    args.use_fusion_attn_v2 = use_fa2

    set_args(args)
    initialize_model_parallel()
    set_random_seed(1234)
    # clear global attn mask set by last test case
    set_attention_mask(None)

    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True)

    flops_counter = get_flops_counter()

    attn = DotProductAttention(
        config=config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type='self')

    b, n, s, d = bs, 4, seq_len, 8

    q = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n, d, dtype=dtype, device='npu', requires_grad=True)
    flops_counter.start()
    # global attn mask will be generated at DotProductAttention forward wrapper
    out = attn(q, k, v, None, None, None)
    counts = flops_counter.get_flops()
    flops_counter.stop()

    return counts[0]


class TestDotProductAttn(DistributedTest):

    @pytest.mark.skip(reason='this UT need updata for new version.')
    def test_mc2_tflops(self):
        run_mc2_tflops(2)

    @pytest.mark.skip(reason='this UT need updata for new version.')
    def test_fa_tflops(self):
        fa1_result = run_fa_tflops(1, 2048, torch.bfloat16, False)

        fa2_result = run_fa_tflops(1, 2048, torch.bfloat16, True)

        assert fa1_result == fa2_result
