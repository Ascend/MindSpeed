import torch
import torch_npu
from unit_tests.common import DistributedTest


class AdaptiveRecomputePolicy:
    # swap_attention
    def __init__(self, args):
        self.interval = 0
        self.threshold_prefetch = 0
        self.num_prefetch = 0
        self.num_layers = 0
        self.args = args
        self.pp_rank = 0
        self.is_last_stage = False

    def granular_module_allocation(self, vpp_size, recompute_num_layers, cur_pp_noop_layers):
        swap_list = []
        recompute_list = []
        args = self.args
        cur_pp_rank = self.pp_rank
        pp_size = args.pipeline_model_parallel_size or 1
        vpp_layer = args.num_layers_per_virtual_pipeline_stage
        if self.num_prefetch <= vpp_size:
            swap_list = [['0'] if i < self.num_prefetch else [''] for i in range(vpp_size)]
        else:
            for chunk in range(vpp_size):
                chunk_swap_layer = ['0']
                for layer_id in range(vpp_size, self.num_prefetch):
                    if layer_id % vpp_size == chunk:
                        chunk_swap_layer.append(f'{layer_id // vpp_size}')
                swap_list.append(chunk_swap_layer)

        if recompute_num_layers <= vpp_size:
            recompute_list = [['0'] if i < recompute_num_layers else [''] for i in range(vpp_size)]
            if self.is_last_stage and args.reduce_recompute_for_last_chunk:
                recompute_list[-1] = ['']
        else:
            for chunk in range(vpp_size):
                chunk_recompute_layer = ['0']
                for layer_id in range(vpp_size, recompute_num_layers):
                    if layer_id % vpp_size == chunk:
                        chunk_recompute_layer.append(f'{layer_id // vpp_size}')
                recompute_list.append(chunk_recompute_layer)
            if self.is_last_stage and args.reduce_recompute_for_last_chunk:
                if recompute_list[-1][-1] == str(args.num_layers_per_virtual_pipeline_stage - 1):
                    recompute_list[-1].pop()
                    if len(recompute_list[-1]) == 0:
                        recompute_list[-1].append('')
        for vpp in range(vpp_size):
            vpp_layers = swap_list[vpp]
            for i in range(len(vpp_layers)):
                layer_id = vpp * vpp_layer * pp_size + i + vpp_layer * cur_pp_rank
                if layer_id in cur_pp_noop_layers:
                    swap_list[vpp][i] = ''
                    if len(recompute_list[vpp]) >= i + 1:
                        recompute_list[vpp][i] = ''

        prefetch_list = swap_list
        interval = 0
        prefetch_recompute_group = [swap_list, prefetch_list, recompute_list]
        return [prefetch_recompute_group, interval, self.num_prefetch, cur_pp_noop_layers]

    def get_cur_stage_noop_layers(self, noop_layers):
        all_args = self.args
        cur_pp_noop_layers = []
        cur_pp_rank = self.pp_rank
        pp_size = all_args.pipeline_model_parallel_size or 1
        layers_per_pp = all_args.num_layers // pp_size
        for i in noop_layers:
            pp_id = i // layers_per_pp
            if pp_id == cur_pp_rank:
                cur_pp_noop_layers.append(i)
        return cur_pp_noop_layers

    def solve_prefetch_policy(self):
        all_args = self.args
        noop_layers = list(all_args.noop_layers) if isinstance(all_args.noop_layers, set) else []
        cur_pp_noop_layers = self.get_cur_stage_noop_layers(noop_layers)
        recompute_num_layers = all_args.recompute_num_layers or 0
        pp_size = all_args.pipeline_model_parallel_size or 1
        vpp_size = all_args.virtual_pipeline_model_parallel_size or 1
        per_pp_layers = all_args.num_layers // pp_size
        if not all_args.enable_recompute_layers_per_pp_rank:
            recompute_num_layers *= vpp_size
        if all_args.recompute_method == 'block':
            self.num_prefetch = recompute_num_layers
        elif all_args.recompute_method == 'uniform':
            recompute_num_layers = per_pp_layers
            self.num_prefetch = recompute_num_layers
        else:
            self.num_prefetch = per_pp_layers
        self.interval = 0
        if vpp_size > 1:
            return self.granular_module_allocation(vpp_size, recompute_num_layers, cur_pp_noop_layers)
        else:
            swap_list, recompute_list = [], []
            for i in range(self.num_prefetch):
                if i + self.pp_rank * per_pp_layers not in cur_pp_noop_layers:
                    swap_list.append(str(i))
                else:
                    swap_list.append('')
            for i in range(recompute_num_layers):
                if i + self.pp_rank * per_pp_layers not in cur_pp_noop_layers:
                    recompute_list.append(str(i))
                else:
                    recompute_list.append('')
            prefetch_list = swap_list
            prefetch_recompute_group = [[swap_list], [prefetch_list], [recompute_list]]
            return [prefetch_recompute_group, 0, len(prefetch_list), cur_pp_noop_layers]


class Config:
    def __init__(self):
        self.noop_layers = None
        self.pipeline_model_parallel_size = 1
        self.num_layers = 8
        self.recompute_num_layers = 4
        self.virtual_pipeline_model_parallel_size = 1
        self.enable_recompute_layers_per_pp_rank = False
        self.recompute_method = None
        self.num_layers_per_virtual_pipeline_stage = 1


class TestSwapAttention(DistributedTest):
    world_size = 1
    reuse_dist_env = False

    @staticmethod
    def check_result(arp, check_swap, check_prefetch, check_recompute, check_noop):
        prefetch_recompute_group, interval, num_prefetch, swap_noop_layers = arp.solve_prefetch_policy()
        swap_list, prefetch_list, recompute_list = prefetch_recompute_group
        assert swap_list == check_swap
        assert prefetch_list == check_prefetch
        assert recompute_list == check_recompute
        assert swap_noop_layers == check_noop

    def test_storage_copy_interface(self):
        tensor1 = torch.randn([2048, 1, 4096], dtype=torch.bfloat16, device='npu:0')
        tensor_cpu = torch.empty(tensor1.shape, dtype=tensor1.dtype, pin_memory=True, device='cpu')
        tensor_storage_size = tensor1.untyped_storage().size()

        stream = torch_npu.npu.Stream(device=torch.npu.current_device)
        with torch_npu.npu.stream(stream):
            stream.wait_stream(torch.npu.current_stream())
            tensor_cpu.untyped_storage().copy_(tensor1.untyped_storage(), non_blocking=True)

        torch.npu.current_stream().wait_stream(stream)
        assert torch.allclose(tensor1.cpu().float().sum(), tensor_cpu.float().sum())

        tensor1.untyped_storage().resize_(0)

        with torch_npu.npu.stream(stream):
            torch.npu.current_stream().wait_stream(stream)
            tensor1.untyped_storage().resize_(tensor_storage_size)
            tensor1.untyped_storage().copy_(tensor_cpu.untyped_storage(), non_blocking=True)

        torch.npu.current_stream().wait_stream(stream)
        assert torch.allclose(tensor1.cpu().float().sum(), tensor_cpu.float().sum())

    def test_swap_attention_cal_prefetch_list(self):
        args = Config()
        arp = AdaptiveRecomputePolicy(args)
        self.check_result(arp,
                          [['0', '1', '2', '3', '4', '5', '6', '7']],
                          [['0', '1', '2', '3', '4', '5', '6', '7']],
                          [['0', '1', '2', '3']],
                          [])

    def test_swap_attention_cal_prefetch_list_enable_pp(self):
        args = Config()
        args.pipeline_model_parallel_size = 2
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp,
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [])

    def test_swap_attention_cal_prefetch_list_enable_pp_enable_noop_layers(self):
        args = Config()
        args.pipeline_model_parallel_size = 2
        args.noop_layers = {0, 7}
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp,
                          [['', '1', '2', '3']],
                          [['', '1', '2', '3']],
                          [['', '1', '2', '3']],
                          [0])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0', '1', '2', '']],
                          [['0', '1', '2', '']],
                          [['0', '1', '2', '']],
                          [7])

    #
    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_noop_layers(self):
        args = Config()
        args.pipeline_model_parallel_size = 2
        args.virtual_pipeline_model_parallel_size = 4
        args.noop_layers = {0, 7}
        args.enable_recompute_layers_per_pp_rank = True
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp,
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [0])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [7])

        args.enable_recompute_layers_per_pp_rank = False
        args.recompute_num_layers = 1
        arp.pp_rank = 0
        self.check_result(arp,
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [0])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [7])

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_multiple_noop_layers(self):
        args = Config()
        args.pipeline_model_parallel_size = 2
        args.virtual_pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 2
        args.noop_layers = {0, 1, 6, 7}
        args.enable_recompute_layers_per_pp_rank = True
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp,
                          [['', ''], ['0', '1']],
                          [['', ''], ['0', '1']],
                          [['', ''], ['0', '1']],
                          [0, 1])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0', '1'], ['', '']],
                          [['0', '1'], ['', '']],
                          [['0', '1'], ['', '']],
                          [6, 7])
