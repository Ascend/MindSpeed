import os
from pathlib import Path

import torch.distributed

from mindspeed import megatron_adaptor
from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
import mindspeed

from mindspeed.core.memory.adaptive_memory.adaptive_memory_cache import AdaptiveLayerMemPolicy, AdaptiveModelMemPolicy, PolicyCacheManager
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from unit_tests.common import DistributedTest
from commons import initialize_model_parallel


def get_custom_args():
    args = parse_args(None, True)
    set_args(args)
    args.global_batch_size = 16
    args.micro_batch_size = 1
    args.seq_length = 1024
    args.hidden_size = 4096
    args.tensor_model_parallel_size = 2
    args.context_parallel_size = 2
    args.sequence_parallel = True
    args.expert_model_parallel_size = 2
    args.pipeline_model_parallel_size = 1
    args.world_size = 8
    return args


def get_layer(module_cnt):
    recom = ["name" + str(i) for i in range(module_cnt) if i % 2]
    swap = ["name" + str(i) for i in range(module_cnt) if i % 2 == 0]
    layer = AdaptiveLayerMemPolicy(recompute=recom, swap=swap, memory=1234.0, time=12.0)
    return layer


def clean_local_policy_file():
    if torch.distributed.get_rank() != 0:
        return
    mindspeed_home = os.path.dirname(os.path.dirname(mindspeed.__file__))
    adaptive_home = os.path.join(mindspeed_home, "adaptive_mem")
    Path(adaptive_home).mkdir(exist_ok=True)
    for f in Path(adaptive_home).glob("*"):
        if f.is_file():
            f.unlink()


class TestPolicyCache(DistributedTest):
    world_size = 8

    def test_create_empty_cache(self):
        try:
            old_path = os.environ.pop('LD_LIBRARY_PATH', "")
            new_path = ":".join(["/usr/local/Ascend/driver/lib64", "/home/chenxu/CANN_B080/ascend-toolkit/latest/tools/"])
            os.environ["LD_LIBRARY_PATH"] = new_path
            get_custom_args()
            pc = PolicyCacheManager()
            assert len(pc.oom_policy_cache) == 0
            assert len(pc.normal_policy_cache) == 0
            os.environ["LD_LIBRARY_PATH"] = old_path
        finally:
            clean_local_policy_file()

    def test_persistence_one_policy(self):
        args = get_custom_args()
        initialize_model_parallel(tensor_model_parallel_size=args.tensor_model_parallel_size,
                                  pipeline_model_parallel_size=args.pipeline_model_parallel_size,
                                  context_parallel_size=args.context_parallel_size)

        try:
            layer = get_layer(10)
            model1 = AdaptiveModelMemPolicy(policy_type="normal", polices=[layer])
            model2 = AdaptiveModelMemPolicy(policy_type="oom", polices=[layer])

            old_path = os.environ.pop('LD_LIBRARY_PATH', "")
            new_path = ":".join(["/usr/local/Ascend/driver/lib64", "/home/chenxu/CANN_B080/ascend-toolkit/latest/tools/"])
            os.environ["LD_LIBRARY_PATH"] = new_path
            pc = PolicyCacheManager()
            pc.load_cache_file()
            pc.add_normal_policy_cache(model1)
            pc.add_oom_policy_cache(model2)

            pc2 = PolicyCacheManager()
            assert len(pc2.normal_policy_cache) == 1
            assert pc2.oom_policy_cache[0].policy_type == "oom"
            assert len(pc2.oom_policy_cache) == 1
            os.environ["LD_LIBRARY_PATH"] = old_path
        finally:
            clean_local_policy_file()

    def test_compare_layer_policy(self):
        layer1 = get_layer(8)
        layer2 = get_layer(8)
        layer3 = get_layer(8)
        layer3.swap.sort(reverse=True)
        assert layer1 == layer2
        assert layer1 == layer3

        layer4 = get_layer(8)
        layer4.swap.append("input_norm")
        assert layer1 != layer4

    def test_compare_model_policy(self):
        model1 = AdaptiveModelMemPolicy(policy_type="normal", polices=[get_layer(8), get_layer(9), get_layer(10)])
        model2 = AdaptiveModelMemPolicy(policy_type="normal", polices=[get_layer(8), get_layer(9), get_layer(10)])
        assert model1 == model2

        model3 = AdaptiveModelMemPolicy(policy_type="oom", polices=[get_layer(8), get_layer(8), get_layer(8)])
        assert model1 != model3

        model4 = AdaptiveModelMemPolicy(policy_type="normal", polices=[get_layer(8), get_layer(9), get_layer(8)])
        assert model1 != model4

        model5 = AdaptiveModelMemPolicy(policy_type="normal", polices=[get_layer(10), get_layer(9), get_layer(8)])
        assert model1 == model5
