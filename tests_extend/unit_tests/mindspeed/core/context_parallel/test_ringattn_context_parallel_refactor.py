import pytest
import torch
import torch_npu
from mindspeed.core.context_parallel.ring_context_parallel import (
    AttentionStrategyFactory,
    CausalRegularAttentionStrategy,
    CausalEodAttentionStrategy,
    GeneralAttentionStrategy,
    KVCacheManager
)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
class TestRefactoredRingAttnCP:

    def test_attention_strategy_factory(self):
        # Test strategy creation
        causal_reg_strategy = AttentionStrategyFactory.get_strategy(True, False)
        assert isinstance(causal_reg_strategy, CausalRegularAttentionStrategy)

        causal_eod_strategy = AttentionStrategyFactory.get_strategy(True, True)
        assert isinstance(causal_eod_strategy, CausalEodAttentionStrategy)

        general_strategy = AttentionStrategyFactory.get_strategy(False, False)
        assert isinstance(general_strategy, GeneralAttentionStrategy)

    def test_kv_cache_manager_full_policy(self):
        # Test full cache policy
        manager = KVCacheManager("full")
        cur_kv = (torch.randn(2, 10, 20), torch.randn(2, 10, 20))

        # Test update and get cache
        manager.update_cache(cur_kv)
        k_stack, v_stack = manager.get_cache(cur_kv)
        assert len(k_stack) == 1
        assert len(v_stack) == 1

    def test_kv_cache_manager_half_policy(self):
        # Test half cache policy
        manager_half = KVCacheManager("half")
        cur_kv = (torch.randn(2, 10, 20), torch.randn(2, 10, 20))

        manager_half.update_cache(cur_kv)
        assert manager_half.v_cache_list is None or len(manager_half.v_cache_list) == 0