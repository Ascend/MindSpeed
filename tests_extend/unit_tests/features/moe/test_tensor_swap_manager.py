import pytest
import unittest
from unittest.mock import patch, MagicMock

import torch
import torch_npu

from mindspeed import megatron_adaptor
from mindspeed.core.pipeline_parallel.fb_overlap.modules.utils import (
    TensorSwapManager,
    make_wait_swap_in_hook,
    make_async_swap_in_hook
    )


class TestTensorSwapManager:

    @classmethod
    def setup_class(cls):
        # Ensure we have a device to test with
        if not torch_npu.npu.is_available():
            pytest.skip("NPU device not available")

    def test_initialization(self):
        """Test basic initialization of TensorSwapManager."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        assert manager.npu_tensor is test_tensor
        assert manager.cpu_tensor is None
        assert manager.swap_out_event is None
        assert manager.swap_in_event is None
        assert not manager.under_swap_in

    def test_swap_group_initialization(self):
        """Test initialization with swap groups."""
        test_tensor = torch.randn(5, 5, device='npu')
        group_name = "test_group"
        
        # First manager should create the group
        manager1 = TensorSwapManager(test_tensor, group_name)
        assert group_name in TensorSwapManager._ALL_SWAP_OUT_QUEUES
        assert len(TensorSwapManager._ALL_SWAP_OUT_QUEUES[group_name]) == 0
        
        # Second manager should use existing group
        manager2 = TensorSwapManager(test_tensor, group_name)
        assert len(TensorSwapManager._ALL_SWAP_OUT_QUEUES[group_name]) == 0

    def test_async_swap_out(self):
        """Test asynchronous swap-out functionality."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        # Perform async swap-out
        manager.async_swap_out()
        
        # Verify swap-out was initiated
        assert manager.cpu_tensor is not None
        assert manager.swap_out_event is not None
        assert manager.cpu_tensor.device.type == 'cpu'
        assert manager.cpu_tensor.is_pinned()

    def test_async_swap_out_wait_conditions(self):
        """Test swap-out with wait conditions."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        # Create dummy event and stream
        dummy_event = torch.npu.Event()
        dummy_event.record()
        dummy_stream = torch.npu.Stream()
        
        # Should complete without errors
        manager.async_swap_out(wait_event=dummy_event, wait_stream=dummy_stream)

    def test_wait_swap_out(self):
        """Test waiting for swap-out completion."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        manager.async_swap_out()
        manager.wait_swap_out()
        
        # Verify NPU storage was released
        assert manager.npu_tensor.untyped_storage().size() == 0

    def test_async_swap_in(self):
        """Test asynchronous swap-in functionality."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        # First swap out
        manager.async_swap_out()
        manager.wait_swap_out()
        
        # Then swap back in
        manager.async_swap_in()
        
        # Verify swap-in was initiated
        assert manager.swap_in_event is not None
        assert manager.under_swap_in

    def test_wait_swap_in(self):
        """Test waiting for swap-in completion."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        # Full swap cycle
        manager.async_swap_out()
        manager.wait_swap_out()
        manager.async_swap_in()
        manager.wait_swap_in()
        
        # Verify state after swap-in
        assert manager.cpu_tensor is None
        assert not manager.under_swap_in
        assert manager.npu_tensor.untyped_storage().size() > 0

    def test_wait_all_swap_out(self):
        """Test waiting for all swaps in a group."""
        test_tensor = torch.randn(5, 5, device='npu')
        group_name = "group_test"
        
        # Create two managers in the same group
        manager1 = TensorSwapManager(test_tensor, group_name)
        manager2 = TensorSwapManager(test_tensor.clone(), group_name)
        
        # Initiate swaps
        manager1.async_swap_out()
        manager2.async_swap_out()
        
        # Wait for all in group
        TensorSwapManager.wait_all_swap_out(group_name)
        
        # Verify both were processed
        assert manager1.npu_tensor.untyped_storage().size() == 0
        assert manager2.npu_tensor.untyped_storage().size() == 0
        assert len(TensorSwapManager._ALL_SWAP_OUT_QUEUES[group_name]) == 0

    def test_hook_functions(self):
        """Test the hook creation functions."""
        test_tensor = torch.randn(5, 5, device='npu')
        manager = TensorSwapManager(test_tensor)
        
        # Test wait swap-in hook
        wait_hook = make_wait_swap_in_hook(manager)
        manager.async_swap_out()
        manager.wait_swap_out()
        manager.async_swap_in()
        wait_hook()  # Should wait for swap-in
        
        # Test async swap-in hook
        managers = [TensorSwapManager(test_tensor.clone()) for _ in range(3)]
        async_hook = make_async_swap_in_hook(managers)
        
        # First swap out all
        for m in managers:
            m.async_swap_out()
            m.wait_swap_out()
        
        # Trigger swap-in
        async_hook()
        
        # Verify all initiated swap-in
        for m in managers:
            assert m.swap_in_event is not None
