import pytest
import torch
import torch_npu

from ascendspeed.ops import npu_dropout_add_layer_norm, npu_dropout_add_rms_norm
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNpuDropoutAddLayerNorm(DistributedTest):
    world_size = 1

    def supported_op_exec(self, x0,
                          weight,
                          residual,
                          bias,
                          rowscale,
                          layerscale,
                          dropout_p,
                          epsilon,
                          prenorm,
                          residual_in_fp32,
                          is_rms_norm,
                          return_dropout_mask):
        device = "cpu"
        x0 = x0.to(device)
        weight = weight.to(device)
        batch, seq, hidden_size = x0.shape
        if rowscale is not None:
            x0 = x0 * rowscale.to(device).view(batch, seq, 1)
        if layerscale is not None:
            x0 = x0 * layerscale.to(device).view(1, 1, hidden_size)
        if residual is not None:
            x0 = x0 + residual.to(device)
        if is_rms_norm:
            rms_norm = x0 / torch.sqrt(torch.sum(x0 * x0, dim=2, keepdim=True) / hidden_size + epsilon)
            result = rms_norm * weight.view(1, 1, hidden_size)
            return result

    def custom_op_exec(self, x0,
                       weight,
                       residual,
                       bias,
                       rowscale,
                       layerscale,
                       dropout_p,
                       epsilon,
                       prenorm,
                       residual_in_fp32,
                       is_rms_norm,
                       return_dropout_mask):
        if is_rms_norm:
            return npu_dropout_add_rms_norm(x0, residual, weight, bias, dropout_p, epsilon,
                                            rowscale, layerscale, prenorm, residual_in_fp32, return_dropout_mask)
        else:
            return npu_dropout_add_layer_norm(x0, residual, weight, bias, dropout_p, epsilon,
                                              rowscale, layerscale, prenorm, residual_in_fp32, return_dropout_mask)

    def test_npu_dropout_add_layer_norm(self):
        batch, seq, hidden_size = 6, 60, 1024
        x0 = torch.randn((batch, seq, hidden_size)).to(torch.float).npu()
        weight = torch.randn((hidden_size, )).to(torch.float).npu()
        residual, bias, rowscale, layerscale = None, None, None, None
        dropout_p = 0.0
        epsilon = 1e-5
        prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask = False, True, True, False
        output_cpu = self.supported_op_exec(x0, weight, residual, bias, rowscale, layerscale,
                                            dropout_p, epsilon, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask)
        output_custom = self.custom_op_exec(x0, weight, residual, bias, rowscale, layerscale,
                                            dropout_p, epsilon, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask)
        assert torch.allclose(output_cpu, output_custom.cpu().detach().clone(), rtol=0.001, atol=0.001)
