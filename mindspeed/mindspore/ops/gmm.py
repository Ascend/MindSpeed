# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from mindspore import ops


class _GmmProxy:
    def npu_gmm(self, *args, **kwargs):
        return ops.function.math_func.gmm(*args, **kwargs)
    
    def npu_gmm_backward(self, *args, **kwargs):
        return ops.function.math_func.gmm_backward(*args, **kwargs)
    
    def npu_gmm_backward_fusion(self, grad_outputs, weight, group_list, group_list_type):
        return ops.function.math_func.gmm_backward_fusion(grad_outputs, weight, group_list, group_list_type)


class _GmmProxy2:
    def npu_gmm(self, *args, **kwargs):
        return ops.function.math_func.gmm(*args, **kwargs)
    
    def npu_gmm_backward(self, *args, **kwargs):
        return ops.function.math_func.gmm_backward(*args, **kwargs)
    
    def npu_gmm_backward_fusion(self, grad_outputs, weight, group_list, group_list_type):
        return ops.function.math_func.gmm_backward_fusion(grad_outputs, weight, group_list, group_list_type)
    

_GMM_PROXY = _GmmProxy()
_GMM_PROXY2 = _GmmProxy2()


def _GMM_patched_load(*_args, **_kwargs):
    return _GMM_PROXY


def _GMM_patched_load2(*_args, **_kwargs):
    return _GMM_PROXY2