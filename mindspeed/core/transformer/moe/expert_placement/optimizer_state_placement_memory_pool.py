import torch


class OptimizerStatePlacementMemoryPool:
    def __init__(self):
        self.pool = {}  
        self.total_allocated = 0  

    def get(self, shape, dtype, init_method='zero'):
        key = (tuple(shape), dtype)
        
        if key in self.pool and self.pool[key]:
            tensor = self.pool[key].pop()
            if init_method == 'zero':
                tensor.zero_()
            elif init_method == 'normal':
                tensor.normal_()
        else:
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            self.total_allocated += 1
            if init_method == 'zero':
                tensor.zero_()
            elif init_method == 'normal':
                tensor.normal_()
            self.put(tensor)
        return tensor

    def put(self, tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self.pool:
            self.pool[key] = []
        self.pool[key].append(tensor.detach())

    def clear(self, empty_cache=True):
        self.pool.clear()
        self.total_allocated = 0
        if not empty_cache:
            return
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.synchronize()
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            import warnings
            warnings.warn(f"Failed to clear device cache: {str(e)}", RuntimeWarning)
