import torch


class SwappableTensor(torch.Tensor):

    @classmethod
    def __new__(cls, tensor, *args, **kwargs):
        # construct a fake tensor to unique tensors
        data = torch.Tensor([id(tensor)])
        return torch.Tensor._make_subclass(cls, data, False)

    def __init__(self, tensor):
        self.id_key = None
        self.inner_tensor = None
        self.inner_tensor_bro_keys = []
        self.inner_tensor_cpu_data = None
        self.storage_data_ptr = None
        self.inner_tensor_data_ptr = None
        self.inner_tensor_origin_storage_size = 0
        self.inner_tensor_origin_storage_ele_size = 0
        self.is_allowed_swap = False
        self._device = None
        self._location = None
        self.is_slice_tensor = tensor.storage().size() != tensor.numel()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def set_tensor(self, id_key, tensor):
        self.id_key = id_key
        self.inner_tensor = tensor
        self.inner_tensor_data_ptr = tensor.data_ptr()
        self.storage_data_ptr = tensor.storage().data_ptr()
        self.inner_tensor_origin_storage_size = tensor.storage().size()
        self.inner_tensor_origin_storage_ele_size = tensor.storage().element_size()
        self._location = "device"
        self._device = tensor.device

    def get_tensor(self):
        return self.inner_tensor

    def set_tensor_location(self, location):
        self._location = location

    def trans_to_cpu(self):
        """
        translate tensor to host memory
        :return:
        """
        with torch.no_grad():
            self.inner_tensor_cpu_data = self.inner_tensor.cpu()
            self.inner_tensor.storage().resize_(0)
            self._location = "cpu"

    def slice_tensor_trans_to_cpu(self):
        """
        translate slice tensor to host memory        
        """
        with torch.no_grad():
            self.inner_tensor_cpu_data = self.inner_tensor.cpu()
            self._location = "cpu"        

    def trans_to_device(self, resize_flag):
        """
        translate tensor to device
        :return:
        """
        with torch.no_grad():
            if resize_flag:
                self.inner_tensor.storage().resize_(self.inner_tensor_origin_storage_size)
            if self.is_slice_tensor:
                self.inner_tensor.copy_(self.inner_tensor_cpu_data)
            else:
                self.inner_tensor.storage().copy_(self.inner_tensor_cpu_data.storage())
            self._location = "device"

    def get_location(self):
        return self._location
    
    def get_slice_tensor(self):
        return self.is_slice_tensor
    
    def get_tensor_origin_storage(self):
        return self.inner_tensor_origin_storage_size * self.inner_tensor_origin_storage_ele_size
    
