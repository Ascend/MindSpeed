#include <torch/extension.h>

void reuse_data_ptr(at::Tensor& des, at::Tensor& src, int64_t offset)
{
    TORCH_CHECK(
        offset >= 0,
        "Expect offset equal or greater than zero, got: ", offset);

    TORCH_CHECK(
        (offset + des.numel()) * des.element_size() <=
        src.numel() * src.element_size(),
        "Offsets overflow, got: ",
        "offset ", offset * des.element_size(),
        ", des storage size ", des.numel() * des.element_size(),
        ", src storage size ", src.numel()* src.element_size());

    char* data_ptr = static_cast<char*>(src.storage().data_ptr().get()) + offset * des.element_size();
    at::DataPtr aim_data_ptr = at::DataPtr(data_ptr, des.storage().device());
    des.storage().set_data_ptr(std::move(aim_data_ptr));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reuse_data_ptr", &reuse_data_ptr, "reuse tensor data ptr");
}