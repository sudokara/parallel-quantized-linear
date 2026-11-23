#include <torch/library.h>
#include <torch/extension.h>

torch::Tensor add_one_cuda(const torch::Tensor& x);

TORCH_LIBRARY(qlinear, m) {
    m.def("add_one(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(qlinear, CUDA, m) {
    m.impl("add_one", &add_one_cuda);
}