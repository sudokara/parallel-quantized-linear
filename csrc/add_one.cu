#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void add_one_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = x[i] + 1.0f;
    }
}

torch::Tensor add_one_cuda(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be on the GPU");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    add_one_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return y;
}