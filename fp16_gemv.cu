#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int warp_size = 32;
constexpr int warps_per_block = 4;

/**
 * @brief warp_sum reduces one FP32 partial sum from each lane.
 * @param[in] sum The calling lane's partial sum.
 * @return The complete warp sum in lane 0; other lanes hold intermediate values.
 * @note All 32 lanes must participate because the shuffle uses a full warp mask.
 */
__device__ __forceinline__ float warp_sum(float sum) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    return sum;
}

/**
 * @brief gemv_kernel computes one FP16 weight row dot product per warp.
 * @param[in] weights The row-major FP16 weight matrix [N, K].
 * @param[in] input The FP16 input vector of length K.
 * @param[out] output The FP16 result vector of length N.
 * @param[in] n The number of output rows.
 * @param[in] k The number of columns per row.
 * @return None.
 * @note Each block covers up to four rows. Each lane visits columns spaced 32 apart.
 * @note Lanes accumulate in FP32. Lane 0 rounds the reduced sum to FP16 and writes it.
 */
__global__ void gemv_kernel(const at::Half* weights, const at::Half* input,
                            at::Half* output, int64_t n, int64_t k) {
    const int lane = threadIdx.x % warp_size;
    const int64_t row = int64_t(blockIdx.x) * warps_per_block
                        + threadIdx.x / warp_size;
    if (row >= n) {
        return;
    }

    const at::Half* row_weights = weights + row * k;
    float sum = 0.0f;
    for (int64_t col = lane; col < k; col += warp_size) {
        sum = fmaf(float(row_weights[col]), float(input[col]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[row] = at::Half(sum);
    }
}

/**
 * @brief gemv validates and launches the FP16 GEMV.
 * @param[in] input The contiguous FP16 input tensor [1, K].
 * @param[in] weights The contiguous FP16 weight tensor [N, K] on the same CUDA device.
 * @return The FP16 output tensor [1, N].
 * @note The launch is asynchronous on the current CUDA stream for the input device.
 */
torch::Tensor gemv(torch::Tensor input, torch::Tensor weights) {
    TORCH_CHECK(input.is_cuda() && weights.is_cuda() &&
                input.device() == weights.device(),
                "input and weights must be on the same CUDA device");
    TORCH_CHECK(input.scalar_type() == at::kHalf &&
                weights.scalar_type() == at::kHalf &&
                input.is_contiguous() && weights.is_contiguous(),
                "input and weights must be contiguous FP16 tensors");
    TORCH_CHECK(input.dim() == 2 && input.size(0) == 1 &&
                weights.dim() == 2 && input.size(1) == weights.size(1) &&
                weights.size(0) > 0 && weights.size(1) > 0,
                "expected input [1, K] and weights [N, K] with positive N and K");

    const c10::cuda::CUDAGuard guard(weights.device());
    const int64_t n = weights.size(0);
    const int64_t k = weights.size(1);
    auto output = at::empty({1, n}, input.options());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    gemv_kernel<<<(n + warps_per_block - 1) / warps_per_block,
                  warps_per_block * warp_size, 0, stream>>>(
        weights.data_ptr<at::Half>(), input.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(), n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

/**
 * @brief dequantize_weight reconstructs one weight from its signed code and scale.
 * @param[in] code The signed INT8 code or unpacked INT4 code.
 * @param[in] scale The FP16 scale for this weight's group.
 * @return The FP16-rounded product of the code and scale.
 * @note The fused and two-step kernels share this rounding point.
 */
__device__ __forceinline__ at::Half dequantize_weight(int8_t code, at::Half scale) {
    return at::Half(float(code) * float(scale));
}

/**
 * @brief read_code loads a signed INT8 code stored as one byte per weight.
 * @param[in] row The start of the INT8 weights to index.
 * @param[in] col The zero-based offset from row, in weights.
 * @return The signed INT8 code at that offset.
 * @note The two-step kernel can index from the matrix base with a flat offset.
 */
__device__ __forceinline__ int8_t read_code(const int8_t* row, int64_t col) {
    return row[col];
}

/**
 * @brief read_code extracts one signed INT4 code from a packed row.
 * @param[in] row The start of the row's packed 32-bit words.
 * @param[in] col The zero-based logical weight column before packing.
 * @return The sign-extended code in the range [-8, 7].
 * @note Each word stores eight weights with the earliest column in its low nibble.
 */
__device__ __forceinline__ int8_t read_code(const int32_t* row, int64_t col) {
    const uint32_t word = static_cast<uint32_t>(row[col / 8]);
    const int nibble = (word >> (4 * (col % 8))) & 0xF;
    return (nibble ^ 0x8) - 0x8;
}

/**
 * @brief dequantize_kernel materializes an FP16 weight matrix for two-step GEMV.
 * @tparam Code int8_t for INT8 bytes or int32_t for packed INT4 words.
 * @param[in] codes The row-major quantized weight storage.
 * @param[in] scales The row-major FP16 group scales.
 * @param[out] weights The contiguous FP16 weight matrix to write.
 * @param[in] count The number of logical weights across all rows.
 * @param[in] k The number of logical weights per row.
 * @param[in] row_stride The number of storage elements per row.
 * @param[in] group_size The number of logical columns per scale, except the last partial group.
 * @param[in] groups_per_row The number of scales per row.
 * @return None.
 * @note Each thread handles one flat weight index. Blocks may span row boundaries.
 */
template <typename Code>
__global__ void dequantize_kernel(const Code* codes, const at::Half* scales,
                                  at::Half* weights, int64_t count, int64_t k,
                                  int64_t row_stride, int64_t group_size,
                                  int64_t groups_per_row) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const int64_t row = index / k;
        const int64_t col = index - row * k;
        const at::Half scale = scales[row * groups_per_row + col / group_size];
        const int8_t code = sizeof(Code) == sizeof(int8_t)
            ? read_code(codes, index)
            : read_code(codes + row * row_stride, col);
        weights[index] = dequantize_weight(code, scale);
    }
}

/**
 * @brief check_quantized_weights validates the layout shared by the quantized launches.
 * @tparam Code int8_t for INT8 bytes or int32_t for packed INT4 words.
 * @param[in] codes The contiguous CUDA tensor [N, K] or [N, ceil(K / 8)].
 * @param[in] scales The contiguous FP16 CUDA tensor [N, ceil(K / group_size)].
 * @param[in] k The logical number of weights per row, excluding packing padding.
 * @param[in] group_size The number of logical columns per scale, which must be positive.
 * @return The number of scales per row.
 * @note The code and scale tensors must share a CUDA device.
 */
template <typename Code>
int64_t check_quantized_weights(const torch::Tensor& codes, const torch::Tensor& scales,
                               int64_t k, int64_t group_size) {
    TORCH_CHECK(codes.is_cuda() && codes.device() == scales.device(),
                "codes and scales must be on the same CUDA device");
    constexpr bool packed = sizeof(Code) == sizeof(int32_t);
    constexpr auto code_dtype = packed ? at::kInt : at::kChar;
    TORCH_CHECK(codes.scalar_type() == code_dtype && scales.scalar_type() == at::kHalf &&
                codes.is_contiguous() && scales.is_contiguous(),
                "expected contiguous codes and FP16 scales with matching dtypes");
    TORCH_CHECK(k > 0 && group_size > 0 && codes.dim() == 2 && codes.size(0) > 0,
                "expected positive N, K, and group size");
    TORCH_CHECK(codes.size(1) == (packed ? 1 + (k - 1) / 8 : k),
                "expected INT8 codes [N, K] or packed INT4 codes [N, ceil(K / 8)]");
    const int64_t groups_per_row = 1 + (k - 1) / group_size;
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == codes.size(0) &&
                scales.size(1) == groups_per_row,
                "expected scales [N, ceil(K / group_size)]");
    return groups_per_row;
}

/**
 * @brief dequantize validates and launches the first step of quantized GEMV.
 * @tparam Code int8_t for INT8 bytes or int32_t for packed INT4 words.
 * @param[in] codes The contiguous quantized weight storage.
 * @param[in] scales The contiguous FP16 group scales.
 * @param[out] workspace The caller's contiguous FP16 tensor [N, K] for reconstructed weights.
 * @param[in] group_size The number of logical columns per scale.
 * @return None.
 * @note The workspace must share the input device and have independent storage.
 * @note The launch is asynchronous on the current CUDA stream.
 */
template <typename Code>
void dequantize(torch::Tensor codes, torch::Tensor scales,
                torch::Tensor workspace, int64_t group_size) {
    TORCH_CHECK(workspace.device() == codes.device() &&
                workspace.scalar_type() == at::kHalf && workspace.is_contiguous() &&
                workspace.dim() == 2,
                "expected contiguous FP16 workspace [N, K] on the same device as codes");
    const int64_t k = workspace.size(1);
    const int64_t groups_per_row = check_quantized_weights<Code>(codes, scales, k, group_size);
    TORCH_CHECK(workspace.size(0) == codes.size(0), "workspace and codes must have the same N");
    TORCH_CHECK(!workspace.is_alias_of(codes) && !workspace.is_alias_of(scales),
                "workspace must have its own storage");

    const c10::cuda::CUDAGuard guard(codes.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    const int64_t count = workspace.numel();
    dequantize_kernel<Code><<<(count + threads - 1) / threads, threads, 0, stream>>>(
        codes.data_ptr<Code>(), scales.data_ptr<at::Half>(),
        workspace.data_ptr<at::Half>(), count, k, codes.size(1), group_size, groups_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/**
 * @brief gemv_fused_kernel reconstructs and consumes quantized weights in one launch.
 * @tparam Code int8_t for INT8 bytes or int32_t for packed INT4 words.
 * @param[in] codes The row-major quantized weight storage.
 * @param[in] scales The row-major FP16 group scales.
 * @param[in] input The FP16 input vector of length K.
 * @param[out] output The FP16 result vector of length N.
 * @param[in] n The number of output rows.
 * @param[in] k The number of logical weights per row.
 * @param[in] row_stride The number of storage elements per row.
 * @param[in] group_size The number of logical columns per scale.
 * @param[in] groups_per_row The number of scales per row.
 * @return None.
 * @note Each warp owns one row. Each lane reuses the scale across its columns in a group.
 * @note Rounds each reconstructed weight to FP16 and preserves the two-step GEMV's accumulation order.
 */
template <typename Code>
__global__ void gemv_fused_kernel(const Code* codes, const at::Half* scales,
                                  const at::Half* input, at::Half* output,
                                  int64_t n, int64_t k, int64_t row_stride,
                                  int64_t group_size, int64_t groups_per_row) {
    const int lane = threadIdx.x % warp_size;
    const int64_t row = int64_t(blockIdx.x) * warps_per_block
                        + threadIdx.x / warp_size;
    if (row >= n) {
        return;
    }

    const Code* row_codes = codes + row * (sizeof(Code) == sizeof(int8_t) ? k : row_stride);
    const at::Half* row_scales = scales + row * groups_per_row;
    float sum = 0.0f;
    // Keep col across groups to preserve the FP16 GEMV's accumulation order.
    int64_t col = lane;
    for (int64_t group = 0; col < k; ++group) {
        const int64_t group_end = group + 1 == groups_per_row
            ? k : (group + 1) * group_size;
        if (col >= group_end) {
            continue;
        }

        const at::Half scale = row_scales[group];
        for (; col < group_end; col += warp_size) {
            const at::Half weight = dequantize_weight(read_code(row_codes, col), scale);
            sum = fmaf(float(weight), float(input[col]), sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[row] = at::Half(sum);
    }
}

/**
 * @brief gemv_fused validates and launches the fused quantized GEMV.
 * @tparam Code int8_t for INT8 bytes or int32_t for packed INT4 words.
 * @param[in] input The contiguous FP16 input tensor [1, K].
 * @param[in] codes The contiguous quantized weight storage.
 * @param[in] scales The contiguous FP16 group scales.
 * @param[in] group_size The number of logical columns per scale.
 * @return The FP16 output tensor [1, N].
 * @note The input, codes, and scales share a CUDA device. The launch uses its current stream.
 */
template <typename Code>
torch::Tensor gemv_fused(torch::Tensor input, torch::Tensor codes,
                        torch::Tensor scales, int64_t group_size) {
    TORCH_CHECK(input.device() == codes.device() && input.scalar_type() == at::kHalf &&
                input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
                "expected contiguous FP16 input [1, K] on the same CUDA device as codes");
    const int64_t k = input.size(1);
    const int64_t groups_per_row = check_quantized_weights<Code>(codes, scales, k, group_size);
    const int64_t n = codes.size(0);

    const c10::cuda::CUDAGuard guard(codes.device());
    auto output = at::empty({1, n}, input.options());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    gemv_fused_kernel<Code><<<(n + warps_per_block - 1) / warps_per_block,
                              warps_per_block * warp_size, 0, stream>>>(
        codes.data_ptr<Code>(), scales.data_ptr<at::Half>(),
        input.data_ptr<at::Half>(), output.data_ptr<at::Half>(), n, k,
        codes.size(1), group_size, groups_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
}  // namespace

/**
 * @brief Registers the FP16, INT8, and packed INT4 entry points with Python.
 * @param[out] module The Python module populated with the CUDA bindings.
 * @return None.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gemv", &gemv);
    module.def("dequantize_int8", &dequantize<int8_t>);
    module.def("gemv_int8_fused", &gemv_fused<int8_t>);
    module.def("dequantize_int4", &dequantize<int32_t>);
    module.def("gemv_int4_fused", &gemv_fused<int32_t>);
}
