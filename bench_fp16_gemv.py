"""Check and time batch-one FP16, W8A16, and W4A16 linear layers on CUDA.

Run on the RTX 2080 with a matching CUDA toolkit and PyTorch build:
    TORCH_CUDA_ARCH_LIST=7.5 python3 bench_fp16_gemv.py
Add --pytorch-only to run the baseline without compiling the extension.
"""

import argparse
from pathlib import Path
import random

import torch
from torch.nn import functional as F


SEED = 42
WARMUP = 20
RUNS = 100
QUANT_GROUP_SIZE = 128  # Set to None for one scale per output row.

# Each shape is (output features, input features).
LLAMA_SHAPES = [
    ("1B Q / O", 2048, 2048),
    ("1B K / V", 512, 2048),
    ("1B gate / up", 8192, 2048),
    ("1B down", 2048, 8192),
    ("1B stacked QKV", 3072, 2048),
    ("1B stacked gate/up", 16384, 2048),
    ("1B lm_head", 128256, 2048),
    ("3B Q / O", 3072, 3072),
    ("3B K / V", 1024, 3072),
    ("3B gate / up", 8192, 3072),
    ("3B down", 3072, 8192),
    ("3B stacked QKV", 5120, 3072),
    ("3B stacked gate/up", 16384, 3072),
    ("3B lm_head", 128256, 3072),
]


## @brief mean_time_us measures a callable with CUDA events on the current stream.
# @param[in] run A zero-argument callable that enqueues the work to time.
# @return The mean elapsed time in microseconds per call.
# @note Discards WARMUP calls and synchronizes before timing RUNS calls.
def mean_time_us(run):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        run()
    torch.cuda.current_stream().synchronize()

    start.record()
    for _ in range(RUNS):
        run()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / RUNS


## @brief quantize_weights computes signed codes and symmetric FP16 scales.
# @param[in] weights The contiguous FP16 weight matrix [N, K].
# @param[in] group_size The number of consecutive columns sharing a scale.
# @param[in] max_code The positive code limit, 127 for INT8 or 7 for INT4.
# @return The unpacked INT8 codes [N, K] and FP16 scales [N, ceil(K / group_size)].
# @note Uses the stored FP16 scale for rounding. All-zero groups use scale 1.
def quantize_weights(weights, group_size, max_code):
    n, k = weights.shape
    pad = (-k) % group_size
    groups = (F.pad(weights, (0, pad)) if pad else weights).view(n, -1, group_size)
    amax = groups.abs().amax(dim=-1).float()
    # Clamp nonzero groups to the smallest positive FP16 scale.
    scales = torch.where(amax == 0, 1.0, (amax / max_code).clamp_min(2 ** -24)).half()
    codes = groups.float().div_(scales.unsqueeze(-1)).round_().clamp_(-max_code, max_code)
    return codes.to(torch.int8).view(n, -1)[:, :k].contiguous(), scales


## @brief pack_int4 packs signed codes into 32-bit words.
# @param[in] codes The unpacked signed INT4 codes in an INT8 tensor [N, K].
# @return The int32 tensor [N, ceil(K / 8)] containing eight codes per word.
# @note The first column occupies the low nibble. Unused nibbles are padded with zeros.
def pack_int4(codes):
    n, k = codes.shape
    pad = (-k) % 8
    padded = F.pad(codes, (0, pad)) if pad else codes
    nibbles = padded.view(n, -1, 8).int().bitwise_and_(0xF)
    shifts = 4 * torch.arange(8, device=codes.device, dtype=torch.int32)
    return nibbles.bitwise_left_shift_(shifts).sum(dim=-1, dtype=torch.int32)


## @brief check_output verifies an FP16 result against its FP32 reference.
# @param[in] label The case and backend name to print on failure.
# @param[in] result The FP16 output tensor.
# @param[in] reference The FP32 reference tensor with the same shape.
# @return None.
# @note Compares in FP32 with relative and absolute tolerances.
def check_output(label, result, reference):
    assert result.dtype == torch.float16, f"{label}: output is not FP16"
    torch.testing.assert_close(
        result.float(), reference, rtol=1e-3, atol=3e-4,
        msg=lambda msg: f"{label}: {msg}",
    )


## @brief check_quantized checks and optionally times both paths for one weight format.
# @param[in] label The shape and quantization settings to print on failure.
# @param[in] x The contiguous FP16 activation tensor [1, K].
# @param[in] weights The source FP16 weight matrix [N, K] to quantize.
# @param[in] extension The loaded CUDA extension.
# @param[in] bits The weight width, 8 or 4.
# @param[in] group_size The columns per scale, or None for one scale per row.
# @param[in] time_it Whether to collect timings after checking correctness.
# @return Mean microseconds for two-step, fused, dequantization, and GEMV, or an empty list.
# @note Checks exact dequantization, agreement with an FP32 reference, and bitwise output parity.
# @note Quantization, packing, and workspace allocation happen outside timing.
def check_quantized(label, x, weights, extension, bits, group_size, time_it):
    k = weights.shape[1]
    group_size = k if group_size is None else group_size
    codes, scales = quantize_weights(weights, group_size, (1 << (bits - 1)) - 1)
    if bits == 4:
        stored_codes = pack_int4(codes)
        launch_dequantize = extension.dequantize_int4
        launch_fused = extension.gemv_int4_fused
    else:
        stored_codes = codes
        launch_dequantize = extension.dequantize_int8
        launch_fused = extension.gemv_int8_fused
    workspace = torch.empty_like(weights)

    ## @brief dequantize reconstructs the captured weights in the reusable FP16 workspace.
    # @return None.
    def dequantize():
        launch_dequantize(stored_codes, scales, workspace, group_size)

    ## @brief gemv multiplies the captured activation by the reconstructed weights.
    # @return The FP16 output tensor [1, N].
    def gemv():
        return extension.gemv(x, workspace)

    ## @brief two_step enqueues dequantization followed by the FP16 GEMV.
    # @return The FP16 output tensor [1, N].
    def two_step():
        dequantize()
        return gemv()

    ## @brief fused enqueues reconstruction and GEMV in one launch.
    # @return The FP16 output tensor [1, N].
    def fused():
        return launch_fused(x, stored_codes, scales, group_size)

    result = two_step()
    groups = torch.arange(k, device=weights.device) // group_size
    expected_weights = codes.half().mul_(scales[:, groups])
    assert torch.equal(workspace, expected_weights), f"{label}: dequantization mismatch"
    reference = torch.mv(expected_weights.float(), x[0].float()).unsqueeze(0)
    check_output(f"{label}, two-step", result, reference)
    fused_result = fused()
    check_output(f"{label}, fused", fused_result, reference)
    assert torch.equal(fused_result.view(torch.int16), result.view(torch.int16)), (
        f"{label}: fused and two-step outputs differ"
    )
    del expected_weights

    if time_it:
        return [mean_time_us(run) for run in (two_step, fused, dequantize, gemv)]
    return []


## @brief check_case generates one batch-one problem and runs the available backends.
# @param[in] label The shape name used in diagnostics and the timing table.
# @param[in] n The number of output features.
# @param[in] k The number of input features.
# @param[in] extension The loaded CUDA extension, or None for PyTorch only.
# @param[in] time_it Whether to print one row of timings.
# @return None.
# @note Correctness-only cases exercise several scale group sizes. Timed cases use QUANT_GROUP_SIZE.
def check_case(label, n, k, extension, time_it=False):
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    weights = torch.randn(n, k, device="cuda", dtype=torch.float16)
    weights.mul_(k ** -0.5)
    if not time_it and n > 1:
        weights[0].zero_()
    reference = torch.mv(weights.float(), x[0].float()).unsqueeze(0)
    backends = {"PyTorch": F.linear}
    if extension is not None:
        backends["Custom CUDA"] = extension.gemv

    for name, gemv in backends.items():
        check_output(f"{label} ({n}, {k}), {name}", gemv(x, weights), reference)

    timings = []
    if time_it:
        timings = [mean_time_us(lambda: gemv(x, weights)) for gemv in backends.values()]
    if extension is not None:
        group_sizes = (QUANT_GROUP_SIZE,) if time_it else (None, 32, 64, 128)
        for bits in (8, 4):
            for group_size in group_sizes:
                timings += check_quantized(
                    f"{label} ({n}, {k}), W{bits}A16 group={group_size}",
                    x, weights, extension, bits, group_size, time_it,
                )
    if time_it:
        print(f"{label:20} {n:>7} {k:>6}" + "".join(f"{us:>14.2f}" for us in timings))


## @brief main runs the edge and random checks, then benchmarks the Llama shapes.
# @return None.
# @note Uses a dedicated CUDA stream and loads the extension unless --pytorch-only is set.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.exit(1, "A CUDA GPU and CUDA-enabled PyTorch are required.\n")

    extension = None
    columns = ["PyTorch"]
    if not args.pytorch_only:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("fp16_gemv.cu")
        extension = load("fp16_gemv", [str(source)])
        columns.append("Custom CUDA")
        for bits in (8, 4):
            columns += [f"W{bits}A16 2-step", f"W{bits}A16 fused",
                        f"W{bits} dequant", f"W{bits} GEMV"]

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    edge_shapes = [(1, 1), (3, 7), (4, 8), (5, 9), (3, 31), (4, 32), (5, 33)]
    random_shapes = [(rng.randint(1, 512), rng.randint(1, 512)) for _ in range(12)]

    print(f"{torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        for n, k in edge_shapes + random_shapes:
            check_case("edge/random", n, k, extension)
        print(f"Passed {len(edge_shapes) + len(random_shapes)} edge and random cases.")
        if extension is not None:
            group_label = "per row" if QUANT_GROUP_SIZE is None else f"group size {QUANT_GROUP_SIZE}"
            print(f"W8A16 and W4A16 benchmarks use {group_label}.")
        print(f"Mean latency in us ({WARMUP} discarded warmups, {RUNS} timed calls per backend)")
        print(f"{'Shape':20} {'N':>7} {'K':>6}" + "".join(f"{name:>14}" for name in columns))
        for label, n, k in LLAMA_SHAPES:
            check_case(label, n, k, extension, time_it=True)
        print(f"Passed correctness checks for all {len(LLAMA_SHAPES)} Llama cases.")


if __name__ == "__main__":
    main()
