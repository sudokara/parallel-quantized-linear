"""Check and time vLLM's Marlin and Exllama kernels on the same weights as bench_fp16_gemv.py"""

import torch
import vllm._custom_ops as ops
from vllm.scalar_type import scalar_types

import re
from torch.nn import functional as F

from bench_fp16_gemv import (LLAMA_SHAPES, QUANT_GROUP_SIZE, RUNS, SEED, WARMUP,
                             mean_time_us, quantize_weights)

def pack(values, bits, dim, offset=0):
    """Packs unsigned codes along dim, 32 // bits per int32, first value in the low bits."""
    # 8 for INT4, 4 for INT8
    per_word = 32 // bits

    # Put the axis to pack last
    values = values.movedim(dim, -1)

    # Group by words
    values = values.reshape(*values.shape[:-1], -1, per_word)

    # All dimensions except reduced packing dimension
    packed = torch.zeros(values.shape[:-1], dtype=torch.int32, device=values.device)

    for i in range(per_word):
        # Pack into word
        packed |= (values[..., i].int() + offset) << (bits * i)

    # Move packed dimension back to where it was originally
    return packed.movedim(-1, dim).contiguous()

def to_gptq(codes, scales, bits):
    """Converts our signed codes [N, K] and scales [N, G] to GPTQ's layout."""
    # Centered zeros
    zero = 1 << (bits - 1) # 8 for INT4, 128 for INT8

    # Pack weights
    qweight = pack(codes.t(), bits, dim=0, offset=zero)   # [K * bits / 32, N]

    # Transpose scales to match weights
    gptq_scales = scales.t().contiguous() # [G, N]

    # Centered zeros [G, N]
    zeros = torch.full(gptq_scales.shape, zero, dtype=torch.int32, device=codes.device)

    # Pack zeros
    qzeros = pack(zeros, bits, dim=1) # [G, N * bits / 32]

    # Returned pack weights, zeros, and scales
    return qweight, qzeros, gptq_scales

def quantize_rows(weights, group_size, max_code, rows=8192):
    """Runs quantize_weights on blocks of rows to keep temporaries small. Rows are independent."""
    parts = [quantize_weights(w, group_size, max_code) for w in weights.split(rows)]

    return torch.cat([c for c, _ in parts]), torch.cat([s for _, s in parts])

def marlin_permute_scales(s, size_k, size_n, group_size):
    """Reorders scales into Marlin's layout. Copied from vLLM's marlin_utils.py (Apache-2.0)"""
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scale_perm_single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    
    if group_size < size_k and group_size != -1:
        s = s.reshape(-1, len(scale_perm))[:, scale_perm]
    else:
        s = s.reshape(-1, len(scale_perm_single))[:, scale_perm_single]

    return s.reshape(-1, size_n).contiguous()

def make_marlin(qweight, gptq_scales, bits, n, k, group_size):
    """Repacks GPTQ weights for Marlin and returns a function computing x @ W.T."""
    device = qweight.device

    # Marlin needs the weights repacked to feed the MMA cores in multiples of 16
    marlin_q = ops.gptq_marlin_repack(qweight, k, n, bits)

    # It also permutes the scales corresponding to the weights
    marlin_s = marlin_permute_scales(gptq_scales, k, n, group_size)

    # No zero point infer it based on wtype
    zp = torch.empty(0, dtype=torch.int, device=device)

    sms = torch.cuda.get_device_properties(device).multi_processor_count

    # Workspace for partial reductions along k
    workspace = torch.zeros(sms * 4, dtype=torch.int, device=device)

    wtype = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128

    return lambda x: ops.marlin_gemm(x, None, marlin_q, None, marlin_s, None, None, zp,
                                     workspace, wtype, x.shape[0], n, k,
                                     use_atomic_add=False, use_fp32_reduce=True)

def make_exllama(qweight, qzeros, gptq_scales, bits):
    """Shuffles GPTQ weights for Exllama and returns a function computing x @ W.T."""
    shuffled = qweight.clone()

    # Shuffle packing within each word to optimize pulling out codes
    ops.gptq_shuffle(shuffled, bits)

    return lambda x: ops.gptq_gemm(x, shuffled, qzeros, gptq_scales, True, True, bits)

def vm_peak_gib():
    """Returns this process's peak address space"""
    kb = int(re.search(r"VmPeak:\s+(\d+)", open("/proc/self/status").read()).group(1))

    return kb / 2**20

def graph_time_us(run, x, calls=20):
    """Times run(x) inside a CUDA graph so Python launch overhead is excluded"""
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for c in range(calls):
            run(x)

    return mean_time_us(graph.replay) / calls

def run_shape(label, n, k):
    """Checks Marlin and Exllama on one shape and prints one row of mean latencies."""
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    weights = torch.randn(n, k, device="cuda", dtype=torch.float16).mul_(k ** -0.5)
    timings = [mean_time_us(lambda: F.linear(x, weights))]

    for bits in (8, 4):
        # Run reference FP16 kernel
        codes, scales = quantize_rows(weights, QUANT_GROUP_SIZE, (1 << (bits - 1)) - 1)
        dequantized = codes.half().view(n, -1, QUANT_GROUP_SIZE).mul_(scales.unsqueeze(-1))

        reference = F.linear(x, dequantized.view(n, k)).float()

        del dequantized

        # Convert to gptq for marlin/exllama
        qweight, qzeros, gptq_scales = to_gptq(codes, scales, bits)

        del codes

        backends = {"Marlin": make_marlin(qweight, gptq_scales, bits, n, k, QUANT_GROUP_SIZE),
                    "Exllama": make_exllama(qweight, qzeros, gptq_scales, bits)}
                    
        del qweight

        # Run both marlin and exllama 
        for name, run in backends.items():
            err = (run(x).float() - reference).abs().max().item()
            assert err < 5e-2, f"{label} W{bits}A16 {name}: max error {err:.2e}"

            # Benchmark
            timings.append(graph_time_us(run, x))

        del backends
    del weights

    torch.cuda.empty_cache()
    
    print(f"{label:20} {n:>7} {k:>6}" + "".join(f"{us:>14.2f}" for us in timings))

if __name__ == "__main__":
    torch.manual_seed(SEED)

    columns = ["PyTorch FP16", "W8 Marlin", "W8 Exllama", "W4 Marlin", "W4 Exllama"]

    print(f"{torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    print(f"Mean latency in us ({WARMUP} discarded warmups, {RUNS} timed calls), group size {QUANT_GROUP_SIZE}")

    print(f"{'Shape':20} {'N':>7} {'K':>6}" + "".join(f"{c:>14}" for c in columns))

    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        for label, n, k in LLAMA_SHAPES:
            run_shape(label, n, k)

    print(f"Peak address space {vm_peak_gib():.2f} GiB of 16")