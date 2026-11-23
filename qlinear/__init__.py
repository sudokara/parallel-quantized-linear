from pathlib import Path
from torch.utils.cpp_extension import load

_CSRC = Path(__file__).resolve().parent.parent / "csrc"

load(
    name="qlinear_ext",
    sources=[str(_CSRC / "ops.cpp"), str(_CSRC / "add_one.cu")],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    is_python_module=False,
)