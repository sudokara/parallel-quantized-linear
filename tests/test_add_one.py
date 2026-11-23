import torch
import qlinear

def test_add_one():
    x = torch.randn(1000, device="cuda")
    y = torch.ops.qlinear.add_one(x)
    torch.testing.assert_close(y, x + 1)