import torch
from torch import nn
from fp_emulation import convert


def test_convert():
    """Converted model matches native fp64 matmul."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

    ref = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    ref[0].weight.data.copy_(model[0].weight.data.to(torch.float64))
    ref[0].bias.data.copy_(model[0].bias.data.to(torch.float64))
    ref[2].weight.data.copy_(model[2].weight.data.to(torch.float64))
    ref[2].bias.data.copy_(model[2].bias.data.to(torch.float64))

    convert(model)

    x = torch.randn(2, 5, 4, dtype=torch.float64)
    assert torch.allclose(model(x), ref(x), atol=1e-12)
