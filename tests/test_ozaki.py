import pytest
import torch
from fp_emulation.ozaki import ozaki2_int8_matmul


def test_accuracy(compensated_matmul):
    """Within fp64 precision of compensated reference."""
    rng = torch.Generator().manual_seed(42)
    A = torch.randn(16, 16, dtype=torch.float64, generator=rng)
    B = torch.randn(16, 16, dtype=torch.float64, generator=rng)
    err = torch.max(
        torch.abs(ozaki2_int8_matmul(A, B) - compensated_matmul(A, B))
    ).item()
    assert err < 1e-14


def test_rect():
    """Non-square matrices."""
    rng = torch.Generator().manual_seed(99)
    A = torch.randn(12, 20, dtype=torch.float64, generator=rng)
    B = torch.randn(20, 8, dtype=torch.float64, generator=rng)
    C = ozaki2_int8_matmul(A, B)
    assert C.shape == (12, 8)
    assert torch.allclose(C, A @ B, atol=1e-12)


@pytest.mark.cuda
def test_cuda():
    """Triton CRT kernel on GPU matches CPU reference."""
    rng = torch.Generator().manual_seed(42)
    A = torch.randn(16, 16, dtype=torch.float64, generator=rng)
    B = torch.randn(16, 16, dtype=torch.float64, generator=rng)
    cpu_result = ozaki2_int8_matmul(A, B)
    cuda_result = ozaki2_int8_matmul(A.cuda(), B.cuda()).cpu()
    assert torch.allclose(cpu_result, cuda_result, atol=1e-12)
