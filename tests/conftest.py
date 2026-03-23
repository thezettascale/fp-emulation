import math
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip)


def _two_product_fma(a, b):
    p = a * b
    return p, math.fma(a, b, -p)


def _compensated_matmul(A, B):
    """TwoProduct + Kahan acc. Scalar reference, slow but accurate."""
    p, q = A.shape
    r = B.shape[1]
    C = torch.zeros(p, r, dtype=torch.float64, device=A.device)

    for i in range(p):
        for j in range(r):
            s, e = 0.0, 0.0
            for k in range(q):
                prod, ep = _two_product_fma(A[i, k].item(), B[k, j].item())
                y = prod - e
                t = s + y
                e = (t - s) - y
                s = t
                s += ep
            C[i, j] = s

    return C


@pytest.fixture
def compensated_matmul():
    return _compensated_matmul
