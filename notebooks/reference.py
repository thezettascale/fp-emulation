import math
import torch


def split(a: float, s: int = 27) -> tuple[float, float]:
    """Veltkamp trick: split fp64 into two ~26-bit halves."""
    factor = float(2**s + 1)
    c = factor * a
    a_hi = c - (c - a)
    return a_hi, a - a_hi


def two_product(a: float, b: float) -> tuple[float, float]:
    """(p, e) where a * b = p + e exactly."""
    p = a * b
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def two_product_fma(a: float, b: float) -> tuple[float, float]:
    """TwoProduct via fused FMA."""
    p = a * b
    return p, math.fma(a, b, -p)


def compensated_matmul(A, B):
    """TwoProduct + Kahan acc per element. Scalar, slow, but accurate."""
    p, q = A.shape
    r = B.shape[1]
    C = torch.zeros(p, r, dtype=torch.float64, device=A.device)

    for i in range(p):
        for j in range(r):
            s, e = 0.0, 0.0
            for k in range(q):
                prod, ep = two_product_fma(A[i, k].item(), B[k, j].item())
                y = prod - e
                t = s + y
                e = (t - s) - y
                s = t
                s += ep

            C[i, j] = s

    return C
