import math
import torch


def _row_max_abs(A):
    return torch.max(torch.abs(A), dim=1).values


def _col_max_abs(B):
    return torch.max(torch.abs(B), dim=0).values


_SCALE = 2**52


def _is_prime(n):
    if n < 2:
        return False

    if n < 4:
        return True

    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False

        i += 6

    return True


# primes that fit in signed int8 (residues in [0, p-1] <= 126)
_INT8_PRIMES = [p for p in range(3, 128) if _is_prime(p)]


def _n_moduli_needed(inner, primes):
    """Min primes so product covers the dot product range."""
    target = 2 * inner * _SCALE * _SCALE
    prod = 1
    for i, p in enumerate(primes):
        prod *= p
        if prod > target:
            return i + 1

    raise ValueError(f"not enough primes for inner dim {inner}")


def _crt(residues, moduli):
    """Chinese remainder theorem. Recover x from residues."""
    M = math.prod(moduli)
    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        x += r * Mi * pow(Mi, -1, m)

    x %= M
    if x > M // 2:
        x -= M

    return x


def _pad_for_int_mm(x, min_dim=32):
    """Pad dims for _int_mm: multiples of 8, minimum min_dim."""

    def _next(d):
        return max(min_dim, d + (-d) % 8)

    pm, pn = _next(x.shape[0]) - x.shape[0], _next(x.shape[1]) - x.shape[1]
    if pm == 0 and pn == 0:
        return x

    return torch.nn.functional.pad(x, (0, pn, 0, pm))


def _int8_matmul(a, b):
    """INT8 matmul -> int32. Tensor cores on CUDA, int32 fallback on CPU."""
    if a.is_cuda:
        m, n = a.shape[0], b.shape[1]
        return torch._int_mm(_pad_for_int_mm(a), _pad_for_int_mm(b))[:m, :n]

    return a.to(torch.int32) @ b.to(torch.int32)


def _mod_matmul(A_int, B_int, m):
    return _int8_matmul((A_int % m).to(torch.int8), (B_int % m).to(torch.int8)) % m


def ozaki2_int8_matmul(A, B):
    """FP64-accurate matmul via INT8 tensor cores (Ozaki Scheme II + CRT)."""
    n_mod = _n_moduli_needed(A.shape[1], _INT8_PRIMES)
    moduli = _INT8_PRIMES[:n_mod]

    a_row_max = torch.clamp(_row_max_abs(A), min=1e-300)
    b_col_max = torch.clamp(_col_max_abs(B), min=1e-300)

    A_int = torch.round(A / a_row_max.unsqueeze(1) * _SCALE).to(torch.int64)
    B_int = torch.round(B / b_col_max.unsqueeze(0) * _SCALE).to(torch.int64)

    residues = [_mod_matmul(A_int, B_int, m) for m in moduli]

    # CRT per element (python ints — overflows int64)
    rows, cols = A.shape[0], B.shape[1]
    C = torch.empty(rows, cols, dtype=torch.float64, device=A.device)
    for i in range(rows):
        for j in range(cols):
            rs = [r[i, j].item() for r in residues]
            C[i, j] = (
                _crt(rs, moduli) / _SCALE**2 * a_row_max[i].item() * b_col_max[j].item()
            )

    return C
