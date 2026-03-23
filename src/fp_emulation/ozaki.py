import math
import torch

_SCALE = float(2**52)

# fmt: off
# primes < 128 (residues fit in signed int8)
_INT8_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
# fmt: on


def _n_moduli_needed(inner, primes):
    """Min primes so product covers dot product range."""
    target = 2 * inner * _SCALE * _SCALE
    prod = 1
    for i, p in enumerate(primes):
        prod *= p
        if prod > target:
            return i + 1

    raise ValueError(f"not enough primes for inner dim {inner}")


def _pad_for_int_mm(x, min_dim=32):
    """Pad to multiples of 8, min min_dim for _int_mm."""

    def _next(d):
        return max(min_dim, d + (-d) % 8)

    pm, pn = _next(x.shape[0]) - x.shape[0], _next(x.shape[1]) - x.shape[1]
    if pm == 0 and pn == 0:
        return x

    return torch.nn.functional.pad(x, (0, pn, 0, pm))


def _int8_matmul(a, b):
    """INT8 matmul -> int32. Tensor cores on CUDA, fallback on CPU."""
    if a.is_cuda:
        m, n = a.shape[0], b.shape[1]
        return torch._int_mm(_pad_for_int_mm(a), _pad_for_int_mm(b))[:m, :n]

    return a.to(torch.int32) @ b.to(torch.int32)


def _mod_matmul(A_int, B_int, m):
    return _int8_matmul((A_int % m).to(torch.int8), (B_int % m).to(torch.int8)) % m


def _crt(residues, moduli):
    """CRT via Python bigints. Exact."""
    M = math.prod(moduli)
    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        x += r * Mi * pow(Mi, -1, m)

    x %= M
    if x > M // 2:
        x -= M

    return x


def _ozaki2_cpu(A, B, moduli, residues, a_row_max, b_col_max):
    """CPU CRT reconstruction via Python bigints."""
    rows, cols = A.shape[0], B.shape[1]
    C = torch.empty(rows, cols, dtype=torch.float64)
    for i in range(rows):
        for j in range(cols):
            rs = [r[i, j].item() for r in residues]
            C[i, j] = (
                _crt(rs, moduli) / _SCALE**2 * a_row_max[i].item() * b_col_max[j].item()
            )

    return C


def _ozaki2_cuda(A, B, moduli, residues, a_row_max, b_col_max):
    """Lazy CUDA CRT reconstruction w/ Triton triple-float FMA kernel."""
    from fp_emulation._triton_crt import crt_reconstruct

    return crt_reconstruct(residues, moduli, _SCALE**2, a_row_max, b_col_max, A.device)


def _ozaki2_forward(A, B):
    """Scale -> modular INT8 matmuls -> CRT reconstruct."""
    n_mod = _n_moduli_needed(A.shape[1], _INT8_PRIMES)
    moduli = _INT8_PRIMES[:n_mod]

    a_row_max = torch.clamp(A.abs().amax(dim=1), min=1e-300)
    b_col_max = torch.clamp(B.abs().amax(dim=0), min=1e-300)

    A_int = torch.round(A / a_row_max.unsqueeze(1) * _SCALE).to(torch.int64)
    B_int = torch.round(B / b_col_max.unsqueeze(0) * _SCALE).to(torch.int64)

    residues = [_mod_matmul(A_int, B_int, m) for m in moduli]

    if A.is_cuda:
        return _ozaki2_cuda(A, B, moduli, residues, a_row_max, b_col_max)

    return _ozaki2_cpu(A, B, moduli, residues, a_row_max, b_col_max)


class _OzakiMatmul(torch.autograd.Function):
    @staticmethod
    def forward(A, B):
        return _ozaki2_forward(A, B)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        A, B = ctx.saved_tensors
        grad_A = ozaki2_int8_matmul(grad, B.T) if ctx.needs_input_grad[0] else None
        grad_B = ozaki2_int8_matmul(A.T, grad) if ctx.needs_input_grad[1] else None
        return grad_A, grad_B


def ozaki2_int8_matmul(A, B):
    """FP64-accurate matmul via INT8 tensor cores (Ozaki scheme II + CRT)."""
    return _OzakiMatmul.apply(A, B)
