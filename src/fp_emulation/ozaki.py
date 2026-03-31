import math

import torch

# fmt: off
_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
]
# fmt: on

_PRECISION_BITS = {
    torch.float16: 11,
    torch.bfloat16: 8,
    torch.float32: 24,
    torch.float64: 53,
}


def _n_moduli(k, bits):
    """Min primes whose product covers dot-product range."""
    target = 2 * k * (2 ** (2 * bits))
    prod = 1
    for i, p in enumerate(_PRIMES):
        prod *= p
        if prod > target:
            return i + 1

    raise ValueError(f"not enough primes for inner dim {k}, bits {bits}")


def _scale_to_int(A, B, bits):
    """Power-of-2 scaling (exact in FP), round to int."""
    row_exp = A.abs().amax(dim=-1).clamp(min=1e-300).frexp()[1]
    col_exp = B.abs().amax(dim=-2).clamp(min=1e-300).frexp()[1]
    A_int = torch.ldexp(A, bits - row_exp.unsqueeze(-1)).round().to(torch.int64)
    B_int = torch.ldexp(B, bits - col_exp.unsqueeze(-2)).round().to(torch.int64)
    return A_int, B_int, row_exp, col_exp


def _residues(X_int, moduli):
    """int64 -> stacked int8 residues per prime."""
    moduli_t = torch.tensor(moduli, dtype=torch.int64, device=X_int.device).view(
        -1, *([1] * X_int.ndim)
    )
    return (X_int.unsqueeze(0) % moduli_t).to(torch.int8)


def _matmul_residues(a_res, b_res, moduli):
    """Per-prime int8 matmul + modular reduction."""
    if a_res.is_cuda:
        from fp_emulation._cuda_crt import cuda_batched_int8_gemm_mod

        return cuda_batched_int8_gemm_mod(a_res, b_res, moduli)

    moduli_t = torch.tensor(moduli, dtype=torch.int32, device=a_res.device).view(
        -1, 1, 1
    )
    return list((a_res.int() @ b_res.int()) % moduli_t)


def _crt_weights(moduli):
    """CRT weights as triple-float (hi, mid, lo) for Kahan-accurate reconstruction."""
    M = math.prod(moduli)
    wh, wm, wl = [], [], []
    for m in moduli:
        Mi = M // m
        w = Mi * pow(Mi, -1, m)
        hi = float(w)
        r = w - int(hi)
        mid = float(r)
        lo = float(r - int(mid))
        wh.append(hi)
        wm.append(mid)
        wl.append(lo)

    M_hi = float(M)
    M_lo = float(M - int(M_hi))
    inv_M = 1.0 / float(M)
    return wh, wm, wl, M_hi, M_lo, inv_M


def _reconstruct(residues, moduli, bits, row_exp, col_exp):
    """CRT reconstruction via compiled C++/CUDA backends."""
    if residues[0].is_cuda:
        from fp_emulation._cuda_crt import cuda_crt_reconstruct

        return cuda_crt_reconstruct(residues, moduli, bits, row_exp, col_exp)

    from fp_emulation._cpu_crt import cpu_crt_reconstruct

    return cpu_crt_reconstruct(residues, moduli, bits, row_exp, col_exp)


def _ozaki2_forward(A, B):
    """Scale -> modular INT8 matmuls -> CRT reconstruct."""
    bits = _PRECISION_BITS.get(A.dtype)
    if bits is None:
        supported = ", ".join(str(d) for d in _PRECISION_BITS)
        raise TypeError(
            f"unsupported dtype {A.dtype}. "
            f"CRT matmul emulates floating-point precision via int8 arithmetic; "
            f"input must be a float type ({supported})"
        )

    orig_dtype = A.dtype
    A, B = A.double(), B.double()
    n_mod = _n_moduli(A.shape[-1], bits)
    moduli = _PRIMES[:n_mod]

    A_int, B_int, row_exp, col_exp = _scale_to_int(A, B, bits)
    a_res = _residues(A_int, moduli)
    b_res = _residues(B_int, moduli)
    residues = _matmul_residues(a_res, b_res, moduli)

    result = _reconstruct(residues, moduli, bits, row_exp, col_exp)
    if orig_dtype != torch.float64:
        result = result.to(orig_dtype)
    return result


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
    """
    FP-accurate matmul via INT8 tensor cores (Ozaki scheme II + CRT).

    Precision matches input dtype (fp16, bf16, fp32, fp64).
    """
    return _OzakiMatmul.apply(A, B)
