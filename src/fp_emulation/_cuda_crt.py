from pathlib import Path

import torch

from fp_emulation.ozaki import _crt_weights

_module = None
_CSRC = Path(__file__).parent / "csrc" / "crt_kernel.cu"


def _load():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load

        _module = load(
            name="crt_cuda",
            sources=[str(_CSRC)],
            verbose=False,
        )
    return _module


def _pad4(x):
    """Pad last two dims to multiples of 4 for cuBLAS int8 alignment."""
    pm = (-x.shape[-2]) % 4
    pn = (-x.shape[-1]) % 4
    if pm == 0 and pn == 0:
        return x
    return torch.nn.functional.pad(x, (0, pn, 0, pm))


def cuda_batched_int8_gemm_mod(a_res, b_res, moduli):
    """Batched int8 GEMM via cublasLt."""
    mod = _load()
    m, n = a_res.shape[-2], b_res.shape[-1]
    a_p = _pad4(a_res).contiguous()
    b_p = _pad4(b_res).contiguous()
    moduli_t = torch.tensor(moduli, dtype=torch.int32, device=a_res.device)

    c = mod.batched_int8_gemm_mod(a_p, b_p, moduli_t)
    return list(c[:, :m, :n])


def cuda_crt_reconstruct(residues, moduli, bits, row_exp, col_exp):
    """CRT reconstruction via CUDA kernel."""
    mod = _load()
    n_mod = len(residues)
    rows, cols = residues[0].shape
    device = residues[0].device

    res_stack = torch.stack(residues).contiguous()
    wh, wm, wl, M_hi, M_lo, inv_M = _crt_weights(moduli)
    wh_t = torch.tensor(wh, dtype=torch.float64, device=device)
    wm_t = torch.tensor(wm, dtype=torch.float64, device=device)
    wl_t = torch.tensor(wl, dtype=torch.float64, device=device)

    return mod.crt_reconstruct(
        res_stack,
        wh_t,
        wm_t,
        wl_t,
        M_hi,
        M_lo,
        inv_M,
        row_exp.int().contiguous(),
        col_exp.int().contiguous(),
        2 * bits,
        rows,
        cols,
        n_mod,
    )
