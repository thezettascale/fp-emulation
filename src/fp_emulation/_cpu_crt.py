from pathlib import Path

import torch

from fp_emulation.ozaki import _crt_weights

_module = None
_CSRC = Path(__file__).resolve().parent / "csrc"


def _load():
    global _module
    if _module is not None:
        return _module

    from torch.utils.cpp_extension import load

    _module = load(
        name="crt_cpu",
        sources=[str(_CSRC / "crt_reconstruct.cpp")],
        extra_cflags=["-O3", "-march=native", "-fopenmp"],
        extra_ldflags=["-lgomp"],
        verbose=False,
    )
    return _module


def cpu_crt_reconstruct(residues, moduli, bits, row_exp, col_exp):
    mod = _load()
    rows, cols = residues[0].shape

    res_stack = torch.stack(residues).to(torch.int32).contiguous()
    wh, wm, wl, M_hi, M_lo, inv_M = _crt_weights(moduli)
    wh_t = torch.tensor(wh, dtype=torch.float64)
    wm_t = torch.tensor(wm, dtype=torch.float64)
    wl_t = torch.tensor(wl, dtype=torch.float64)

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
        len(moduli),
    )
