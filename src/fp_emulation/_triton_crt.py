import math

import torch
import triton
import triton.language as tl


def _precompute_crt_weights(moduli):
    """CRT weights as triple-float (hi, mid, lo). Exact via Python bigints."""
    M = math.prod(moduli)
    wh, wm, wl = [], [], []
    for m in moduli:
        Mi = M // m
        w = Mi * pow(Mi, -1, m)
        hi = float(w)
        remainder = w - int(hi)
        mid = float(remainder)
        lo = float(remainder - int(mid))
        wh.append(hi)
        wm.append(mid)
        wl.append(lo)

    M_hi = float(M)
    M_lo = float(M - int(M_hi))
    inv_M = float(1.0 / float(M))
    return wh, wm, wl, M_hi, M_lo, inv_M


@triton.jit
def _crt_kernel(
    residues_ptr,  # [n_moduli, rows, cols]
    wh_ptr,
    wm_ptr,
    wl_ptr,  # CRT weights (hi, mid, lo)
    constants_ptr,  # [M_hi, M_lo, inv_M, scale_sq] as fp64 tensor
    row_max_ptr,
    col_max_ptr,
    out_ptr,
    rows,
    cols,
    n_moduli: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_elems = rows * cols
    mask = offs < n_elems

    # load fp64 constants from tensor (scalar args get truncated to fp32)
    M_hi = tl.load(constants_ptr + 0).to(tl.float64)
    M_lo = tl.load(constants_ptr + 1).to(tl.float64)
    inv_M = tl.load(constants_ptr + 2).to(tl.float64)
    scale_sq = tl.load(constants_ptr + 3).to(tl.float64)

    acc_hi = tl.zeros([BLOCK], dtype=tl.float64)
    acc_lo = tl.zeros([BLOCK], dtype=tl.float64)

    for i in tl.static_range(n_moduli):
        r = tl.load(residues_ptr + i * n_elems + offs, mask=mask).to(tl.float64)
        w_hi = tl.load(wh_ptr + i).to(tl.float64)
        w_mid = tl.load(wm_ptr + i).to(tl.float64)
        w_lo = tl.load(wl_ptr + i).to(tl.float64)

        # two-product via FMA
        p = r * w_hi
        e = tl.fma(r, w_hi, -p)

        # two-sum accumulate
        s = acc_hi + p
        v = s - acc_hi
        err = (acc_hi - (s - v)) + (p - v)
        acc_hi = s
        acc_lo = acc_lo + tl.fma(r, w_mid, e) + r * w_lo + err

    # symmetric modulo via round-and-subtract
    q = tl.extra.cuda.libdevice.rint((acc_hi + acc_lo) * inv_M)
    pm = q * M_hi
    em = tl.fma(q, M_hi, -pm)
    result = (acc_hi - pm) + (acc_lo - em - q * M_lo)

    row_idx = offs // cols
    col_idx = offs % cols
    row_scale = tl.load(row_max_ptr + row_idx, mask=mask)
    col_scale = tl.load(col_max_ptr + col_idx, mask=mask)
    result = result / scale_sq * row_scale * col_scale

    tl.store(out_ptr + offs, result, mask=mask)


def crt_reconstruct(residues, moduli, scale_sq, a_row_max, b_col_max, device):
    """CRT reconstruct via Triton triple-float FMA kernel."""
    wh, wm, wl, M_hi, M_lo, inv_M = _precompute_crt_weights(moduli)
    n_mod = len(residues)
    rows, cols = residues[0].shape

    res_stack = torch.stack(residues).contiguous()
    wh_t = torch.tensor(wh, dtype=torch.float64, device=device)
    wm_t = torch.tensor(wm, dtype=torch.float64, device=device)
    wl_t = torch.tensor(wl, dtype=torch.float64, device=device)

    # pack fp64 constants into tensor to avoid fp32 truncation of scalar args
    constants = torch.tensor(
        [M_hi, M_lo, inv_M, scale_sq], dtype=torch.float64, device=device
    )
    out = torch.empty(rows, cols, dtype=torch.float64, device=device)

    n_elems = rows * cols
    BLOCK = 1024
    grid = (triton.cdiv(n_elems, BLOCK),)

    _crt_kernel[grid](
        res_stack,
        wh_t,
        wm_t,
        wl_t,
        constants,
        a_row_max,
        b_col_max,
        out,
        rows,
        cols,
        n_moduli=n_mod,
        BLOCK=BLOCK,
        enable_fp_fusion=False,
    )

    return out
