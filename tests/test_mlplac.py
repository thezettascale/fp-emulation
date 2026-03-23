import torch
from fp_emulation import fit_pwl, eval_pwl, eval_pwl_shifts, auto_segment


def test_tanh():
    """PWL tanh within target error."""
    bp = auto_segment(torch.tanh, -3, 3, target_mae=1e-3)
    slopes, intercepts, terms = fit_pwl(torch.tanh, bp)

    x = torch.linspace(-3, 3, 10000, dtype=torch.float64)
    err = torch.max(torch.abs(torch.tanh(x) - eval_pwl(x, bp, slopes, intercepts)))
    assert err < 2e-3


def test_shifts():
    """Shift-based eval matches multiply-based eval."""
    bp = auto_segment(torch.tanh, -3, 3, target_mae=1e-3)
    slopes, intercepts, terms = fit_pwl(torch.tanh, bp)

    x = torch.linspace(-3, 3, 1000, dtype=torch.float64)
    y_mul = eval_pwl(x, bp, slopes, intercepts)
    y_shift = eval_pwl_shifts(x, bp, terms, intercepts)
    assert torch.allclose(y_mul, y_shift, atol=1e-12)
