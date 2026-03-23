import math
import torch


def quantize_slope(s, n_terms=2, min_exp=-16, max_exp=4):
    """Approx slope as sum of n_terms signed powers of 2"""
    terms = []
    remaining = s
    for _ in range(n_terms):
        if abs(remaining) < 2.0 ** (min_exp - 1):
            break

        sign = 1 if remaining > 0 else -1
        log_val = math.log2(abs(remaining))
        candidates = [int(math.floor(log_val)), int(math.ceil(log_val))]
        candidates = [e for e in candidates if min_exp <= e <= max_exp]
        if not candidates:
            break

        best_exp = min(candidates, key=lambda e: abs(abs(remaining) - 2.0**e))
        terms.append((sign, best_exp))
        remaining -= sign * 2.0**best_exp

    val = sum(sgn * 2.0**exp for sgn, exp in terms) if terms else 0.0
    return terms, val


def _call_f(f, x):
    """Call f on a scalar, handling torch funcs that need tensors."""
    result = f(torch.tensor(x, dtype=torch.float64))
    return result.item() if isinstance(result, torch.Tensor) else result


def fit_segment(f, x0, x1, n_terms=2, n_samples=1000):
    """Fit one segment: quantized slope + Chebyshev-optimal intercept."""
    s_ideal = (_call_f(f, x1) - _call_f(f, x0)) / (x1 - x0) if x1 != x0 else 0.0
    terms, s_q = quantize_slope(s_ideal, n_terms)

    xs = torch.linspace(x0, x1, n_samples, dtype=torch.float64)
    residuals = f(xs) - s_q * xs
    b = (residuals.max().item() + residuals.min().item()) / 2
    return terms, s_q, b


def fit_pwl(f, breakpoints, n_terms=2):
    """Fit PWL with quantized slopes. Returns (slopes, intercepts, terms)."""
    slopes, intercepts, all_terms = [], [], []
    for i in range(len(breakpoints) - 1):
        terms, s_q, b = fit_segment(f, breakpoints[i], breakpoints[i + 1], n_terms)
        slopes.append(s_q)
        intercepts.append(b)
        all_terms.append(terms)

    return slopes, intercepts, all_terms


def _segment_masks(x, breakpoints, n_segments):
    """Yield (index, mask) per segment. Last segment closed on right."""
    for i in range(n_segments):
        left = x >= breakpoints[i]
        if i == n_segments - 1:
            right = x <= breakpoints[i + 1] if i + 1 < len(breakpoints) else True

        else:
            right = x < breakpoints[i + 1]

        yield i, left & right


def eval_pwl(x, breakpoints, slopes, intercepts):
    """Evaluate piecewise linear function."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)

    y = slopes[-1] * x + intercepts[-1]
    for i, mask in _segment_masks(x, breakpoints, len(slopes)):
        y[mask] = slopes[i] * x[mask] + intercepts[i]

    return y


def eval_pwl_shifts(x, breakpoints, slope_terms, intercepts):
    """Evaluate PWL using only shifts and adds. No multiplier."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)

    y = torch.full_like(x, intercepts[-1])
    for i, mask in _segment_masks(x, breakpoints, len(slope_terms)):
        xm = x[mask]
        acc = sum(
            (sign * xm * 2.0**exp for sign, exp in slope_terms[i]), torch.zeros_like(xm)
        )
        y[mask] = acc + intercepts[i]

    return y


def _segment_error(f, x0, x1, n_terms=2, n_samples=500):
    """Max abs error of a single quantized-slope segment."""
    _, s_q, b = fit_segment(f, x0, x1, n_terms, n_samples)
    xs = torch.linspace(x0, x1, n_samples, dtype=torch.float64)
    return torch.max(torch.abs(f(xs) - (s_q * xs + b))).item()


def auto_segment(f, x_lo, x_hi, target_mae, n_terms=2, tol=1e-6):
    """
    Find fewest segments to approx f under target_mae.
    Greedy left-to-right binary search for longest valid segment.
    """
    breakpoints = [x_lo]
    x = x_lo
    while x < x_hi - tol:
        lo, hi = x, x_hi

        for _ in range(60):
            mid = (lo + hi) / 2
            if mid - x < tol:
                break

            if _segment_error(f, x, mid, n_terms) <= target_mae:
                lo = mid

            else:
                hi = mid

        if lo - x < tol:
            lo = min(x + (x_hi - x_lo) / 1000, x_hi)

        breakpoints.append(lo)
        x = lo

    breakpoints[-1] = x_hi
    return breakpoints


def max_abs_error(f, breakpoints, slopes, intercepts, n_samples=10000):
    x = torch.linspace(breakpoints[0], breakpoints[-1], n_samples, dtype=torch.float64)
    return torch.max(
        torch.abs(f(x) - eval_pwl(x, breakpoints, slopes, intercepts))
    ).item()


def terms_to_str(terms):
    parts = []
    for i, (sign, exp) in enumerate(terms):
        shift = f"x<<{exp}" if exp > 0 else ("x" if exp == 0 else f"x>>{-exp}")
        if i == 0:
            parts.append(f"-{shift}" if sign < 0 else shift)

        else:
            parts.append(f" - {shift}" if sign < 0 else f" + {shift}")

    return "".join(parts) if parts else "0"
