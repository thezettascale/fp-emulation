import numpy as np


def quantize_slope(s, n_terms=2, min_exp=-16, max_exp=4):
    """Approximate slope as sum of n_terms signed powers of 2 (greedy)."""
    terms = []
    remaining = s
    for _ in range(n_terms):
        if abs(remaining) < 2.0 ** (min_exp - 1):
            break

        sign = 1 if remaining > 0 else -1
        log_val = np.log2(abs(remaining))
        candidates = [int(np.floor(log_val)), int(np.ceil(log_val))]
        candidates = [e for e in candidates if min_exp <= e <= max_exp]
        if not candidates:
            break

        best_exp = min(candidates, key=lambda e: abs(abs(remaining) - 2.0**e))
        terms.append((sign, best_exp))
        remaining -= sign * 2.0**best_exp

    val = sum(sgn * 2.0**exp for sgn, exp in terms) if terms else 0.0
    return terms, val


def fit_segment(f, x0, x1, n_terms=2, n_samples=1000):
    """Fit one segment: quantized slope + Chebyshev-optimal intercept."""
    y0, y1 = f(x0), f(x1)
    s_ideal = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0

    terms, s_q = quantize_slope(s_ideal, n_terms)

    xs = np.linspace(x0, x1, n_samples)
    residuals = f(xs) - s_q * xs
    b = (np.max(residuals) + np.min(residuals)) / 2

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
    """Return (index, mask) for each segment. Last segment uses <= on right."""
    for i in range(n_segments):
        x0 = breakpoints[i]
        x1 = breakpoints[i + 1] if i + 1 < len(breakpoints) else np.inf
        if i == n_segments - 1:
            yield i, (x >= x0) & (x <= x1)

        else:
            yield i, (x >= x0) & (x < x1)


def eval_pwl(x, breakpoints, slopes, intercepts):
    """Evaluate piecewise linear function."""
    x = np.asarray(x, dtype=np.float64)
    y = np.full_like(x, slopes[-1] * x + intercepts[-1])
    for i, mask in _segment_masks(x, breakpoints, len(slopes)):
        y[mask] = slopes[i] * x[mask] + intercepts[i]

    return y


def eval_pwl_shifts(x, breakpoints, slope_terms, intercepts):
    """Evaluate PWL using only shifts and adds. No multiplier."""
    x = np.asarray(x, dtype=np.float64)
    y = np.full_like(x, intercepts[-1])
    for i, mask in _segment_masks(x, breakpoints, len(slope_terms)):
        xm = x[mask]
        acc = sum(sign * np.ldexp(xm, exp) for sign, exp in slope_terms[i])
        y[mask] = acc + intercepts[i]

    return y


def _segment_error(f, x0, x1, n_terms=2, n_samples=500):
    """Max abs error of a single quantized-slope segment."""
    _, s_q, b = fit_segment(f, x0, x1, n_terms, n_samples)
    xs = np.linspace(x0, x1, n_samples)
    return np.max(np.abs(f(xs) - (s_q * xs + b)))


def auto_segment(f, x_lo, x_hi, target_mae, n_terms=2, tol=1e-6):
    """
    Find fewest segments to approx f under target_mae.

    Greedy left-to-right: binary search for the longest segment
    that stays under the error bound, then start a new segment.
    """
    breakpoints = [x_lo]
    x = x_lo
    while x < x_hi - tol:
        lo, hi = x, x_hi

        # binary search
        for _ in range(60):
            mid = (lo + hi) / 2
            if mid - x < tol:
                break

            err = _segment_error(f, x, mid, n_terms)
            if err <= target_mae:
                lo = mid

            else:
                hi = mid

        # lo is farthest point under threshold
        if lo - x < tol:
            lo = min(x + (x_hi - x_lo) / 1000, x_hi)  # Force min step

        breakpoints.append(lo)
        x = lo

    # snap last to x_hi
    breakpoints[-1] = x_hi
    return breakpoints


def max_abs_error(f, breakpoints, slopes, intercepts, n_samples=10000):
    x = np.linspace(breakpoints[0], breakpoints[-1], n_samples)
    return np.max(np.abs(f(x) - eval_pwl(x, breakpoints, slopes, intercepts)))


def terms_to_str(terms):
    parts = []
    for i, (sign, exp) in enumerate(terms):
        if exp >= 0:
            shift = f"x<<{exp}" if exp > 0 else "x"
        else:
            shift = f"x>>{-exp}"

        if i == 0:
            parts.append(f"-{shift}" if sign < 0 else shift)

        else:
            parts.append(f" - {shift}" if sign < 0 else f" + {shift}")

    return "".join(parts) if parts else "0"
