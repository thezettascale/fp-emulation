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


def eval_pwl(x, breakpoints, slopes, intercepts):
    """Evaluate piecewise linear function."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y[:] = slopes[-1] * x + intercepts[-1]

    for i in range(len(slopes)):
        x0 = breakpoints[i]
        x1 = breakpoints[i + 1] if i + 1 < len(breakpoints) else np.inf
        mask = (x >= x0) & (x <= x1) if i == len(slopes) - 1 else (x >= x0) & (x < x1)
        y[mask] = slopes[i] * x[mask] + intercepts[i]

    return y


def eval_pwl_shifts(x, breakpoints, slope_terms, intercepts):
    """Evaluate PWL using only shifts and adds. No multiplier."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y[:] = intercepts[-1]

    for i in range(len(slope_terms)):
        x0 = breakpoints[i]
        x1 = breakpoints[i + 1] if i + 1 < len(breakpoints) else np.inf
        mask = (
            (x >= x0) & (x <= x1) if i == len(slope_terms) - 1 else (x >= x0) & (x < x1)
        )
        xm = x[mask]

        acc = np.zeros_like(xm)
        for sign, exp in slope_terms[i]:
            acc += sign * np.ldexp(xm, exp)

        y[mask] = acc + intercepts[i]

    return y


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
