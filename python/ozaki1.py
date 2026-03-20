import numpy as np
import math


def split(a: float, s: int = 27) -> tuple[float, float]:
    """Veltkamp trick. Split fp64 (53-bit significand bits) into high/low parts with <= s-1 = 26 significands."""
    factor = float(2**s + 1)
    c = factor * a

    # <= 26 bits halves
    a_hi = c - (c - a)
    a_lo = a - a_hi

    return a_hi, a_lo


def two_product(a: float, b: float) -> tuple[float, float]:
    """Returns (p, e) where a * b = p + e (rounded product + error)."""
    p = a * b
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def two_product_fma(a: float, b: float) -> tuple[float, float]:
    """TwoProduct via fused FMA."""
    p = a * b
    e = math.fma(a, b, -p)  # a * b + (-p)
    return p, e


def extract_slice(A: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matrix-level Veltkamp split. Round each row to approx 26 significand bits using row scales mu."""

    # power-of-two scale per row: 2^(ceil( log2( mu_i ) ) + 27)
    exp = np.ceil(np.log2(np.maximum(mu, 1e-300))).astype(int) + 27
    sigma = np.ldexp(np.ones_like(mu), exp)[:, np.newaxis]

    # add-then-subtract rounds to limited bits (same trick as scalar split)
    S = (sigma + A) - sigma
    R = A - S

    return S, R


def row_max_abs(A: np.ndarray) -> np.ndarray:
    return np.max(np.abs(A), axis=1)


def col_max_abs(B: np.ndarray) -> np.ndarray:
    return np.max(np.abs(B), axis=0)


def ozaki_split(A: np.ndarray, n_slices: int) -> list[np.ndarray]:
    """Split A into n_slices where each has limited significand bits."""
    slices = []
    R = A.copy()
    for _ in range(n_slices - 1):
        mu = row_max_abs(R)
        S, R = extract_slice(R, mu)
        slices.append(S)

    slices.append(R)  # last is remainder
    return slices


def ozaki1_matmul(A: np.ndarray, B: np.ndarray, n_slices: int = 4) -> np.ndarray:
    """
    Ozaki Scheme I: high-prec matmul via matrix splitting.

    Split A and B into slices, find all cross-products,
    acc in FP64.

    Cost: n_slices * (n_slices + 1) / 2 matmuls.
    """
    A_slices = ozaki_split(A, n_slices)
    B_slices = ozaki_split(B.T, n_slices)  # split B by columns (transpose, split rows)
    B_slices = [s.T for s in B_slices]  # transpose back

    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(n_slices):
        for j in range(n_slices):
            C += A_slices[i] @ B_slices[j]

    return C


def compensated_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reference using TwoProduct + Kahan acc per element."""
    p, q = A.shape
    r = B.shape[1]
    C = np.zeros((p, r), dtype=np.float64)

    for i in range(p):
        for j in range(r):
            s, e = 0.0, 0.0
            for k in range(q):
                prod, ep = two_product_fma(A[i, k], B[k, j])
                y = prod - e
                t = s + y
                e = (t - s) - y
                s = t
                s += ep

            C[i, j] = s

    return C


def main():
    rng = np.random.default_rng(42)

    sizes = [8, 32, 64]
    n_slices_list = [2, 3, 4, 6]

    print("Ozaki Scheme I: accuracy vs native FP64 matmul")
    print("reference: compensated dot product (TwoProduct + Kahan)\n")

    for n in sizes:
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))

        C_naive = A @ B
        C_ref = compensated_matmul(A, B)

        err_naive = np.max(np.abs(C_naive - C_ref))
        print(f"n={n}:")
        print(f" naive fp64: max|err| = {err_naive:.3e}")

        for ns in n_slices_list:
            C_oz = ozaki1_matmul(A, B, n_slices=ns)
            err = np.max(np.abs(C_oz - C_ref))
            print(f" ozaki1 k={ns:d} ({ns * ns:2d} muls): max|err| = {err:.3e}")

        print()


if __name__ == "__main__":
    main()
