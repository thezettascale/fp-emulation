import math
import torch


def split(a: float, s: int = 27) -> tuple[float, float]:
    """Veltkamp trick: split fp64 into two ~26-bit halves."""
    factor = float(2**s + 1)
    c = factor * a
    a_hi = c - (c - a)
    return a_hi, a - a_hi


def two_product(a: float, b: float) -> tuple[float, float]:
    """(p, e) where a * b = p + e exactly."""
    p = a * b
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def two_product_fma(a: float, b: float) -> tuple[float, float]:
    """TwoProduct via fused FMA."""
    p = a * b
    return p, math.fma(a, b, -p)


def extract_slice(A, mu):
    """Matrix Veltkamp split. Round rows to ~26 sig bits."""
    exp = torch.ceil(torch.log2(torch.clamp(mu, min=1e-300))).to(torch.int32) + 27
    sigma = torch.ldexp(torch.ones_like(mu), exp).unsqueeze(1)
    S = (sigma + A) - sigma
    return S, A - S


def row_max_abs(A):
    return torch.max(torch.abs(A), dim=1).values


def col_max_abs(B):
    return torch.max(torch.abs(B), dim=0).values


def ozaki_split(A, n_slices):
    """Split A into n_slices of limited significand bits."""
    slices, R = [], A.clone()
    for _ in range(n_slices - 1):
        S, R = extract_slice(R, row_max_abs(R))
        slices.append(S)

    slices.append(R)
    return slices


def ozaki1_matmul(A, B, n_slices=4):
    """
    Ozaki Scheme I — split + cross-product acc in fp64.
    Cost: n_slices^2 matmuls.
    """
    A_slices = ozaki_split(A, n_slices)
    B_slices = [s.T for s in ozaki_split(B.T.contiguous(), n_slices)]

    C = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float64, device=A.device)
    for i in range(n_slices):
        for j in range(n_slices):
            C += A_slices[i] @ B_slices[j]

    return C


def compensated_matmul(A, B):
    """Reference: TwoProduct + Kahan acc per element (scalar, slow)."""
    p, q = A.shape
    r = B.shape[1]
    C = torch.zeros(p, r, dtype=torch.float64, device=A.device)

    for i in range(p):
        for j in range(r):
            s, e = 0.0, 0.0
            for k in range(q):
                prod, ep = two_product_fma(A[i, k].item(), B[k, j].item())
                y = prod - e
                t = s + y
                e = (t - s) - y
                s = t
                s += ep

            C[i, j] = s

    return C


def _is_prime(n):
    if n < 2:
        return False

    if n < 4:
        return True

    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False

        i += 6

    return True


def _primes_above(start, count):
    """Find `count` primes >= start."""
    primes = []
    n = start if start % 2 != 0 else start + 1
    while len(primes) < count:
        if _is_prime(n):
            primes.append(n)

        n += 2

    return primes


def _crt(residues, moduli):
    """
    Chinese remainder theorem

    Recover x from x % m_i = r_i (Python ints, arbitrary precision).
    """
    M = math.prod(moduli)
    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        x += r * Mi * pow(Mi, -1, m)

    x %= M
    if x > M // 2:
        x -= M

    return x


def ozaki2_matmul(A, B, n_moduli=5):
    """
    Ozaki Scheme II — CRT-based matmul.
    L modular int64 matmuls + CRT reconstruction. Cost: L matmuls.
    """
    moduli = _primes_above(2**26, n_moduli)
    scale = 2**52

    a_row_max = torch.clamp(row_max_abs(A), min=1e-300)
    b_col_max = torch.clamp(col_max_abs(B), min=1e-300)

    A_int = torch.round(A / a_row_max.unsqueeze(1) * scale).to(torch.int64)
    B_int = torch.round(B / b_col_max.unsqueeze(0) * scale).to(torch.int64)

    # modular matmuls (vectorized, the expensive part)
    residues = [(A_int % m) @ (B_int % m) % m for m in moduli]

    # CRT per element (python ints for arbitrary precision)
    rows, cols = A.shape[0], B.shape[1]
    C = torch.empty(rows, cols, dtype=torch.float64, device=A.device)
    for i in range(rows):
        for j in range(cols):
            rs = [r[i, j].item() for r in residues]
            C[i, j] = (
                _crt(rs, moduli) / scale**2 * a_row_max[i].item() * b_col_max[j].item()
            )

    return C


def main():
    rng = torch.Generator().manual_seed(42)

    sizes = [8, 32, 64]
    n_slices_list = [2, 3, 4, 6]

    print("Ozaki Scheme I: accuracy vs native FP64 matmul")
    print("reference: compensated dot product (TwoProduct + Kahan)\n")

    for n in sizes:
        A = torch.randn(n, n, dtype=torch.float64, generator=rng)
        B = torch.randn(n, n, dtype=torch.float64, generator=rng)

        C_naive = A @ B
        C_ref = compensated_matmul(A, B)

        err_naive = torch.max(torch.abs(C_naive - C_ref)).item()
        print(f"n={n}:")
        print(f" naive fp64: max|err| = {err_naive:.3e}")

        for ns in n_slices_list:
            C_oz = ozaki1_matmul(A, B, n_slices=ns)
            err = torch.max(torch.abs(C_oz - C_ref)).item()
            print(f" ozaki1 k={ns:d} ({ns * ns:2d} muls): max|err| = {err:.3e}")

        print()


if __name__ == "__main__":
    main()
