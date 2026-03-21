import struct
import math
from decimal import Decimal, getcontext


def sep(title: str):
    print("-" * 10)
    print(f"{title}:")


def end():
    print("-" * 10)
    print()


def bits(label: str, x: float):
    """Print IEEE 754 breakdown on two lines."""
    raw = struct.unpack(">Q", struct.pack(">d", x))[0]
    sign = (raw >> 63) & 1
    exp = (raw >> 52) & 0x7FF
    frac = raw & ((1 << 52) - 1)
    sign_ch = "-" if sign else "+"

    sig_bin = f"{frac:052b}"
    grouped = " ".join(sig_bin[i : i + 4] for i in range(0, 52, 4))

    print(f" {label} = {x}")
    pad = " " * (len(label) + 2)
    print(f" {pad}{sign_ch} 1.{grouped} * 2^{exp - 1023}")


def trace_split(a, s=27):
    sep(f"Veltkamp split (s={s})")

    bits("a", a)
    print()

    factor = float(2**s + 1)
    c = factor * a
    bits("c", c)
    cma = c - a
    bits("c-a", cma)
    print()

    a_hi = c - cma
    bits("a_hi", a_hi)
    a_lo = a - a_hi
    bits("a_lo", a_lo)
    print()

    print(f" a_hi + a_lo == a: {a_hi + a_lo == a}")
    end()

    return a_hi, a_lo


def trace_two_product(a, b):
    sep("TwoProduct")

    bits("a", a)
    bits("b", b)
    print()

    p = a * b
    bits("p=a*b", p)
    e = math.fma(a, b, -p)
    bits("e=fma", e)
    print()

    getcontext().prec = 50
    exact = Decimal(a) * Decimal(b)
    recovered = Decimal(p) + Decimal(e)
    print(f" p + e == a*b: {exact == recovered}")
    end()

    return p, e


def trace_kahan(values):
    sep(f"Kahan summation ({len(values)} values)")

    s, c = 0.0, 0.0
    for i, x in enumerate(values):
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t

        if i < 6 or i == len(values) - 1:
            print(f" [{i:>5d}] x={x:>10.4e}  comp={c:>11.2e}  sum={s:.15e}")

    print()
    naive = sum(values)
    print(f" naive = {naive:.15e}")
    print(f" kahan = {s:.15e}")
    end()

    return s


def trace_fixed(v, qf=8):
    """Q8.8 fixed-point encoding."""
    scale = 1 << qf
    raw = int(v * scale)
    clamped = max(-(1 << 15), min((1 << 15) - 1, raw))
    u = clamped & 0xFFFF

    b = f"{u:016b}"
    recovered = clamped / scale
    print(f" {v:>8.4f} -> {b[:8]}.{b[8:]}  ({recovered:.4f}, err={v - recovered:+.6f})")


if __name__ == "__main__":
    trace_split(3.141592653589793)
    trace_two_product(3.141592653589793, 2.718281828459045)
    trace_kahan([1.0] + [1e-16] * 10000)

    sep("Q8.8 fixed-point")
    for v in [1.5, -1.0, 2.25, 0.1, -42.75]:
        trace_fixed(v)
    end()

    sep("Q8.8 MAC: 1.5*2.25 + (-1.0)*4.0")
    scale = 256
    a1, b1 = int(1.5 * scale), int(2.25 * scale)
    a2, b2 = int(-1.0 * scale), int(4.0 * scale)
    acc = a1 * b1 + a2 * b2
    print(f" {a1} * {b1} = {a1 * b1}")
    print(f" {a2} * {b2} = {a2 * b2}")
    print()
    print(f" acc = {acc} -> {acc / (scale * scale)}")
    end()
