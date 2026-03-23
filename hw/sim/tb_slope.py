import random

import cocotb
from cocotb.triggers import Timer

W = 16
MASK = (1 << W) - 1


def to_unsigned(val):
    return val & MASK


def to_signed(val):
    if val >= (1 << (W - 1)):
        val -= 1 << W
    return val


def ref_slope_shift(dx, sh0, neg0, sh1, neg1):
    """Python ref for slope_shift."""
    t0 = dx >> sh0  # right shift
    if neg0:
        t0 = -t0
    t1 = dx >> sh1
    if neg1:
        t1 = -t1
    # truncate to W bits (signed)
    result = (t0 + t1) & MASK
    return to_signed(result)


@cocotb.test()
async def test_known_values(dut):
    cases = [
        # (dx, sh0, neg0, sh1, neg1, expected)
        (256, 1, 0, 3, 0, 128 + 32),  # 1.0 -> 0.5 + 0.125 = 0.625 in Q8.8
        (256, 0, 0, 1, 1, 256 - 128),  # 1.0 - 0.5 = 0.5
        (-512, 2, 0, 4, 0, -128 + -32),  # -2.0 -> -0.5 + -0.125
        (256, 1, 1, 3, 1, -128 + -32),  # negate both terms
        (0, 0, 0, 0, 0, 0),  # zero in, zero out
    ]

    for dx, sh0, neg0, sh1, neg1, expected in cases:
        dut.dx_i.value = to_unsigned(dx)
        dut.sh0_i.value = sh0
        dut.neg0_i.value = neg0
        dut.sh1_i.value = sh1
        dut.neg1_i.value = neg1
        await Timer(1, unit="ns")

        got = to_signed(dut.y_o.value.to_unsigned())
        assert got == expected, (
            f"dx={dx} sh0={sh0} neg0={neg0} sh1={sh1} neg1={neg1}: "
            f"got {got}, expected {expected}"
        )


@cocotb.test()
async def test_matches_reference(dut):
    """Sweep dx values against reference."""
    random.seed(42)

    for _ in range(50):
        dx = random.randint(-2048, 2047)
        sh0 = random.randint(0, 15)
        sh1 = random.randint(0, 15)
        neg0 = random.randint(0, 1)
        neg1 = random.randint(0, 1)

        dut.dx_i.value = to_unsigned(dx)
        dut.sh0_i.value = sh0
        dut.neg0_i.value = neg0
        dut.sh1_i.value = sh1
        dut.neg1_i.value = neg1
        await Timer(1, unit="ns")

        got = to_signed(dut.y_o.value.to_unsigned())
        expected = ref_slope_shift(dx, sh0, neg0, sh1, neg1)
        assert got == expected, (
            f"dx={dx} sh0={sh0} neg0={neg0} sh1={sh1} neg1={neg1}: "
            f"got {got}, expected {expected}"
        )
