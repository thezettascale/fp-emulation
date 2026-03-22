import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

FRAC_W = 8
SCALE = 1 << FRAC_W


def to_fixed(v):
    """Double to Q8.8 signed fixed-point (16-bit)."""
    raw = int(v * SCALE)
    raw = max(-32768, min(32767, raw)) # clamp to signed 16-bit range
    return raw & 0xFFFF # unsigned repr for cocotb


def to_fixed_signed(v):
    """Double to Q8.8 as signed int (for reference math)."""
    raw = int(v * SCALE)
    return max(-32768, min(32767, raw))


def from_acc(v):
    """Q16.16 acc to double."""
    if v >= (1 << 31):
        v -= 1 << 32

    return v / (SCALE * SCALE)


async def reset(dut):
    dut.rst_ni.value = 0
    dut.clear_i.value = 0
    dut.valid_i.value = 0
    dut.a_i.value = 0
    dut.b_i.value = 0
    await ClockCycles(dut.clk_i, 2)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def mac_once(dut, a, b):
    dut.a_i.value = to_fixed(a)
    dut.b_i.value = to_fixed(b)
    dut.valid_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.valid_i.value = 0


async def clear(dut):
    dut.clear_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.clear_i.value = 0


@cocotb.test()
async def test_accumulate(dut):
    """1.5*2.25 + (-1.0)*4.0 = 3.375 - 4.0 = -0.625"""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    await mac_once(dut, 1.5, 2.25)
    await mac_once(dut, -1.0, 4.0)
    await RisingEdge(dut.clk_i)

    result = from_acc(dut.acc_o.value.to_unsigned())
    assert abs(result - (-0.625)) < 1e-6, f"expected -0.625, got {result}"


@cocotb.test()
async def test_clear_and_dot_product(dut):
    """dot([1.5, -0.5, 2.0], [3.0, 4.0, -1.25]) = 0.0"""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    a_vec = [1.5, -0.5, 2.0]
    b_vec = [3.0, 4.0, -1.25]

    for a, b in zip(a_vec, b_vec):
        await mac_once(dut, a, b)

    await RisingEdge(dut.clk_i)

    result = from_acc(dut.acc_o.value.to_unsigned())
    assert abs(result) < 1e-6, f"expected 0.0, got {result}"


@cocotb.test()
async def test_random_accumulation(dut):
    """10 random MAC ops against reference."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    random.seed(42)
    ref_acc = 0.0

    for _ in range(10):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        ref_acc += to_fixed_signed(a) * to_fixed_signed(b) / (SCALE * SCALE)
        await mac_once(dut, a, b)

    await RisingEdge(dut.clk_i)

    result = from_acc(dut.acc_o.value.to_unsigned())
    assert abs(result - ref_acc) < 1e-6, f"expected {ref_acc}, got {result}"
