import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

SCALE = 1 << 8  # Q8.8


def to_fixed(v):
    raw = int(v * SCALE)
    raw = max(-32768, min(32767, raw))
    return raw & 0xFFFF


def from_acc(v):
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


@cocotb.test()
async def test_acc(dut):
    """1.5*2.25 + (-1.0)*4.0 = 3.375 - 4.0 = -0.625"""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    await mac_once(dut, 1.5, 2.25)
    await mac_once(dut, -1.0, 4.0)
    await RisingEdge(dut.clk_i)

    result = from_acc(dut.acc_o.value.to_unsigned())
    assert abs(result - (-0.625)) < 1e-6, f"expected -0.625, got {result}"


@cocotb.test()
async def test_clear_and_dot(dut):
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
