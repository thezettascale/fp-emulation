import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import math

SCALE = 256  # Q8.8


def to_q88(v):
    raw = int(round(v * SCALE))
    return max(-32768, min(32767, raw)) & 0xFFFF


def from_q88(v):
    if v >= 0x8000:
        v -= 0x10000
    return v / SCALE


async def reset(dut):
    dut.rst_ni.value = 0
    dut.valid_i.value = 0
    dut.x_i.value = 0
    await ClockCycles(dut.clk_i, 2)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


@cocotb.test()
async def test_tanh_sweep(dut):
    """Sweep x in [-4, 4], check ML-PLAC output against math.tanh."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    test_points = [i * 0.5 for i in range(-8, 9)]  # -4.0 to 4.0
    max_err = 0.0

    for x in test_points:
        dut.x_i.value = to_q88(x)
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.valid_i.value = 0
        await RisingEdge(dut.clk_i)

        got = from_q88(dut.y_o.value.to_unsigned())
        expected = math.tanh(x)
        err = abs(got - expected)
        max_err = max(max_err, err)

        # 10-seg PWL approx, ~0.05 typical error
        assert err < 0.15, f"x={x}: got {got:.4f}, expected {expected:.4f}"

    dut._log.info(f"max abs error over sweep: {max_err:.4f}")
