import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


def float_to_fp8_e5m2(f):
    """Convert float to FP8 E5M2 (IEEE-like, no denorms)."""
    bits = struct.pack(">f", f)
    val = struct.unpack(">I", bits)[0]
    sign = (val >> 31) & 1
    exp = ((val >> 23) & 0xFF) - 127 + 15  # rebias to 5-bit
    exp = max(0, min(31, exp))
    mant = (val >> 21) & 0x3  # top 2 mantissa bits
    return (sign << 7) | (exp << 2) | mant


def fp8_e5m2_to_float(bits):
    """Convert FP8 E5M2 back to float."""
    sign = (bits >> 7) & 1
    exp = (bits >> 2) & 0x1F
    mant = bits & 0x3
    if exp == 0:
        return 0.0
    val = (1.0 + mant / 4.0) * (2.0 ** (exp - 15))
    return -val if sign else val


def fp12_e6m5_to_float(bits):
    """Convert FP12 E6M5 to float."""
    sign = (bits >> 11) & 1
    exp = (bits >> 5) & 0x3F
    mant = bits & 0x1F
    if exp == 0 and mant == 0:
        return 0.0
    val = (1.0 + mant / 32.0) * (2.0 ** (exp - 31))
    return -val if sign else val


async def reset(dut):
    dut.rst_ni.value = 0
    dut.clear_i.value = 0
    dut.valid_i.value = 0
    dut.a_i.value = 0
    dut.b_i.value = 0
    dut.rand_i.value = 0
    await ClockCycles(dut.clk_i, 2)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


@cocotb.test()
async def test_basic_mac(dut):
    """Accumulate 1.5 * 2.0 = 3.0, check result is reasonable."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    dut.a_i.value = float_to_fp8_e5m2(1.5)
    dut.b_i.value = float_to_fp8_e5m2(2.0)
    dut.rand_i.value = 0
    dut.valid_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.valid_i.value = 0
    await RisingEdge(dut.clk_i)

    result = fp12_e6m5_to_float(dut.acc_o.value.to_unsigned())
    err = abs(result - 3.0)
    dut._log.info(f"1.5 * 2.0: got {result:.4f}, expected 3.0, err={err:.4f}")
    assert err < 0.5, f"too far off: got {result}"


@cocotb.test()
async def test_clear(dut):
    """Accumulate, clear, check zero."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    dut.a_i.value = float_to_fp8_e5m2(1.0)
    dut.b_i.value = float_to_fp8_e5m2(1.0)
    dut.rand_i.value = 0
    dut.valid_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.valid_i.value = 0

    dut.clear_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.clear_i.value = 0
    await RisingEdge(dut.clk_i)

    result = dut.acc_o.value.to_unsigned()
    assert result == 0, f"expected 0 after clear, got {result}"
