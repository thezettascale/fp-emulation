SIM           ?= verilator
TOPLEVEL_LANG  = verilog
TEST          ?= mac

ifeq ($(TEST),mac)
  VERILOG_SOURCES     = ../rtl/mac.sv
  TOPLEVEL            = mac
  COCOTB_TEST_MODULES = tb_mac
else ifeq ($(TEST),pwl)
  VERILOG_SOURCES     = ../rtl/pwl_mlplac.sv
  TOPLEVEL            = pwl_mlplac
  COCOTB_TEST_MODULES = tb_pwl
else ifeq ($(TEST),slope)
  VERILOG_SOURCES     = ../rtl/slope_shift.sv
  TOPLEVEL            = slope_shift
  COCOTB_TEST_MODULES = tb_slope
else ifeq ($(TEST),mac_sr)
  VERILOG_SOURCES     = ../rtl/mac_sr.sv
  TOPLEVEL            = mac_sr
  COCOTB_TEST_MODULES = tb_mac_sr
endif

SIM_BUILD     = sim_build/$(TEST)
ifdef TRACE
EXTRA_ARGS   += --trace
endif

.PHONY: all run
all:
	$(MAKE) run TEST=mac
	$(MAKE) run TEST=pwl
	$(MAKE) run TEST=slope
	$(MAKE) run TEST=mac_sr

run: sim

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	rm -rf __pycache__ results.xml sim_build *.vcd
