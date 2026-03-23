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
endif

SIM_BUILD     = sim_build/$(TEST)
ifdef TRACE
EXTRA_ARGS   += --trace
endif

.PHONY: all
all:
	$(MAKE) TEST=mac
	$(MAKE) TEST=pwl
	$(MAKE) TEST=slope

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	rm -rf __pycache__ results.xml sim_build *.vcd
