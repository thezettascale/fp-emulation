SIM        ?= verilator
TOPLEVEL_LANG = verilog

VERILOG_SOURCES = ../rtl/mac.sv
TOPLEVEL = mac
COCOTB_TEST_MODULES = tb_mac

EXTRA_ARGS += --trace

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	rm -rf __pycache__ results.xml sim_build *.vcd
