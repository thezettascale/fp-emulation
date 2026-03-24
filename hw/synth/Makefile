RTL := ../rtl

.PHONY: all clean

all: macs activations slopes

macs:
	@echo "MACs"
	@printf "  int8:   %6s cells\n" "$$($(call synth_p,mac,QI 4 QF 4))"
	@printf "  fp8+sr: %6s cells\n" "$$($(call synth,mac_sr))"
	@printf "  fp64:   %6s cells\n" "$$($(call synth,mac_fp64))"
	@echo

activations:
	@echo "Activations (tanh, 10-seg PWL)"
	@printf "  with multiplier:  %6s cells\n" "$$($(call synth,pwl))"
	@printf "  ML-PLAC (shifts): %6s cells\n" "$$($(call synth,pwl_mlplac))"
	@echo

slopes:
	@echo "PWL slope: multiplier vs shift-add"
	@printf "  %5s  %10s  %10s\n" "width" "multiplier" "shift-add"
	@printf "  %5d  %10s  %10s\n" 16 "$$($(call synth_p,slope_mul,W 16))" "$$($(call synth_p,slope_shift,W 16))"
	@printf "  %5d  %10s  %10s\n" 32 "$$($(call synth_p,slope_mul,W 32))" "$$($(call synth_p,slope_shift,W 32))"
	@printf "  %5d  %10s  %10s\n" 64 "$$($(call synth_p,slope_mul,W 64))" "$$($(call synth_p,slope_shift,W 64))"

define synth
yosys -p "read_verilog -sv $(RTL)/$(1).sv; synth -top $(1); stat" \
	2>/dev/null | grep -oP '\d+(?=\s+cells$$)' | tail -1
endef

define synth_p
yosys -p "read_verilog -sv $(RTL)/$(1).sv; chparam -set $(2) $(1); synth -top $(1); stat" \
	2>/dev/null | grep -oP '\d+(?=\s+cells$$)' | tail -1
endef


clean:
	rm -rf *.log
