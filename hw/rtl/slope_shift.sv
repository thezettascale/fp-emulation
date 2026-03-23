// slope * dx via shift-add (ML-PLAC)
module slope_shift #(
    parameter int W = 16
) (
    input  logic signed [W-1:0] dx_i,
    input  logic        [  4:0] sh0_i,
    input  logic                neg0_i,
    input  logic        [  4:0] sh1_i,
    input  logic                neg1_i,
    output logic signed [W-1:0] y_o
);
  logic signed [W-1:0] t0, t1;

  always_comb begin
    t0 = dx_i >>> sh0_i;
    if (neg0_i) t0 = -t0;
    t1 = dx_i >>> sh1_i;
    if (neg1_i) t1 = -t1;
  end

  assign y_o = t0 + t1;
endmodule
