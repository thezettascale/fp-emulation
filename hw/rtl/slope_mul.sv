// slope * dx via multiplier
module slope_mul #(
    parameter int W = 16
) (
    input  logic signed [W-1:0] slope_i,
    input  logic signed [W-1:0] dx_i,
    output logic signed [W-1:0] y_o
);
  wire signed [2*W-1:0] prod = slope_i * dx_i;
  assign y_o = prod[2*W-2:W-1];
endmodule
