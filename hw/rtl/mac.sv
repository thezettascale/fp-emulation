// Q(QI.QF) signed format
module mac #(
    parameter int QI = 8,
    parameter int QF = 8
) (
    input logic clk_i,
    input logic rst_ni,
    input logic clear_i,
    input logic valid_i,

    input logic signed [QI+QF-1:0] a_i,
    input logic signed [QI+QF-1:0] b_i,

    output logic signed [2*(QI+QF)-1:0] acc_o,
    output logic valid_o
);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_o   <= '0;
      valid_o <= 1'b0;
    end else if (clear_i) begin
      acc_o   <= '0;
      valid_o <= 1'b0;
    end else if (valid_i) begin
      acc_o   <= acc_o + (a_i * b_i);
      valid_o <= 1'b1;
    end else begin
      valid_o <= 1'b0;
    end
  end
endmodule
