// PWL activation (with multiplier), tanh 10-seg Q8.8
module pwl (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                valid_i,
    input  logic signed [ 15:0] x_i,
    input  logic signed [159:0] slopes_i,  // 10 x 16-bit packed
    output logic signed [ 15:0] y_o,
    output logic                valid_o
);

  // segment finder
  logic [3:0] seg;
  always_comb begin
    if      (x_i >= 16'sd780)  seg = 4'd9;
    else if (x_i >= 16'sd434)  seg = 4'd8;
    else if (x_i >= 16'sd276)  seg = 4'd7;
    else if (x_i >= 16'sd156)  seg = 4'd6;
    else if (x_i >= 16'sd47)   seg = 4'd5;
    else if (x_i >= -16'sd124) seg = 4'd4;
    else if (x_i >= -16'sd211) seg = 4'd3;
    else if (x_i >= -16'sd317) seg = 4'd2;
    else if (x_i >= -16'sd483) seg = 4'd1;
    else                        seg = 4'd0;
  end

  // breakpoint + offset ROM
  logic signed [15:0] bp, off;
  always_comb begin
    case (seg)
      4'd0: begin bp = -16'sd1024; off = -16'sd238; end
      4'd1: begin bp = -16'sd483;  off = -16'sd169; end
      4'd2: begin bp = -16'sd317;  off = -16'sd97;  end
      4'd3: begin bp = -16'sd211;  off = -16'sd40;  end
      4'd4: begin bp = -16'sd124;  off =  16'sd0;   end
      4'd5: begin bp =  16'sd47;   off =  16'sd5;   end
      4'd6: begin bp =  16'sd156;  off =  16'sd59;  end
      4'd7: begin bp =  16'sd276;  off =  16'sd140; end
      4'd8: begin bp =  16'sd434;  off =  16'sd221; end
      4'd9: begin bp =  16'sd780;  off =  16'sd252; end
      default: begin bp = '0; off = '0; end
    endcase
  end

  wire signed [15:0] sl = slopes_i[seg*16+:16];
  wire signed [15:0] dx = x_i - bp;
  wire signed [31:0] prod = sl * dx;
  wire signed [15:0] y = prod[23:8] + off;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o     <= '0;
      valid_o <= 1'b0;
    end else if (valid_i) begin
      y_o     <= y;
      valid_o <= 1'b1;
    end else begin
      valid_o <= 1'b0;
    end
  end

endmodule
