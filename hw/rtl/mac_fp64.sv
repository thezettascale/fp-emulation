// FP64 (no denormal/NaN/Inf)
module mac_fp64 (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        clear_i,
    input  logic        valid_i,
    input  logic [63:0] a_i,
    input  logic [63:0] b_i,
    output logic [63:0] acc_o,
    output logic        valid_o
);

  // multiply
  wire          a_s = a_i[63], b_s = b_i[63];
  wire  [ 10:0] a_e = a_i[62:52], b_e = b_i[62:52];
  wire  [ 52:0] a_m = {1'b1, a_i[51:0]};
  wire  [ 52:0] b_m = {1'b1, b_i[51:0]};

  wire          p_s = a_s ^ b_s;
  wire  [105:0] p_m_raw = a_m * b_m;
  wire          p_ovf = p_m_raw[105];
  wire  [ 52:0] p_m = p_ovf ? p_m_raw[105:53] : p_m_raw[104:52];
  wire  [ 11:0] p_e = {1'b0, a_e} + {1'b0, b_e} - 12'd1023 + {11'b0, p_ovf};

  // unpack acc
  wire          c_s = acc_o[63];
  wire  [ 11:0] c_e = {1'b0, acc_o[62:52]};
  wire          c_zero = ~|acc_o[62:0];
  wire  [ 52:0] c_m = c_zero ? 53'b0 : {1'b1, acc_o[51:0]};

  // align to larger exponent
  wire          p_ge = (p_e >= c_e);
  wire  [ 11:0] ediff = p_ge ? (p_e - c_e) : (c_e - p_e);
  wire  [  5:0] shamt = |ediff[11:6] ? 6'd53 : ediff[5:0];

  wire  [ 52:0] hi_m = p_ge ? p_m : c_m;
  wire          hi_s = p_ge ? p_s : c_s;
  wire  [ 11:0] hi_e = p_ge ? p_e : c_e;
  wire  [ 52:0] lo_m = (p_ge ? c_m : p_m) >> shamt;

  // add/subtract
  wire          eff_sub = hi_s ^ (p_ge ? c_s : p_s);
  wire  [ 53:0] sum_raw = eff_sub ? ({1'b0, hi_m} - {1'b0, lo_m}) : ({1'b0, hi_m} + {1'b0, lo_m});

  // normalize
  logic [  5:0] lzc;
  always_comb begin
    lzc = 6'd53;
    for (int i = 0; i <= 52; i++) begin
      if (sum_raw[i]) lzc = 6'(52 - i);
    end
  end

  wire        r_ovf = sum_raw[53];
  wire [52:0] r_m = r_ovf ? sum_raw[53:1] : (sum_raw[52:0] << lzc);
  wire [11:0] r_e = r_ovf ? (hi_e + 12'd1) : (hi_e - {6'b0, lzc});

  // register
  wire [63:0] result = c_zero ? {p_s, p_e[10:0], p_m[51:0]} : {hi_s, r_e[10:0], r_m[51:0]};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_o   <= 64'b0;
      valid_o <= 1'b0;
    end else if (clear_i) begin
      acc_o   <= 64'b0;
      valid_o <= 1'b0;
    end else if (valid_i) begin
      acc_o   <= result;
      valid_o <= 1'b1;
    end else begin
      valid_o <= 1'b0;
    end
  end

endmodule
