// FP8 (E5M2) MAC with FP12 (E6M5) accumulator + stochastic rounding https://arxiv.org/abs/2404.14010
// Lazy SR: round after norm by adding random bits to truncated portion
module mac_sr #(
    parameter int R = 9  // random bits for SR
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        clear_i,
    input  logic        valid_i,
    input  logic  [7:0] a_i,      // FP8 E5M2
    input  logic  [7:0] b_i,      // FP8 E5M2
    input  logic [R-1:0] rand_i,  // random bits from LFSR/PRNG
    output logic [11:0] acc_o,    // FP12 E6M5
    output logic        valid_o
);

  // unpack FP8 E5M2
  wire        a_s = a_i[7], b_s = b_i[7];
  wire  [4:0] a_e = a_i[6:2], b_e = b_i[6:2];
  wire  [2:0] a_m = {1'b1, a_i[1:0]};
  wire  [2:0] b_m = {1'b1, b_i[1:0]};

  // multiply
  wire        p_s = a_s ^ b_s;
  wire  [5:0] p_m_raw = a_m * b_m;
  wire        p_ovf = p_m_raw[5];
  wire  [5:0] p_m = p_ovf ? p_m_raw[5:0] : {p_m_raw[4:0], 1'b0};
  wire  [6:0] p_e = {1'b0, a_e} + {1'b0, b_e} + 7'd1 + {6'b0, p_ovf};

  // unpack acc
  wire        c_s = acc_o[11];
  wire  [6:0] c_e = {1'b0, acc_o[10:5]};
  wire        c_zero = ~|acc_o[10:0];
  wire  [5:0] c_m = c_zero ? 6'b0 : {1'b1, acc_o[4:0]};

  // align to larger exponent
  wire        p_ge = (p_e >= c_e);
  wire  [6:0] ediff = p_ge ? (p_e - c_e) : (c_e - p_e);
  wire  [3:0] shamt = |ediff[6:4] ? 4'd6 : ediff[3:0];

  wire  [5:0] hi_m = p_ge ? p_m : c_m;
  wire        hi_s = p_ge ? p_s : c_s;
  wire  [6:0] hi_e = p_ge ? p_e : c_e;
  wire  [5:0] lo_m = (p_ge ? c_m : p_m) >> shamt;

  // add/subtract
  wire        eff_sub = hi_s ^ (p_ge ? c_s : p_s);
  wire  [6:0] sum_raw = eff_sub ? ({1'b0, hi_m} - {1'b0, lo_m})
                                : ({1'b0, hi_m} + {1'b0, lo_m});

  // normalize
  logic [3:0] lzc;
  always_comb begin
    lzc = 4'd6;
    for (int i = 0; i <= 5; i++) begin
      if (sum_raw[i]) lzc = 4'(5 - i);
    end
  end

  wire        r_ovf = sum_raw[6];
  wire  [5:0] r_m = r_ovf ? sum_raw[6:1] : (sum_raw[5:0] << lzc);
  wire  [6:0] r_e = r_ovf ? (hi_e + 7'd1) : (hi_e - {3'b0, lzc});

  // stochastic rounding
  wire [R:0] sr_add = {r_m[0], rand_i} + {{R{1'b0}}, 1'b1};
  wire       sr_carry = sr_add[R];
  wire [5:0] r_m_sr = sr_carry ? (r_m + 6'd1) : r_m;

  // pack result
  wire [11:0] result = c_zero ? {p_s, p_e[5:0], p_m[4:0]}
                               : {hi_s, r_e[5:0], r_m_sr[4:0]};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_o   <= 12'b0;
      valid_o <= 1'b0;
    end else if (clear_i) begin
      acc_o   <= 12'b0;
      valid_o <= 1'b0;
    end else if (valid_i) begin
      acc_o   <= result;
      valid_o <= 1'b1;
    end else begin
      valid_o <= 1'b0;
    end
  end

endmodule
