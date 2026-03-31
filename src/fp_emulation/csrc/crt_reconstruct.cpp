#include <cmath>
#include <cstdint>
#include <torch/extension.h>

/* CRT reconstruction. Triple-float weights + Kahan summation. */
static void crt_reconstruct_impl(const int32_t *residues, const double *wh, const double *wm,
                                  const double *wl, double M_hi, double M_lo, double inv_M,
                                  const int32_t *row_exp, const int32_t *col_exp, int two_bits,
                                  double *out, int rows, int cols, int n_mod) {
    int n_elems = rows * cols;

#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_elems; idx++) {
        double acc_hi = 0.0, acc_lo = 0.0;

        for (int k = 0; k < n_mod; k++) {
            double r = (double)residues[k * n_elems + idx];

            double prod = r * wh[k];
            double e = fma(r, wh[k], -prod);

            double s = acc_hi + prod;
            double v = s - acc_hi;
            double err = (acc_hi - (s - v)) + (prod - v);
            acc_hi = s;
            acc_lo = acc_lo + fma(r, wm[k], e) + r * wl[k] + err;
        }

        double q = rint((acc_hi + acc_lo) * inv_M);
        double pm = q * M_hi;
        double em = fma(q, M_hi, -pm);
        double result = (acc_hi - pm) + (acc_lo - em - q * M_lo);

        int i = idx / cols;
        int j = idx % cols;
        out[idx] = ldexp(result, row_exp[i] + col_exp[j] - two_bits);
    }
}

torch::Tensor crt_reconstruct(torch::Tensor residues, torch::Tensor wh, torch::Tensor wm,
                               torch::Tensor wl, double M_hi, double M_lo, double inv_M,
                               torch::Tensor row_exp, torch::Tensor col_exp, int two_bits, int rows,
                               int cols, int n_mod) {
    auto out = torch::empty({rows, cols}, torch::kFloat64);
    crt_reconstruct_impl(residues.data_ptr<int32_t>(), wh.data_ptr<double>(), wm.data_ptr<double>(),
                         wl.data_ptr<double>(), M_hi, M_lo, inv_M, row_exp.data_ptr<int32_t>(),
                         col_exp.data_ptr<int32_t>(), two_bits, out.data_ptr<double>(), rows, cols,
                         n_mod);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("crt_reconstruct", &crt_reconstruct); }
