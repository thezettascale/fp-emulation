#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void mod_reduce(int32_t *data, const int32_t *moduli, int elems_per_batch, int n_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems_per_batch * n_batch)
        return;

    int32_t m = moduli[idx / elems_per_batch];
    int32_t v = data[idx] % m;
    if (v < 0)
        v += m;

    data[idx] = v;
}

torch::Tensor batched_int8_gemm_mod(torch::Tensor a, torch::Tensor b, torch::Tensor moduli) {
    int batch = a.size(0);
    int M = a.size(1);
    int K = a.size(2);
    int N = b.size(2);

    auto c = torch::zeros({batch, M, N}, torch::dtype(torch::kInt32).device(a.device()));

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    // row-major A(M,K)@B(K,N)=C(M,N) via col-major: B^T(N,K)@A^T(K,M)=C^T(N,M)
    cublasLtMatrixLayout_t layoutB, layoutA, layoutC;
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, N, K, N);
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, K, M, K);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, N, M, N);

    int64_t strideB = (int64_t)K * N, strideA = (int64_t)M * K, strideC = (int64_t)M * N;
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                     sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB,
                                     sizeof(strideB));
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                     sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA,
                                     sizeof(strideA));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                     sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC,
                                     sizeof(strideC));

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_size = 4 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size,
                                         sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heur;
    int n_algo = 0;
    auto hstatus = cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutB, layoutA, layoutC,
                                                  layoutC, pref, 1, &heur, &n_algo);
    TORCH_CHECK(hstatus == CUBLAS_STATUS_SUCCESS && n_algo > 0,
                "cublasLt: no int8 algorithm found, status=", (int)hstatus);

    void *workspace = nullptr;
    if (heur.workspaceSize > 0)
        cudaMalloc(&workspace, heur.workspaceSize);

    int32_t alpha = 1, beta = 0;
    auto status = cublasLtMatmul(handle, matmulDesc, &alpha, b.data_ptr<int8_t>(), layoutB,
                                 a.data_ptr<int8_t>(), layoutA, &beta, c.data_ptr<int32_t>(),
                                 layoutC, c.data_ptr<int32_t>(), layoutC, &heur.algo, workspace,
                                 heur.workspaceSize, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed, status=", (int)status);

    if (workspace)
        cudaFree(workspace);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtDestroy(handle);

    int total = batch * M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mod_reduce<<<blocks, threads>>>(c.data_ptr<int32_t>(), moduli.data_ptr<int32_t>(), M * N,
                                    batch);

    return c;
}

__global__ void crt_kernel(const int *__restrict__ residues, const double *__restrict__ wh,
                           const double *__restrict__ wm, const double *__restrict__ wl,
                           double M_hi, double M_lo, double inv_M, const int *__restrict__ row_exp,
                           const int *__restrict__ col_exp, int two_bits, double *__restrict__ out,
                           int n_elems, int cols, int n_mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems)
        return;

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

torch::Tensor crt_reconstruct(torch::Tensor residues, torch::Tensor wh, torch::Tensor wm,
                              torch::Tensor wl, double M_hi, double M_lo, double inv_M,
                              torch::Tensor row_exp, torch::Tensor col_exp, int two_bits, int rows,
                              int cols, int n_mod) {
    int n_elems = rows * cols;
    auto out = torch::empty({rows, cols}, torch::dtype(torch::kFloat64).device(residues.device()));

    int threads = 256;
    int blocks = (n_elems + threads - 1) / threads;

    crt_kernel<<<blocks, threads>>>(residues.data_ptr<int>(), wh.data_ptr<double>(),
                                    wm.data_ptr<double>(), wl.data_ptr<double>(), M_hi, M_lo, inv_M,
                                    row_exp.data_ptr<int>(), col_exp.data_ptr<int>(), two_bits,
                                    out.data_ptr<double>(), n_elems, cols, n_mod);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crt_reconstruct", &crt_reconstruct);
    m.def("batched_int8_gemm_mod", &batched_int8_gemm_mod);
}
