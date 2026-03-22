# fp-emulation

High-precision floating-point from low-precision integer ops. Ozaki scheme matmul + ML-PLAC activations.

## Benchmarks

Emulated high-precision forward pass on a small PDE solver (Poisson/Burgers MLP). Ozaki II matmul + ML-PLAC activations, all low-precision integer ops. Measure PDE residual vs precision level, compare against native FP32/FP64.

## Motivation

- Awesome PINNs/SciML people have asked me for higher precision, since higher-order PDE residuals (2nd/3rd derivatives) accumulate rounding error through the chain rule.
- FP32's 24-bit mantissa is not enough. FP64 helps but is slow (1/32 throughput on consumer NVIDIA GPUs).
- Ozaki schemes emulate FP64+ using fast INT8/FP8 tensor cores. Precision without the silicon tax.

### Why this is so awkward :P

- Ozaki helps HPC/scientific computing (climate sims, quantum chemistry). Quantized inference comes from LLMs. The requirements for these groups don't really overlap.
- HPC people don't work in matmul -> activations, Ozaki is just for matmul. People serving inference don't care about FP64, low precision INT8 matmul/act is enough.
- PINNs benefit from both! High precision matmul/act emulated in cheaper silicon for fast, stable convergence

### Autodiff

- Pure matmul + act is mostly useful for inference, but the people want autodiff/training
- [DT-PINNs](https://arxiv.org/abs/2205.09332) address the autodiff bottleneck by replacing with numerical derivatives (RBF-FD), making matmul precision the bottleneck. Their fp64 DT-PINNs already outperforms fp32 vanilla PINNs in wall-clock (higher quality grads -> faster convergence).

## Ref.

- [TwoProduct/TwoSum for exact sum and dotprod](https://doi.org/10.1137/030601818)
- [Ozaki II Scheme for matmul and dotprod](https://arxiv.org/abs/2504.08009)
- [ML-PLAC for bit-shift only piecewise linear](https://www.mdpi.com/2076-3417/12/20/10616)
- [DT-PINNs: meshless discretizations replacing autodiff](https://arxiv.org/abs/2205.09332)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf)
