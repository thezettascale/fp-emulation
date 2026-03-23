# fp-emulation

FP64-exact PDE solving on 16x smaller silicon, using only INT8 integer arithmetic.

## Why

PDE solvers need FP64. Higher-order derivatives amplify rounding error through the chain rule. FP32 breaks.

But FP64 is expensive: 1/32 throughput on consumer GPUs, and even datacenter GPUs dedicate most die area to FP32/FP16/INT8. FP64 units are large and power-hungry.

**What if you could get FP64-exact results from INT8 ops?** This repo shows how and works today (on any GPU with INT8 tensor cores), but the real point is the hardware: INT8 fixed-point silicon is 16x smaller than FP64 for the same precision.

<figure>
<img src="figures/synth.png" alt="Yosys comparison">
<figcaption>
Yosys gate-level cell counts (<code>hw/synth/</code>).
<b>Left:</b> INT8 vs FP64 MAC. Same precision, 16x less silicon.
<b>Right:</b> ML-PLAC bit-shift slopes vs multiplier. Gap widens with data width.
</figcaption>
</figure>

## How it works

- **Matmul**: [Ozaki scheme II](https://arxiv.org/abs/2504.08009) splits floats into integers, does L modular matmuls, reconstructs via Chinese Remainder Theorem. Exact to FP64.
- **Activations**: [ML-PLAC](https://www.mdpi.com/2076-3417/12/20/10616) approximates nonlinear functions with piecewise-linear segments using only bit-shifts and adds. No multiplier needed. O(N) area vs multiplier O(N²), perfect replacements for arbitrary precision piecewise linear activations.

<figure>
<img src="figures/accuracy.png" alt="Accuracy">
<figcaption>INT8 Ozaki max absolute error stays near machine epsilon. Relative error grows for near-zero entries (small denominator).</figcaption>
</figure>

## Hardware target

On current GPUs, Ozaki is slower than native FP64 at small matrix sizes due to L kernel launches and Python overhead. But INT8 tensor cores are ~500x faster than FP64 units (T4: 130 TOPS vs 0.25 TFLOPS), so Ozaki wins at large n where compute (not kernel launch) dominates.

<figure>
<img src="figures/benchmark.png" alt="Benchmark">
<figcaption>Matmul latency on T4 GPU. Ozaki overhead is fixed (L kernel launches + CRT), so the ratio shrinks as n grows. At large n, INT8 throughput dominates and Ozaki beats native FP64.</figcaption>
</figure>

The real target is dedicated fixed-point silicon:

- INT8 MAC is 16x smaller -> same die area, 16x more compute
- L matmuls pipeline in hardware, no kernel launches
- ML-PLAC slope cores use only bit-shifts. 16x smaller than multipliers at 64-bit

RTL in `hw/rtl/`, testbenches in `hw/sim/`, synthesis in `hw/synth/`.

## Application: DT-PINNs

[DT-PINNs](https://arxiv.org/abs/2205.09332) replace autodiff with numerical differentiation matrices (the paper uses RBF-FD). We use [Chebyshev spectral](https://people.maths.ox.ac.uk/trefethen/spectral.html) differentiation instead, (dense matmuls suited to INT8 Ozaki acceleration).

<figure>
<img src="figures/dt_pinn_loss.png" alt="DT-PINN loss">
<figcaption>Burgers' equation training loss. DT-PINN replaces autograd with matmul derivatives. INT8 Ozaki and FP64 curves overlap, (identical precision).</figcaption>
</figure>

<figure>
<img src="figures/dt_pinn_solution.png" alt="DT-PINN solution">
<figcaption>Burgers' solution at t=1. All three methods converge to the same shock profile — INT8 Ozaki is indistinguishable from native FP64.</figcaption>
</figure>

## Try:

```python
from fp_emulation import ozaki2_int8_matmul, convert

C = ozaki2_int8_matmul(A, B)   # FP64-exact via INT8 tensor cores
model = convert(model)         # swap all nn.Linear layers
```

[`notebooks/04_demo.ipynb`](notebooks/04_demo.ipynb) — accuracy, benchmarks, Burgers PINN
[`notebooks/05_dt_pinn.ipynb`](notebooks/05_dt_pinn.ipynb) — DT-PINN with Ozaki INT8

Run free on Colab with T4 GPU (fast INT8, slow native FP64):
1. Fork this repo
2. Open in Colab (File -> Open notebook -> GitHub -> your fork)
3. Set runtime to T4

## References

- [Ozaki II scheme for matmul](https://arxiv.org/abs/2504.08009)
- [Ozaki error-free transformations](https://dl.acm.org/doi/epdf/10.1145/3731599.3767539)
- [TwoProduct/TwoSum](https://doi.org/10.1137/030601818)
- [ML-PLAC bit-shift piecewise linear](https://www.mdpi.com/2076-3417/12/20/10616)
- [DT-PINNs](https://arxiv.org/abs/2205.09332)
- [What Every CS Should Know About FP Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
