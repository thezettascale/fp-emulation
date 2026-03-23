# fp-emulation

FP64-exact matmul from INT8 ops. Drop-in for PyTorch.

```python
from fp_emulation import ozaki2_int8_matmul, convert

C = ozaki2_int8_matmul(A, B)   # FP64-exact via INT8 tensor cores
model = convert(model)         # swap all nn.Linear layers
```

## Why

FP32 loses precision in PDE residuals (chain rule through 2nd/3rd derivatives). FP64 fixes this but runs at 1/32 throughput on consumer GPUs (non-SOTA).

This gives you FP64 precision using INT8 tensor cores for older GPUs ([Ozaki scheme II](https://arxiv.org/abs/2504.08009)): split into integers, L modular matmuls via small primes, CRT reconstruct. Exact to FP64.

## Hardware case

On SOTA GPUs, Ozaki is slower than native FP64 (L kernel launches, overhead). The real target is dedicated fixed-point silicon:

- L matmuls pipeline in hardware, no kernel launch overhead
- CRT is a fixed datapath, not a general-purpose kernel

<figure>
<img src="figures/synth.png" alt="Synthesis comparison">
<figcaption>
<b>Left:</b> INT8 MAC is 16x smaller than FP64 MAC. Same die area fits 16x more cores, 16x more throughput.
<b>Right:</b> <a href="https://www.mdpi.com/2076-3417/12/20/10616">ML-PLAC</a> replaces multipliers with VERY CHEAP bit-shifts for piecewise-linear activations. Multiplier grows O(N²), shift-add grows O(N). 16x smaller at 64-bit!
Cell counts from Yosys open-source synthesis (<code>hw/synth/</code>).
</figcaption>
</figure>

## Demo

[`notebooks/04_demo.ipynb`](notebooks/04_demo.ipynb)

<figure>
<img src="figures/accuracy.png" alt="Accuracy">
<figcaption>Ozaki INT8 matmul matches native FP64 to machine epsilon across all matrix sizes.</figcaption>
</figure>

<figure>
<img src="figures/burgers_loss.png" alt="Burgers training loss">
<img src="figures/burgers_solution.png" alt="Burgers solution">
<figcaption>Burgers' equation PINN (u_t + u u_x = v u_xx, steep shock). FP32 struggles near the discontinuity. INT8 Ozaki tracks FP64.</figcaption>
</figure>

Run on Colab with T4 GPU (fast INT8, crippled FP64):
1. Fork this repo
2. Open notebook in Colab (File -> Open notebook -> GitHub -> your fork)
3. Set runtime to T4 (Runtime -> Change runtime type -> T4)

## References

- [Ozaki II scheme for matmul](https://arxiv.org/abs/2504.08009)
- [Ozaki error-free transformations](https://dl.acm.org/doi/epdf/10.1145/3731599.3767539)
- [TwoProduct/TwoSum](https://doi.org/10.1137/030601818)
- [ML-PLAC bit-shift piecewise linear](https://www.mdpi.com/2076-3417/12/20/10616)
- [DT-PINNs](https://arxiv.org/abs/2205.09332)
- [What Every CS Should Know About FP Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
