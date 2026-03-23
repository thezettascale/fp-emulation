# fp-emulation

FP64 precision from INT8 integer ops. Ozaki scheme II matmul + ML-PLAC activations.

```python
from fp_emulation import ozaki2_int8_matmul, convert

C = ozaki2_int8_matmul(A, B)   # FP64-exact, INT8 tensor cores
model = convert(model)          # swap all nn.Linear layers
```

## Why

Higher-order PDE residuals (2nd/3rd derivatives) accumulate rounding error through the chain rule. FP32 is not enough. FP64 helps but costs 1/32 throughput on consumer GPUs (e.g. RTX 3090).

Ozaki scheme II emulates FP64 using INT8 tensor cores: scale to integers, L modular matmuls via small primes, CRT reconstruction. Result is exact to FP64.

## Dedicated fixed-point accelerator

On SOTA GPUs, Ozaki is slower than native FP64 (L kernel launches, GPU already has FP64 units). The point is dedicated fixed-point hardware.

A dedicated INT8 ASIC eliminates FP64 hardware entirely:
- INT8 MACs are tiny, pack thousands on small die
- L matmuls pipeline in hardware (no kernel launch overhead)
- CRT is a fixed datapath, not a general-purpose kernel

Yosys synthesis (gate-level cell counts, `hw/synth/`):

| Unit | Cells |
|------|------:|
| INT8 MAC (Q4.4) | 1,239 |
| FP64 MAC | 20,071 |

| Activation (tanh, 10-seg) | Cells |
|----------------------------|------:|
| PWL with multiplier | 2,162 |
| ML-PLAC (shifts only) | 2,046 |

| Slope core width | Multiplier | Shift-add |
|-----------------:|-----------:|----------:|
| 16-bit | 1,840 | 384 |
| 32-bit | 7,137 | 817 |
| 64-bit | 27,578 | 1,683 |

Multiplier scales O(N^2), shift-add scales O(N). At 64-bit the shift-add slope core is 16x smaller.

PINNs benefit because they need both:
- High precision matmul (Ozaki)
- High precision activations (ML-PLAC bit-shift approximations)
- Autodiff through INT8 path (supported via `torch.autograd`)

[DT-PINNs](https://arxiv.org/abs/2205.09332) replace autodiff with numerical derivatives (RBF-FD), making matmul precision the bottleneck. FP64 DT-PINNs already outperform FP32 vanilla PINNs in wall-clock time.

## Demo

[`notebooks/04_demo.ipynb`](notebooks/04_demo.ipynb) — accuracy, benchmarks, Burgers' equation PINN.

![Accuracy](figures/accuracy.png)
![Burgers loss](figures/burgers_loss.png)
![Burgers solution](figures/burgers_solution.png)

To try for free on Nvidia T4 GPU:
1. Make a personal fork of this repo
2. Open notebook in Google Colab (File -> Open notebook -> GitHub -> your fork)
3. Set runtime to T4 GPU (Runtime -> Change runtime type -> T4)

T4 is ideal: fast INT8 tensor cores, crippled FP64 (1/32 rate).

## References

- [Ozaki II scheme for matmul](https://arxiv.org/abs/2504.08009)
- [Ozaki error-free transformations](https://dl.acm.org/doi/epdf/10.1145/3731599.3767539)
- [TwoProduct/TwoSum](https://doi.org/10.1137/030601818)
- [ML-PLAC bit-shift piecewise linear](https://www.mdpi.com/2076-3417/12/20/10616)
- [DT-PINNs](https://arxiv.org/abs/2205.09332)
- [What Every CS Should Know About FP Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
