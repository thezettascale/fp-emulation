import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# yosys cell counts (from `make` in hw/synth/)
MAC = {"INT8 MAC": 1239, "FP64 MAC": 20071}
SLOPE_MUL = {16: 1840, 32: 7137, 64: 27578}
SLOPE_SHIFT = {16: 384, 32: 817, 64: 1683}

GREEN, RED = "#4CAF50", "#F44336"

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# MAC
ax = axes[0]
names, vals = list(MAC.keys()), list(MAC.values())
ax.bar(names, vals, color=[GREEN, RED], width=0.5)
ax.set_ylabel("Cells")
ax.set_title("MAC unit (16x smaller)")
for i, v in enumerate(vals):
    ax.text(i, v + 400, f"{v:,}", ha="center", fontsize=10)
ax.set_ylim(0, 24000)

# Slope scale
ax = axes[1]
widths = list(SLOPE_MUL.keys())
mul = list(SLOPE_MUL.values())
shift = list(SLOPE_SHIFT.values())
x = np.arange(len(widths))
w = 0.35
ax.bar(x - w / 2, mul, w, label="Multiplier O(N²)", color=RED)
ax.bar(x + w / 2, shift, w, label="Shift-add O(N)", color=GREEN)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}-bit" for b in widths])
ax.set_ylabel("Cells")
ax.set_title("PWL slope scaling")
ax.legend()
for i in range(len(widths)):
    ax.text(i - w / 2, mul[i] + 500, f"{mul[i]:,}", ha="center", fontsize=9)
    ax.text(i + w / 2, shift[i] + 500, f"{shift[i]:,}", ha="center", fontsize=9)
ax.set_ylim(0, 33000)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent.parent / "figures" / "synth.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
