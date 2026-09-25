"""Figure 5 (v5): inter-backbone Spearman rho on the >=3-feature composite.

Source: results/v5/test6_v5.json (corrected inputs, keep-last for duplicated keys).
Drawn at its placed width (5.40 in, \\textwidth) so native font size = printed size.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "results" / "v5" / "test6_v5.json"
OUT_PDF = BASE / "figures" / "fig5_intermodel_v5.pdf"
OUT_PNG = BASE / "figures" / "fig5_intermodel_v5.png"

d = json.loads(SRC.read_text(encoding="utf-8"))
BACKS = ["hubert-base", "hubert-large", "wavlm", "wav2vec2", "xlsr", "mms"]
LABELS = ["HuBERT-base", "HuBERT-large", "WavLM", "wav2vec 2.0", "XLS-R", "MMS"]
n = len(BACKS)
rho = np.full((n, n), np.nan)
for i, a in enumerate(BACKS):
    for j, b in enumerate(BACKS):
        if i == j:
            rho[i, j] = 1.0
            continue
        k = f"{a}|{b}" if f"{a}|{b}" in d["per_pair"] else f"{b}|{a}"
        rho[i, j] = d["per_pair"][k]["rho"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(5.40, 4.30))
im = ax.imshow(rho, vmin=0.80, vmax=1.0, cmap="YlGnBu")
for i in range(n):
    for j in range(n):
        ax.text(j, i, "1" if i == j else f"{rho[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                color="white" if rho[i, j] > 0.93 else "#0B1F5C")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(LABELS, fontsize=7.5, rotation=30, ha="right")
ax.set_yticklabels(LABELS, fontsize=7.5)
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Spearman rho (per-speaker composite)", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)
fig.subplots_adjust(left=0.20, right=0.96, top=0.97, bottom=0.20)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print("wrote", OUT_PDF, "n range", d["n_range"], "rho", d["rho_min"], d["rho_max"])
