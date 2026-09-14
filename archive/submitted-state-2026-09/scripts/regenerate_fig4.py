#!/usr/bin/env python3
"""Regenerate Figure 4 (severity gradient across backbones) with bootstrap CIs."""

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
MASTER = BASE / "results" / "track4" / "track4_master.csv"
BACKBONE_DIR = BASE / "results" / "track4"
OUT = BASE / "results" / "track4" / "figures" / "cross_model_severity_bars_ci.png"

CONS_FEATS = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
SEV_ORDER = ["control", "mild", "moderate", "severe"]
SEV_MAP = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}

BACKBONES = {
    "HuBERT-base": MASTER,
    "HuBERT-large": BACKBONE_DIR / "track4_results_hubert-large.csv",
    "WavLM": BACKBONE_DIR / "track4_results_wavlm.csv",
    "wav2vec2": BACKBONE_DIR / "track4_results_wav2vec2.csv",
    "XLS-R": BACKBONE_DIR / "track4_results_xlsr.csv",
    "MMS": BACKBONE_DIR / "track4_results_mms.csv",
}

N_BOOT = 1000
RNG = np.random.RandomState(42)

def safe_float(v):
    if v in ("", "nan", None): return None
    try:
        fv = float(v)
        return fv if not np.isnan(fv) else None
    except: return None

def composite(row):
    vals = [safe_float(row.get(f)) for f in CONS_FEATS]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if len(vals) >= 3 else None

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def bootstrap_ci(values, n_boot=N_BOOT):
    values = np.array(values)
    means = []
    for _ in range(n_boot):
        idx = RNG.choice(len(values), size=len(values), replace=True)
        means.append(np.mean(values[idx]))
    return np.percentile(means, [2.5, 97.5])

# Collect data
all_data = {}
for name, path in BACKBONES.items():
    if not path.exists():
        print(f"  Skipping {name}: {path} not found")
        continue
    rows = load_csv(path)
    by_sev = defaultdict(list)
    for r in rows:
        sev = r.get("severity_label", "")
        if sev not in SEV_MAP:
            continue
        c = composite(r)
        if c is not None:
            by_sev[sev].append(c)
    all_data[name] = by_sev
    print(f"  {name}: {', '.join(f'{s}={len(by_sev[s])}' for s in SEV_ORDER)}")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

colors = {"control": "#2ecc71", "mild": "#f1c40f", "moderate": "#e67e22", "severe": "#e74c3c"}

for idx, (name, by_sev) in enumerate(all_data.items()):
    ax = axes[idx]
    means = []
    ci_lo = []
    ci_hi = []
    ns = []
    for sev in SEV_ORDER:
        vals = by_sev.get(sev, [])
        if vals:
            m = np.mean(vals)
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            ci_lo.append(m - lo)
            ci_hi.append(hi - m)
            ns.append(len(vals))
        else:
            means.append(0)
            ci_lo.append(0)
            ci_hi.append(0)
            ns.append(0)

    bars = ax.bar(range(4), means, color=[colors[s] for s in SEV_ORDER],
                  yerr=[ci_lo, ci_hi], capsize=4, error_kw={'linewidth': 1.2})
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Control", "Mild", "Moderate", "Severe"], fontsize=9)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.25)

    # Annotate n
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + ci_hi[i] + 0.05, f'n={n}', ha='center', va='bottom', fontsize=7, color='gray')

axes[0].set_ylabel("Composite consonant d\u2032", fontsize=11)
axes[3].set_ylabel("Composite consonant d\u2032", fontsize=11)

plt.suptitle("Severity gradient across 6 SSL backbones\n(95% bootstrap CIs, 1000 resamples)",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(str(OUT), dpi=300, bbox_inches='tight')
print(f"\nSaved: {OUT}")
