"""v6: re-analysis of the direction-sensitivity (Test 4) and degradation / random-direction
(Test 5) controls against the corrected master.

The per-speaker d-primes for these controls were computed on the DGX from the phone-embedding
cache (unchanged by the data corrections). Direction estimation used `scripts/hc_speakers.tsv`
(aetiology == healthy); this script checks that list equals the healthy set of the corrected
master, so the direction pools match the corrected inputs. Speaker labels for every summary
below come from the corrected master.
Test 5b definitions (reconstructed; they reproduce the saved submitted-state summaries):
HC = aetiology healthy; dysarthric = PD, CP, ALS; composite = mean of the available consonant
d-primes; SMD = pooled-SD standardised mean difference HC - dysarthric per language; random
floor = mean and 95th percentile over the 50 random-direction draws. Cell cosines: HC-normalised
5-feature profiles of language x aetiology cells (PD/CP/ALS, >= 3 speakers), cosine over all
cross-language cell pairs, averaged per draw.
Test 5a: condition/clean ratios of group-mean d-prime per feature (20 LibriSpeech speakers);
English aetiology ratios = aetiology mean / English HC mean from the corrected master;
'cos_dev' = cosine of (1 - ratio) vectors.
"""
import itertools
import json

import numpy as np
import pandas as pd

from common import MASTER, REV, write

C = ["nasal", "voicing", "sonorant", "strident", "manner"]
RES = REV / "results"
HC_LIST = REV / "scripts" / "hc_speakers.tsv"
RAND = RES / "test5_random_dprime.csv"
DEG = RES / "test5_degradation_dprime.csv"
T4 = RES / "v6" / "v6_test4_summary.json"
T4CSV = RES / "test4_speaker_dprime.csv"

m = pd.read_csv(MASTER)
m["sid"] = m.speaker_id.astype(str).str.replace("/", "_").str.replace("\\", "_").str.replace(" ", "_")
hc_master = set(zip(m[m.aetiology == "healthy"].dataset, m[m.aetiology == "healthy"].sid))
hc_list = set(tuple(l.rstrip("\n").split("\t")) for l in open(HC_LIST, encoding="utf-8") if l.strip())
out = {"definition": __doc__,
       "hc_list_check": {"n_list": len(hc_list), "n_master_healthy": len(hc_master),
                         "identical": hc_list == hc_master}}

# Test 4 summary (produced by v6_test4_analysis.py on the corrected master)
t4 = json.loads(T4.read_text(encoding="utf-8"))
out["test4"] = t4

# Test 5b
r = pd.read_csv(RAND)
r["speaker_id"] = r.speaker_id.astype(str)
meta = m.drop_duplicates(["dataset", "sid"]).set_index(["dataset", "sid"])
r = r.join(meta[["aetiology"] + [c + "_dprime" for c in C]], on=["dataset", "speaker_id"])
A = ["parkinsons", "cerebral_palsy", "als"]


def smd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s)


sep = {}
for lang, x in r.groupby("language"):
    d0 = x[x.draw == 0]
    hc, dys = d0.aetiology == "healthy", d0.aetiology.isin(A)
    if dys.sum() < 2:
        continue
    real = d0[[c + "_dprime" for c in C]].mean(1)
    rands = []
    for _, y in x.groupby("draw"):
        comp = y[C].mean(1)
        rands.append(smd(comp[y.aetiology == "healthy"].dropna(), comp[y.aetiology.isin(A)].dropna()))
    sep[lang] = {"real_smd": smd(real[hc].dropna(), real[dys].dropna()), "random_smd_mean": float(np.mean(rands)),
                 "random_smd_p95": float(np.percentile(rands, 95)), "n_hc": int(hc.sum()), "n_dys": int(dys.sum())}
out["test5_random_separation"] = sep


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


dm, wi, be = [], [], []
for _, y in r.groupby("draw"):
    cells = {}
    for lang, z in y.groupby("language"):
        hcm = z[z.aetiology == "healthy"][C].mean()
        for a in A:
            w = z[z.aetiology == a]
            if len(w) >= 3:
                cells[(lang, a)] = (w[C].mean() / hcm).to_numpy(float)
    cs, w_, b_ = [], [], []
    for (k1, v1), (k2, v2) in itertools.combinations(cells.items(), 2):
        if k1[0] == k2[0] or np.isnan(v1).any() or np.isnan(v2).any():
            continue
        c = cos(v1, v2)
        cs.append(c)
        (w_ if k1[1] == k2[1] else b_).append(c)
    dm.append(np.mean(cs)); wi.append(np.mean(w_)); be.append(np.mean(b_))
out["test5_random_cell_cosine"] = {"mean": float(np.mean(dm)), "draw_range": [float(min(dm)), float(max(dm))],
                                   "within_mean": float(np.mean(wi)), "between_mean": float(np.mean(be)),
                                   "delta_mean": float(np.mean(np.array(wi) - np.array(be))), "n_draws": len(dm)}

# Test 5a
g = pd.read_csv(DEG)
clean = g[g.condition == "clean"][C].mean()
shapes = {cond: (x[C].mean() / clean).round(4).to_dict() for cond, x in g.groupby("condition") if cond != "clean"}
en = m[m.language == "en"]
hcm = en[en.aetiology == "healthy"][[c + "_dprime" for c in C]].mean().to_numpy(float)
aet = {}
for a, lab in (("parkinsons", "PD"), ("cerebral_palsy", "CP"), ("als", "ALS")):
    aet[lab] = (en[en.aetiology == a][[c + "_dprime" for c in C]].mean().to_numpy(float) / hcm)
cosd = {}
for cond, sh in shapes.items():
    v = np.array([sh[c] for c in C])
    for lab, av in aet.items():
        cosd[f"{cond}|{lab}"] = {"cos": cos(v, av), "cos_dev": cos(1 - v, 1 - av)}
noise0 = shapes["noise_snr0"]
out["test5_degradation"] = {"shapes": shapes, "aetiology_en_ratio": {k: v.round(4).tolist() for k, v in aet.items()},
                            "cosines": cosd,
                            "noise_snr0_range": [min(noise0.values()), max(noise0.values())],
                            "noise_snr0_most_preserved": max(noise0, key=noise0.get),
                            "noise_snr0_most_reduced": min(noise0, key=noise0.get)}

write("v6_test45_analysis.json", out, [MASTER, HC_LIST, RAND, DEG, T4CSV, T4])
print(out["hc_list_check"])
print({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sep.items()})
print(out["test5_random_cell_cosine"])
print(out["test5_degradation"]["noise_snr0_range"], out["test5_degradation"]["noise_snr0_most_preserved"],
      out["test5_degradation"]["noise_snr0_most_reduced"], noise0)
print({k: round(v["cos_dev"], 3) for k, v in cosd.items() if "noise_snr0" in k})
