"""v6: master-based analyses on the corrected inputs.

Severity convention (Statistical conventions): six analysis groups; healthy controls
(is_control) = 0, others by severity label (mild 1, moderate 2, severe 3).
Composite = mean of available consonant d-primes, >= 3 of 5 valid.
"""
import itertools
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon, ttest_ind
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import MASTER, C5, V4, F13, SEV, MAIN, write, CORR

SEED = 42
AM = {"parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS", "healthy": "HC",
      "down_syndrome": "DS", "stroke": "Stroke"}
m = pd.read_csv(MASTER)
m["g"] = m.aetiology.map(AM)
isc = m.is_control.astype(str) == "True"
m["s"] = m.severity_label.map(SEV)
m.loc[isc, "s"] = 0
m.loc[m.g.isna(), "s"] = np.nan          # convention: six groups only
m["s_label"] = m.severity_label.map(SEV)  # label-only coding (ablation script)
nv = m[C5].notna().sum(1)
m["comp"] = m[C5].mean(1).where(nv >= 3)
res = {"definition": __doc__}


def eps2(groups):
    groups = [g for g in groups if len(g) >= 1]
    H, p = kruskal(*groups)
    N, k = sum(len(g) for g in groups), len(groups)
    return float((H - k + 1) / (N - k)), float(H), float(p), int(N)


def cd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s)


# 1. Severity correlations with token-quartile-stratified bootstrap (§4.1), per-feature (§4.4), VTA (§5.5)
rng = np.random.default_rng(SEED)
sv = {}
for name, col in [("composite", "comp")] + [(c, c) for c in C5] + [("vowel_triangle_area", "vowel_triangle_area")]:
    d = m[m.s.notna() & m[col].notna()]
    x, y, tok = d[col].to_numpy(float), d.s.to_numpy(float), d.n_phones.to_numpy(float)
    r, p = spearmanr(x, y)
    q = np.quantile(tok, [0.25, 0.5, 0.75])
    st = np.digitize(tok, q, right=True)
    idx = [np.where(st == k)[0] for k in range(4)]
    bs = []
    for _ in range(1000):
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx if len(ix)])
        bs.append(spearmanr(x[ii], y[ii])[0])
    sv[name] = {"rho": float(r), "p": float(p), "n": int(len(d)),
                "ci95_stratified": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
res["severity"] = sv

# 2. CTC-Conf under the convention
d = m[m.s.notna() & m.artp_score.notna()]
r, p = spearmanr(d.artp_score, d.s)
res["ctc_conf_severity"] = {"rho": float(r), "p": float(p), "n": int(len(d)),
                            "means": {k: float(d[d.s == v].artp_score.mean()) for k, v in SEV.items()},
                            "n_per_level": {k: int((d.s == v).sum()) for k, v in SEV.items()}}

# 3. Severity-source ablation (label-only coding in both arms, as the ablation script)
CLIN_EXCL = {"LibriSpeech_English", "EasyCall", "SVD", "AVFAD", "SLR65_Tamil", "EWA-DB", "Hungarian_HC", "CV_Hungarian"}
lab = m[m.s_label.notna() & m.comp.notna() & m.aetiology.isin(MAIN)]
out = {}
for arm, dd in (("all_labelled", lab), ("clinical_only", lab[~lab.dataset.isin(CLIN_EXCL)])):
    r, p = spearmanr(dd.comp, dd.s_label)
    e, H, pk, N = eps2([x.comp.values for _, x in dd.groupby("aetiology") if len(x) >= 5])
    out[arm] = {"n": int(len(dd)), "rho": float(r), "p": float(p), "eps2": e, "kw_p": pk,
                "means": {k: float(dd[dd.s_label == v].comp.mean()) for k, v in SEV.items()}}
res["severity_source_ablation"] = out
res["severity_source_ablation"]["note"] = ("clinical_only drops datasets whose severity labels are metadata-derived; "
                                           "same label-only coding and six groups in both arms")

# 4. Leave-one-dataset-out (convention coding; eps2 over six groups on composite)
full_e = eps2([x.comp.values for _, x in m[m.s.notna() & m.comp.notna()].groupby("g") if len(x) >= 5])[0]
full_r = spearmanr(m[m.s.notna() & m.comp.notna()].comp, m[m.s.notna() & m.comp.notna()].s)[0]
lodo = {}
for ds in sorted(m.dataset.unique()):
    sub = m[m.dataset != ds]
    ss = sub[sub.s.notna() & sub.comp.notna()]
    if len(ss) < 20 or ss.s.nunique() < 2:
        continue
    r, rp = spearmanr(ss.comp, ss.s)
    gs = [x.comp.values for _, x in ss.groupby("g") if len(x) >= 5]
    e_, _, ep_, _ = eps2(gs)
    lodo[ds] = {"rho": float(r), "p": float(rp), "eps2": e_, "kw_p": ep_, "n": int(len(ss))}
rh = [v["rho"] for v in lodo.values()]
ee = [v["eps2"] for v in lodo.values()]
res["lodo"] = {"full_rho": float(full_r), "full_eps2": float(full_e), "folds": lodo, "n_folds": len(lodo),
               "rho_range": [min(rh), max(rh)], "rho_mean": float(np.mean(rh)),
               "eps2_range": [min(ee), max(ee)], "eps2_mean": float(np.mean(ee)),
               "most_influential": min(lodo, key=lambda k: abs(lodo[k]["rho"])),
               "max_p_rho": max(v["p"] for v in lodo.values()), "max_kw_p": max(v["kw_p"] for v in lodo.values())}

# 5. Holm-corrected pairwise Mann-Whitney + rank-biserial + signed d (composite)
G = {g: m[m.g == g].comp.dropna().values for g in ["HC", "PD", "CP", "ALS", "DS", "Stroke"]}
pairs, ps = [], []
for a, b in itertools.combinations(G, 2):
    U = mannwhitneyu(G[a], G[b], alternative="two-sided")
    pairs.append({"pair": f"{a}|{b}", "p": float(U.pvalue),
                  "rank_biserial": float(1 - 2 * U.statistic / (len(G[a]) * len(G[b]))),
                  "d_second_minus_first": cd(G[b], G[a]), "n": [len(G[a]), len(G[b])]})
    ps.append(U.pvalue)
order = np.argsort(ps)
holm = np.empty(len(ps))
run = 0
for rank, i in enumerate(order):
    run = max(run, min(ps[i] * (len(ps) - rank), 1.0))
    holm[i] = run
for pr, h in zip(pairs, holm):
    pr["p_holm"] = float(h)
res["posthoc"] = {"pairs": pairs, "n_sig_0.001": int((holm < 0.001).sum()),
                  "group_n": {g: int(len(v)) for g, v in G.items()}}

# 6. SAP-excluded (HC/PD/CP/ALS), convention coding
four = ["HC", "PD", "CP", "ALS"]
sapx = {}
for lab_, dd in (("with_sap", m), ("without_sap", m[m.dataset != "SAP"])):
    o = {"n_rows": int(len(dd)), "group_n_composite": {g: int(dd[(dd.g == g)].comp.notna().sum()) for g in AM.values()}}
    ss = dd[dd.s.notna() & dd.comp.notna()]
    r, p = spearmanr(ss.comp, ss.s)
    o["severity"] = {"rho": float(r), "p": float(p), "n": int(len(ss))}
    for c in ["comp"] + C5 + V4:
        x = dd[dd.g.isin(four) & dd[c].notna()]
        e, H, p, N = eps2([y[c].values for _, y in x.groupby("g")])
        o[c] = {"eps2": e, "H": H, "p": p, "N": N}
    sapx[lab_] = o
sapx["sap_by_aetiology"] = m[m.dataset == "SAP"].aetiology.value_counts().to_dict()
res["sap_excluded"] = sapx

# 7. Token-count adjustment: linear regression of each consonant d' on n_phones (pooled),
#    compare group means / rankings before and after
adj = m.copy()
for c in C5 + V4 + ["vowel_triangle_area"]:   # method of frozen adjust_token_count.py
    ok = adj[c].notna() & (adj.n_phones > 0)
    X = adj.loc[ok, "n_phones"].to_numpy(float)
    slope = np.polyfit(X, adj.loc[ok, c].to_numpy(float), 1)[0]
    gm = m[c].mean()
    adj.loc[ok, c] = adj.loc[ok, c] - slope * X + gm
adj["comp"] = adj[C5].mean(1).where(adj[C5].notna().sum(1) >= 3)
def gm(df):
    return {g: float(df[df.g == g].comp.mean()) for g in ["HC", "PD", "CP", "ALS", "DS", "Stroke"]}
def sm(df):
    return {k: float(df[df.s == v].comp.mean()) for k, v in SEV.items()}
b_g, a_g, b_s, a_s = gm(m), gm(adj), sm(m), sm(adj)
rank_b = sorted(b_g, key=b_g.get, reverse=True)
rank_a = sorted(a_g, key=a_g.get, reverse=True)
pair_flips = [f"{x}|{y}" for x, y in itertools.combinations(b_g, 2)
              if np.sign(b_g[x] - b_g[y]) != np.sign(a_g[x] - a_g[y])]
res["token_adjustment"] = {"method": "frozen adjust_token_count.py: per feature, slope of d' on n_phones over all speakers; adjusted = d' - slope*n_phones + grand mean",
                           "group_means_before": b_g, "group_means_after": a_g,
                           "ranking_before": rank_b, "ranking_after": rank_a,
                           "pairwise_order_flips": pair_flips,
                           "severity_means_before": b_s, "severity_means_after": a_s,
                           "severity_monotonic_after": bool(all(np.diff([a_s[k] for k in SEV]) < 0)),
                           "hc_ds_gap_before": b_g["HC"] - b_g["DS"], "hc_ds_gap_after": a_g["HC"] - a_g["DS"]}

# 8. Token-matched adjacent-severity comparisons (greedy nearest match on n_phones, +/-20%)
tm = {}
rng = np.random.default_rng(SEED)
base = m[m.s.notna() & m.comp.notna() & m.n_phones.notna()]
for lo, hi in ((0, 1), (1, 2), (2, 3)):
    A = base[base.s == lo].sample(frac=1, random_state=SEED)
    B = base[base.s == hi]
    used, pa, pb = set(), [], []
    bn = B.n_phones.to_numpy(float)
    for _, r_ in A.iterrows():
        tol = 0.2 * r_.n_phones
        cand = [i for i in np.argsort(np.abs(bn - r_.n_phones)) if i not in used and abs(bn[i] - r_.n_phones) <= tol]
        if not cand:
            continue
        i = cand[0]
        used.add(i)
        pa.append(r_.comp)
        pb.append(B.iloc[i].comp)
    pa, pb = np.array(pa), np.array(pb)
    w = wilcoxon(pa, pb)
    t = mannwhitneyu(pa, pb)
    tm[f"{lo}->{hi}"] = {"n_pairs": int(len(pa)), "cohen_d": cd(pa, pb), "wilcoxon_p": float(w.pvalue),
                         "mannwhitney_p": float(t.pvalue), "mean_lo": float(pa.mean()), "mean_hi": float(pb.mean())}
res["token_matched"] = {"method": ("each speaker at the lower level (random order, seed 42) matched without "
                                   "replacement to the nearest-n_phones speaker at the next level within +/-20%"),
                        "levels": tm}

# 9. Ridge regression: severity (convention) from the 13 features, with/without CTC-Conf
rd = m[m.s.notna()].dropna(subset=F13 + ["artp_score"])
y = rd.s.to_numpy(float)
kf = KFold(10, shuffle=True, random_state=SEED)
def cv(cols):
    pred = np.zeros(len(y))
    for tr, te in kf.split(rd):
        sc = StandardScaler().fit(rd.iloc[tr][cols])
        mdl = Ridge(alpha=1.0).fit(sc.transform(rd.iloc[tr][cols]), y[tr])
        pred[te] = mdl.predict(sc.transform(rd.iloc[te][cols]))
    return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "rho": float(spearmanr(pred, y)[0])}
res["ridge"] = {"n": int(len(rd)), "features13": cv(F13), "features13_plus_ctc": cv(F13 + ["artp_score"]),
                "ctc_only": cv(["artp_score"]), "alpha": 1.0, "folds": 10, "seed": SEED}

# 10. Feature coverage accounting (§3.2)
D9 = C5 + V4
res["coverage"] = {"no_dprime_any": int((m[D9].notna().sum(1) == 0).sum()),
                   "no_consonant_dprime": int((m[C5].notna().sum(1) == 0).sum()),
                   "nasal_valid": int(m.nasal_dprime.notna().sum()),
                   "no_nasal_but_other_consonant": int((m.nasal_dprime.isna() & (m[C5].notna().sum(1) > 0)).sum()),
                   "no_dprime_by_dataset": m[m[D9].notna().sum(1) == 0].dataset.value_counts().to_dict(),
                   "svd_speakers_without_any_dprime": int(((m.dataset == "SVD") & (m[D9].notna().sum(1) == 0)).sum()),
                   "group_counts_nasal_valid": {g: int(m[(m.g == g)].nasal_dprime.notna().sum()) for g in AM.values()},
                   "group_counts_composite3": {g: int(m[(m.g == g)].comp.notna().sum()) for g in AM.values()}}

# 11. English control token share (LibriSpeech)
en = m[(m.language == "en") & isc]
tot = en.n_phones.sum()
res["english_control_tokens"] = {k: int(v) for k, v in en.groupby("dataset").n_phones.sum().items()}
res["english_control_tokens"]["total"] = int(tot)
res["english_control_tokens"]["librispeech_share"] = float(en[en.dataset == "LibriSpeech_English"].n_phones.sum() / tot)

# 12. Backbone coverage (§3.3)
bb = {}
keys_m = set(zip(m.dataset, m.speaker_id))
for f in ["track4_results_hubert-large.csv", "track4_results_wavlm.csv", "track4_results_wav2vec2.csv",
          "track4_results_xlsr.csv", "track4_results_mms.csv"]:
    b = pd.read_csv(CORR / f)
    kb = set(zip(b.dataset, b.speaker_id))
    miss = keys_m - kb
    miss_d = m[[k in miss for k in zip(m.dataset, m.speaker_id)]]
    bb[f] = {"rows": int(len(b)), "unique_keys": len(kb), "master_keys_missing": len(miss),
             "missing_by_dataset": miss_d.dataset.value_counts().to_dict(),
             "missing_with_hubert_dprime": int((miss_d[D9].notna().sum(1) > 0).sum()),
             "missing_with_hubert_dprime_by_dataset":
                 miss_d[miss_d[D9].notna().sum(1) > 0].dataset.value_counts().to_dict()}
res["backbone_coverage"] = bb

write("v6_master_analyses.json", res,
      [MASTER] + [CORR / f for f in ["track4_results_hubert-large.csv", "track4_results_wavlm.csv",
                                     "track4_results_wav2vec2.csv", "track4_results_xlsr.csv",
                                     "track4_results_mms.csv"]])
import json
print(json.dumps({k: res[k] for k in ["severity", "ctc_conf_severity", "severity_source_ablation"]}, indent=0, default=float)[:3000])
print({k: res["lodo"][k] for k in res["lodo"] if k != "folds"})
print([(p["pair"], round(p["rank_biserial"], 3), round(p["d_second_minus_first"], 3), "%.1e" % p["p_holm"]) for p in pairs])
for k in ("with_sap", "without_sap"):
    o = sapx[k]; print(k, o["n_rows"], o["severity"], {c: round(o[c]["eps2"], 3) for c in ["comp", "nasal_dprime", "high_dprime", "round_dprime"]}, "H", round(o["comp"]["H"], 1), "%.1e" % o["comp"]["p"])
print(res["token_adjustment"]["ranking_before"], res["token_adjustment"]["ranking_after"], res["token_adjustment"]["pairwise_order_flips"], res["token_adjustment"]["hc_ds_gap_before"], res["token_adjustment"]["hc_ds_gap_after"], res["token_adjustment"]["severity_monotonic_after"])
print(tm)
print(res["ridge"])
print(res["coverage"])
print(res["english_control_tokens"])
print({k: (v["unique_keys"], v["master_keys_missing"], v["missing_with_hubert_dprime"], v["missing_by_dataset"]) for k, v in bb.items()})
