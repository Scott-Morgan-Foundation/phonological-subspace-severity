#!/usr/bin/env python3
"""v17: compare the submitted/v12 fixed-token run with the corrected v17 run.

Reads results/v12/v12_fixed_token_hcdir{.json,_speakers.csv} and
results/v17/v17_fixed_token_hcdir{.json,_speakers.csv}. Reports, per budget and per scoring
(own axis / HC direction):
  - n, rho, monotonic severity means, KW groups and eps2 (from each run's JSON);
  - who qualifies: speakers by language and by aetiology;
  - the restricted analysis excluding Slovak and Swahili, recomputed from the per-speaker rows,
    with its own common-speaker set;
  - per-speaker movement between runs for speakers whose language map did not change
    (pure resampling noise from the shared RNG sequence).
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REV = Path(__file__).resolve().parents[2]
RUNS = {"v12": REV / "results" / "v12", "v17": REV / "results" / "v17"}
OUT = REV / "results" / "v17" / "v17_fixed_token_comparison.json"
BUDGETS = ["20", "50", "100", "200"]
AET_SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP",
             "als": "ALS", "down_syndrome": "DS", "stroke": "Stroke"}
EXCLUDE = {"sk", "sw"}


def load(run):
    d = RUNS[run]
    j = json.loads((d / f"{run}_fixed_token_hcdir.json").read_text())
    rows = list(csv.DictReader(open(d / f"{run}_fixed_token_hcdir_speakers.csv", encoding="utf-8")))
    for r in rows:
        r["severity"] = int(r["severity"])
        for v in ("own", "hc", "hc_full"):
            r[v] = float(r[v]) if r[v] not in ("", "None") else None
    return j, rows


def stats_for(rows, var):
    ok = [r for r in rows if r[var] is not None]
    out = {"n": len(ok)}
    if len(ok) < 20:
        return out
    rho, p = stats.spearmanr([r["severity"] for r in ok], [r[var] for r in ok])
    out.update(rho=round(float(rho), 4), p=float(p))
    by = defaultdict(list)
    for r in ok:
        by[r["severity"]].append(r[var])
    means = [np.mean(by[k]) for k in sorted(by)]
    out["sev_means"] = {k: round(float(np.mean(by[k])), 3) for k in sorted(by)}
    out["monotonic"] = bool(all(a > b for a, b in zip(means, means[1:])))
    ag = defaultdict(list)
    for r in ok:
        ag[AET_SHORT[r["aetiology"]]].append(r[var])
    valid = {a: g for a, g in ag.items() if len(g) >= 5}
    if len(valid) >= 2:
        H, pk = stats.kruskal(*valid.values())
        N, k = sum(map(len, valid.values())), len(valid)
        out.update(kw_groups={a: len(g) for a, g in valid.items()}, kw_p=float(pk),
                   eps2=round(float(max((H - k + 1) / (N - k), 0)), 4))
    return out


def main():
    res = {}
    data = {run: load(run) for run in RUNS}
    for run, (j, rows) in data.items():
        r = {"faithfulness": j["faithfulness_hc_full_vs_published"], "budgets": {}}
        for b in BUDGETS:
            sub = [x for x in rows if x["budget"] == b]
            pb = j["per_budget"][b]
            r["budgets"][b] = {
                "qualify_by_language": dict(Counter(x["language"] for x in sub)),
                "qualify_by_aetiology": dict(Counter(AET_SHORT[x["aetiology"]] for x in sub)),
                "own": {k: pb["own"].get(k) for k in ("n", "rho", "monotonic", "kw_groups", "eps2")},
                "hc": {k: pb["hc"].get(k) for k in ("n", "rho", "monotonic", "kw_groups", "eps2")},
                "spearman_own_vs_hc": pb.get("spearman_own_vs_hc"),
                "spearman_hc_vs_hcfull": pb.get("spearman_hc_vs_hcfull"),
            }
        r["common_speaker_set"] = j["common_speaker_set"]
        # restricted: no Slovak, no Swahili
        rest = [x for x in rows if x["language"] not in EXCLUDE]
        r["restricted_no_sk_sw"] = {b: {v: stats_for([x for x in rest if x["budget"] == b], v)
                                        for v in ("own", "hc")} for b in BUDGETS}
        common = {}
        for v in ("own", "hc"):
            keep = set.intersection(*[{x["speaker"] for x in rest if x["budget"] == b and x[v] is not None}
                                      for b in BUDGETS])
            common[v] = {"n": len(keep)}
            for b in BUDGETS:
                s = [x for x in rest if x["budget"] == b and x["speaker"] in keep]
                common[v][b] = round(float(stats.spearmanr([x["severity"] for x in s], [x[v] for x in s])[0]), 4)
        r["restricted_no_sk_sw"]["common_speaker_set"] = common
        res[run] = r

    # movement of unaffected speakers (language map unchanged) between runs
    a = {(x["budget"], x["speaker"]): x for x in data["v12"][1]}
    b = {(x["budget"], x["speaker"]): x for x in data["v17"][1]}
    move = {}
    for bud in BUDGETS:
        keys = [k for k in a if k in b and k[0] == bud and a[k]["language"] not in EXCLUDE]
        mv = {}
        for v in ("own", "hc"):
            pairs = [(a[k][v], b[k][v]) for k in keys if a[k][v] is not None and b[k][v] is not None]
            x, y = np.array(pairs).T
            mv[v] = {"n": len(pairs), "spearman": round(float(stats.spearmanr(x, y)[0]), 5),
                     "median_abs_diff": round(float(np.median(np.abs(x - y))), 5),
                     "max_abs_diff": round(float(np.max(np.abs(x - y))), 4)}
        mv["only_in_v12"] = sum(1 for k in a if k[0] == bud and k not in b)
        mv["only_in_v17"] = sum(1 for k in b if k[0] == bud and k not in a)
        move[bud] = mv
    res["unaffected_speaker_movement_v12_to_v17"] = move
    OUT.write_text(json.dumps(res, indent=2))
    for run in RUNS:
        print(f"=== {run} === faithfulness", {k: round(v["spearman"], 4) for k, v in res[run]["faithfulness"].items()})
        for bud in BUDGETS:
            x = res[run]["budgets"][bud]
            print(f" {bud}: own n={x['own']['n']} rho={x['own']['rho']:.3f} eps2={x['own'].get('eps2')} | "
                  f"hc rho={x['hc']['rho']:.3f} eps2={x['hc'].get('eps2')} mono={x['hc']['monotonic']} | "
                  f"lang {x['qualify_by_language']}")
        cs = res[run]["common_speaker_set"]
        print("  common", {v: [cs[v]["n"]] + [round(cs[v][bb]["rho"], 3) for bb in BUDGETS] for v in ("own", "hc")})
        rr = res[run]["restricted_no_sk_sw"]
        print("  restricted", {bud: (rr[bud]["own"]["n"], rr[bud]["own"].get("rho"), rr[bud]["hc"].get("rho"),
                                     rr[bud]["hc"].get("eps2")) for bud in BUDGETS},
              "common", rr["common_speaker_set"])
    print("movement", json.dumps(move))


if __name__ == "__main__":
    main()
