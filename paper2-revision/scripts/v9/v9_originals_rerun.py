#!/usr/bin/env python3
"""v9: run the ORIGINAL submitted analysis scripts, unchanged, on the corrected and the submitted inputs.

Originals (byte-identical copies of frozen_package/scripts, SHA256 asserted below):
  robustness_analyses.py  - Analysis 1 stratified bootstrap severity correlation (Section 4.1),
                            Analysis 2 cross-lingual cosine bootstrap (contribution (b)),
                            Analysis 3 leave-one-dataset-out.            Runs locally (master only).
  fixed_token_dprime.py   - Experiment A fixed-token d-prime (Table 6),
                            Experiment B within-language permutation baseline.   Runs on the DGX.
  robustness_pass5.py     - Experiment 1 common-speaker fixed-token set, Experiment 2 minimum-HC
                            cross-lingual cosine, Experiment 3 final-layer backbone rho. Runs on the DGX.
  reviewer_experiments.py - Holm post hoc and minimum-n cosines; already run unchanged in v8
                            (results/v8/originals/<state>/reviewer_experiments.log); parsed here.
Each original reads $DYSARTHRIA_BASE/results/track4/*.csv (+ embeddings/, config/ for the DGX ones).
The only change is that path: a staging directory holding copies of the chosen inputs.
Usage:
  python v9_originals_rerun.py local      # robustness_analyses.py, both states
  python v9_originals_rerun.py dgx        # stages + runs fixed_token_dprime.py and robustness_pass5.py over ssh
  python v9_originals_rerun.py parse      # parse all logs -> results/v9/v9_originals.json
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
ORIG = Path(__file__).resolve().parent / "originals"
OUT = REV / "results" / "v9" / "originals"
STATES = {"corrected": REV / "corrected", "submitted": REV / "frozen_package" / "results"}
INPUTS = ["track4_master.csv", "track4_results_hubert-large.csv", "track4_results_wavlm.csv",
          "track4_results_wav2vec2.csv", "track4_results_xlsr.csv", "track4_results_mms.csv"]
EXPECTED = {"robustness_analyses.py": "f7670dd479b77250b99946399d9f312922884ed7dafee749f06ea01cc0e83697",
            "fixed_token_dprime.py": "02edb7dad2734ca25a900fc0ff082e37a05a1b7b5b05d5a26a8e532f73b62d5c",
            "robustness_pass5.py": "4e206aa1ccbd8524229f2a8e99c3c81c30927e243d89dfb071deb24d97c8749b",
            "reviewer_experiments.py": "b5c4d1f71de2114b25521a6743c2787f0c1fbaf9c9afc53f5197725629f80432"}
# DGX connection from the environment (not hard-coded): P2_DGX_TARGET=user@host, P2_DGX_KEY=<ssh key path>,
# P2_DGX_REMOTE=<scratch dir on the host>, P2_DGX_HOME=<home holding dysarthria/{venv,config,results/track4/embeddings}>.
TARGET = os.environ.get("P2_DGX_TARGET", "")
KEY = os.environ.get("P2_DGX_KEY", "")
REMOTE = os.environ.get("P2_DGX_REMOTE", "")
DHOME = os.environ.get("P2_DGX_HOME", "")
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", TARGET]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_scripts():
    for s, h in EXPECTED.items():
        got = sha(ORIG / s)
        assert got == h, f"{s} is not the original ({got})"


def run_local():
    rec = {}
    for state, src in STATES.items():
        stage = OUT / state / "stage"
        (stage / "results" / "track4").mkdir(parents=True, exist_ok=True)
        for fn in INPUTS:
            if (src / fn).exists():
                shutil.copy2(src / fn, stage / "results" / "track4" / fn)
        env = {**os.environ, "DYSARTHRIA_BASE": str(stage), "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([sys.executable, str(ORIG / "robustness_analyses.py")], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = OUT / state / "robustness_analyses.log"
        log.write_text(r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else ""), encoding="utf-8")
        rec[state] = r.returncode
    print(rec)


def run_dgx():
    assert TARGET and KEY and REMOTE and DHOME, "set P2_DGX_TARGET, P2_DGX_KEY, P2_DGX_REMOTE, P2_DGX_HOME"
    for state, src in STATES.items():
        rd = f"{REMOTE}/{state}"
        subprocess.run(SSH + [f"rm -rf {rd}/results/track4/*.csv; mkdir -p {rd}/results/track4 {rd}/logs && "
                              f"ln -sfn {DHOME}/dysarthria/results/track4/embeddings {rd}/results/track4/embeddings && "
                              f"ln -sfn {DHOME}/dysarthria/config {rd}/config"], check=True)
        for fn in INPUTS:
            if (src / fn).exists():
                subprocess.run(["scp", "-i", KEY, "-q", str(src / fn),
                                f"{TARGET}:{rd}/results/track4/{fn}"], check=True)
    subprocess.run(["scp", "-i", KEY, "-q",
                    str(ORIG / "fixed_token_dprime.py"), str(ORIG / "robustness_pass5.py"),
                    f"{TARGET}:{REMOTE}/"], check=True)
    # Single-threaded BLAS: the per-draw d-prime is a handful of tiny matrix-vector products, and on a shared
    # machine a multi-threaded BLAS spends its time in thread contention (a first attempt stalled for >10 min on
    # one budget). The thread count changes nothing in the computation except floating-point summation order.
    # The four runs (2 scripts x 2 input states) are independent and run concurrently.
    thr = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    py = f"{DHOME}/dysarthria/venv/bin/python"
    runs = " ".join(f"({thr} DYSARTHRIA_BASE={REMOTE}/{st} {py} -u {s}.py > {REMOTE}/{st}/logs/{s}.log 2>&1) &"
                    for st in STATES for s in ("fixed_token_dprime", "robustness_pass5"))
    cmd = (f"cd {REMOTE} || exit 1; sha256sum fixed_token_dprime.py robustness_pass5.py; "
           f"sha256sum {REMOTE}/*/results/track4/*.csv; {runs} wait; echo DONE")
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    for state in STATES:
        (OUT / state).mkdir(parents=True, exist_ok=True)
        for lg in ("fixed_token_dprime.log", "robustness_pass5.log"):
            subprocess.run(["scp", "-i", KEY, "-q",
                            f"{TARGET}:{REMOTE}/{state}/logs/{lg}", str(OUT / state / lg)],
                           check=True)


def f(x):
    return float(x)


def parse_state(state):
    d = OUT / state
    res = {}
    ra = (d / "robustness_analyses.log").read_text(encoding="utf-8")
    a1 = {"mean_rho": f(re.search(r"Mean rho: ([+-]?[\d.]+)", ra).group(1)),
          "ci95": [f(x) for x in re.search(r"95% CI: \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", ra).groups()],
          "n": int(re.search(r"Speakers with labels \+ features: (\d+)", ra).group(1)),
          "per_feature": {m.group(1): {"rho": f(m.group(2)), "ci95": [f(m.group(3)), f(m.group(4))]}
                          for m in re.finditer(r"^\s+(\w+): rho = ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]",
                                               ra, re.M)}}
    a2 = {}
    for m in re.finditer(r"^\s+(PD|CP|ALS|HC) \((\d+) languages\):\n\s+Mean cosine: ([\d.]+) \[([\d.]+), ([\d.]+)\]\n"
                         r"\s+Min pair CI lower bound: ([\d.]+)\n\s+Tightest pair: (\w+)-(\w+): ([\d.]+) \[([\d.]+), ([\d.]+)\]",
                         ra, re.M):
        a2[m.group(1)] = {"n_languages": int(m.group(2)), "mean_cosine": f(m.group(3)),
                          "mean_of_pair_ci_lower": f(m.group(4)), "mean_of_pair_ci_upper": f(m.group(5)),
                          "min_pair_ci_lower": f(m.group(6)),
                          "tightest_pair": {"pair": f"{m.group(7)}-{m.group(8)}", "cos": f(m.group(9)),
                                            "ci95": [f(m.group(10)), f(m.group(11))]}}
    a3 = {"per_dataset": {m.group(1): {"n_drop": int(m.group(2)), "rho": f(m.group(3)), "eta2": f(m.group(4))}
                          for m in re.finditer(r"^\s+(\S+)\s+(\d+)\s+([+-][\d.]+)\s+([\d.]+)\s+(YES|CHECK)", ra, re.M)}}
    g = re.search(r"LODO severity rho: range \[([+-][\d.]+), ([+-][\d.]+)\], mean ([+-][\d.]+)", ra)
    if g:
        a3["rho_range"] = [f(g.group(1)), f(g.group(2))]
        a3["rho_mean"] = f(g.group(3))
    g = re.search(r"LODO eta2: range \[([\d.]+), ([\d.]+)\], mean ([\d.]+)", ra)
    if g:
        a3["eta2_range"] = [f(g.group(1)), f(g.group(2))]
        a3["eta2_mean"] = f(g.group(3))
    res["robustness_analyses"] = {"a1_stratified_bootstrap": a1, "a2_crosslingual_bootstrap": a2, "a3_lodo": a3}

    ft = d / "fixed_token_dprime.log"
    if ft.exists():
        t = ft.read_text(encoding="utf-8")
        budgets = {}
        for blk in re.split(r"--- Token budget: ", t)[1:]:
            b = int(re.match(r"(\d+)", blk).group(1))
            e = {"n": int(re.search(r"Processed: (\d+)", blk).group(1))}
            s = re.search(r"Severity rho: ([+-][\d.]+) \(p=([\d.e+-]+), n=(\d+)\)", blk)
            if s:
                e.update(rho=f(s.group(1)), p=f(s.group(2)), n_rho=int(s.group(3)))
            e["severity_means"] = {m.group(1): {"mean": f(m.group(2)), "n": int(m.group(3))}
                                   for m in re.finditer(r"^\s+(control|mild|moderate|severe): mean=([\d.]+), n=(\d+)",
                                                        blk, re.M)}
            a = re.search(r"Aetiology: H=([\d.]+), p=([\d.e+-]+), eps2=([\d.]+)", blk)
            if a:
                e.update(aet_H=f(a.group(1)), aet_p=f(a.group(2)), aet_eps2=f(a.group(3)))
            budgets[b] = e
        perm = {m.group(1): {"n_languages": int(m.group(2)), "n_pairs": int(m.group(3)), "observed": f(m.group(4)),
                             "null_mean": f(m.group(5)), "null_ci": [f(m.group(6)), f(m.group(7))], "p": f(m.group(8))}
                for m in re.finditer(r"^\s+(PD|CP|ALS|HC) \((\d+) languages, (\d+) pairs\):\n\s+Observed mean cosine: ([\d.]+)\n"
                                     r"\s+Null distribution: mean=([\d.]+) \[([\d.]+), ([\d.]+)\]\n\s+p\(perm >= observed\): ([\d.]+)",
                                     t, re.M)}
        eps = [e["aet_eps2"] for e in budgets.values() if "aet_eps2" in e]
        ps = [e["p"] for e in budgets.values() if "p" in e]
        res["fixed_token_dprime"] = {"expA_per_budget": budgets, "expB_permutation": perm,
                                     "derived": {"eps2_min": min(eps) if eps else None, "eps2_max": max(eps) if eps else None,
                                                 "p_max": max(ps) if ps else None}}
    p5 = d / "robustness_pass5.log"
    if p5.exists():
        t = p5.read_text(encoding="utf-8")
        common = {int(m.group(1)): {"n": int(m.group(2)), "rho": f(m.group(3)), "p": f(m.group(4))}
                  for m in re.finditer(r"Budget\s+(\d+): n=(\d+), rho=([+-][\d.]+), p=([\d.e+-]+)", t)}
        q = re.search(r"Speakers qualifying at all budgets \(>=200 tokens/class\): (\d+)", t)
        minhc = {}
        for blk in re.split(r"=== Minimum HC >= ", t)[1:]:
            thr = int(re.match(r"(\d+)", blk).group(1))
            minhc[thr] = {m.group(1): {"n_languages": int(m.group(2)), "cos": f(m.group(3)),
                                       "ci95": [f(m.group(4)), f(m.group(5))]}
                          for m in re.finditer(r"^\s+(PD|CP|ALS): (\d+) langs, cos=([\d.]+) \[([\d.]+), ([\d.]+)\]",
                                               blk.split("===", 1)[1] if "===" in blk else blk, re.M)}
        layer = {m.group(1): {"n": int(m.group(2)), "rho": f(m.group(3)), "p": f(m.group(4))}
                 for m in re.finditer(r"^\s+(HuBERT-base|HuBERT-large|WavLM|wav2vec2|XLS-R|MMS)\s+(\d+)\s+([+-][\d.]+)\s+([\d.e+-]+)$",
                                      t, re.M)}
        crho = [e["rho"] for e in common.values()]
        cp = [e["p"] for e in common.values()]
        res["robustness_pass5"] = {"exp1_common_set": {"n_qualifying": int(q.group(1)) if q else None,
                                                       "per_budget": common,
                                                       "derived": {"rho_min": min(crho) if crho else None,
                                                                   "rho_max": max(crho) if crho else None,
                                                                   "p_max": max(cp) if cp else None}},
                                   "exp2_min_hc": minhc, "exp3_final_layer_rho": layer}
    rv = (REV / "results" / "v8" / "originals" / state / "reviewer_experiments.log").read_text(encoding="utf-8")
    holm = []
    for m in re.finditer(r"^\s+(\w+) vs (\w+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+(\*+|ns)$",
                         rv, re.M):
        holm.append({"pair": f"{m.group(1)} vs {m.group(2)}", "p_raw": f(m.group(3)), "p_holm": f(m.group(4)),
                     "r_rb": f(m.group(5)), "cohen_d_first_minus_second": f(m.group(6)), "sig": m.group(7)})
    minn = {}
    for blk in re.split(r"=== Minimum n >= ", rv)[1:]:
        thr = int(re.match(r"(\d+)", blk).group(1))
        body = blk.split("\n\n")[0]
        minn[thr] = {m.group(1): {"n_languages": int(m.group(2)), "n_pairs": int(m.group(3)), "mean": f(m.group(4)),
                                  "ci95": [f(m.group(5)), f(m.group(6))], "min": f(m.group(7))}
                     for m in re.finditer(r"^\s+(PD|CP|ALS|HC): (\d+) langs, (\d+) pairs, mean cos=([\d.]+) "
                                          r"\[([\d.]+), ([\d.]+)\], min=([\d.]+)", body, re.M)}
    om = re.search(r"Omnibus: H=([\d.]+), p=([\d.e+-]+), eps2=([\d.]+)", rv)
    gs = re.search(r"Group sizes: (.+)", rv)
    res["reviewer_experiments"] = {"exp3_holm": {"omnibus": {"H": f(om.group(1)), "p": f(om.group(2)), "eps2": f(om.group(3))},
                                                 "group_sizes": dict((k, int(v)) for k, v in
                                                                     re.findall(r"(\w+)=(\d+)", gs.group(1))),
                                                 "pairs": holm},
                                   "exp2_min_n_language_pair_bootstrap": minn}
    return res


def parse():
    rec = {"definition": __doc__, "scripts": {s: sha(ORIG / s) for s in EXPECTED}, "states": {}}
    for state, src in STATES.items():
        rec["states"][state] = {"inputs": {fn: sha(src / fn) for fn in INPUTS if (src / fn).exists()},
                                "logs": {p.name: sha(p) for p in sorted((OUT / state).glob("*.log"))},
                                "parsed": parse_state(state)}
    p = REV / "results" / "v9" / "v9_originals.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    # reproduction check: submitted-input re-runs against the submitted logs (timing lines and the
    # embeddings-directory line, which names the staging path, are not compared)
    drop = re.compile(r"completed in \d+s|^Embeddings dir:")
    rep = {}
    for lg in ("robustness_analyses.log", "fixed_token_dprime.log", "robustness_pass5.log"):
        a, b = REV / "frozen_package" / "logs" / lg, OUT / "submitted" / lg
        if not b.exists():
            rep[lg] = {"status": "not run"}
            continue
        la = [x for x in a.read_text(encoding="utf-8").splitlines() if not drop.search(x)]
        lb = [x for x in b.read_text(encoding="utf-8").splitlines() if not drop.search(x)]
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
        rep[lg] = {"identical": not diff and len(la) == len(lb), "n_lines": [len(la), len(lb)],
                   "differing_lines": diff[:40], "frozen_log_sha256": sha(a), "rerun_log_sha256": sha(b)}
    (REV / "results" / "v9" / "v9_reproduction_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v.get("identical", v.get("status")) for k, v in rep.items()}))
    print(json.dumps({s: rec["states"][s]["parsed"].get("robustness_analyses", {}).get("a2_crosslingual_bootstrap")
                      for s in STATES}, indent=1))


if __name__ == "__main__":
    check_scripts()
    {"local": run_local, "dgx": run_dgx, "parse": parse}[sys.argv[1]]()
