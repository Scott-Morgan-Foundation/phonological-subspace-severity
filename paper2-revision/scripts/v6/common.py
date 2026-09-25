"""Shared helpers for the v6 reruns on the corrected inputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parents[2]
CORR = REV / "corrected"
MASTER = CORR / "track4_master.csv"
FROZEN_MASTER = REV / "frozen_package" / "results" / "track4_master.csv"
OUT = REV / "results" / "v6"
OUT.mkdir(parents=True, exist_ok=True)

C5 = ["nasal_dprime", "voicing_dprime", "sonorant_dprime", "strident_dprime", "manner_dprime"]
V4 = ["high_dprime", "low_dprime", "back_dprime", "round_dprime"]
F13 = C5 + V4 + ["vowel_triangle_area", "speech_rate", "pause_rate", "vowel_duration_cv"]
SEV = {"control": 0, "mild": 1, "moderate": 2, "severe": 3}
MAIN = ["healthy", "parkinsons", "cerebral_palsy", "als", "down_syndrome", "stroke"]
SHORT = {"healthy": "HC", "parkinsons": "PD", "cerebral_palsy": "CP", "als": "ALS",
         "down_syndrome": "DS", "stroke": "Stroke"}
BACKBONES = {"HuBERT-base": "track4_master.csv",
             "HuBERT-large": "track4_results_hubert-large.csv",
             "WavLM-base": "track4_results_wavlm.csv",
             "wav2vec2-base": "track4_results_wav2vec2.csv",
             "XLS-R-300M": "track4_results_xlsr.csv",
             "MMS-300M": "track4_results_mms.csv"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def f(v):
    if v in ("", None, "nan", "NaN", "None"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def load(path=MASTER):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_keylast(path):
    """Unique (dataset, speaker_id) keys; a repeated key keeps its last row."""
    d = {}
    for r in load(path):
        d[(r["dataset"], r["speaker_id"])] = r
    return list(d.values())


def comp(r, feats=C5, mn=3):
    v = [f(r.get(c)) for c in feats]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) >= mn else None


def sev_code(r):
    """Stated convention: six analysis groups; healthy controls = 0, others by label."""
    if r["aetiology"] not in MAIN:
        return None
    if r["is_control"] == "True":
        return 0
    return SEV.get(r["severity_label"])


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    s = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s)


def write(name, payload, inputs):
    def rel(p):
        p = Path(p)
        try:
            return str(p.relative_to(REV)).replace("\\", "/")
        except ValueError:
            return "../../" + str(p.relative_to(REV.parents[1])).replace("\\", "/")
    payload = {"inputs": {rel(p): sha256(p) for p in inputs}, **payload}
    p = OUT / name
    p.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print("wrote", p)
    return p
