#!/usr/bin/env python3
"""Build the MSD master CSV — Czech + German + PC-GITA OneVoice-MSD26 only.

This is a SEPARATE master from the main Track 4 master. It bundles the three
DUA-restricted OneVoice-MSD-2026 corpora into one row-aligned CSV with a
unified schema, ready for per-dataset Stipancic-calibrated pseudo-labels.

Inputs (Windows paths under track4/results/):
  - track4_master_czech.csv         segmental (101 spk, schema: nasality_dprime,
                                     also has prosodic merged in)
  - track4_results_german.csv       segmental (176 spk, nasal_dprime, no prosodic)
  - track4_results_pcgita_msd.csv   segmental (loads with leaked Neurovoz +
                                     PC-GITA HC pool rows; we filter to MSD only)
  - track4_prosodic_results.csv     prosodic for all datasets in the inventory
                                     (Czech/German/PC-GITA-MSD subset where
                                     extracted)
  - track4_voice_quality.csv        voice quality for all datasets in the
                                     inventory (subset where extracted)

Output:
  - track4_master_msd.csv

Reconciliations applied:
  1. Rename Czech `nasality_dprime` -> `nasal_dprime` (other corpora's name).
  2. Drop the `_read` suffix from Czech and German dataset names so all three
     read flat (`Czech_OneVoice-MSD26`, `German_OneVoice-MSD26`,
     `PC-GITA_OneVoice-MSD26`).
  3. Filter the PC-GITA-MSD segmental file to MSD rows only (Neurovoz and the
     existing PC-GITA leak in via the language HC pool during extraction).
  4. Outer-join segmental + prosodic + voice quality on (dataset, speaker_id),
     filling missing numeric cells with empty strings (NOT NaN — keeps CSV
     parsable downstream).

Coverage notes (kept honest, not silently filled):
  - Prosodic + voice quality currently read from `results/track4/corpus/<DS>/`
    which contains read audio only. ddk + vowel audio sit in
    `data/processed/<DS>/wav_16k/`. Speakers without read audio under the
    corpus dir get blank prosodic/voice-quality cells in the output.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # track4/
RES = ROOT / "results"
OUT = RES / "track4_master_msd.csv"

SEG_FILES = {
    # source CSV -> dataset filter (None means take everything in file,
    # rename happens via DS_RENAME)
    RES / "track4_master_czech.csv": {"Czech_OneVoice-MSD26_read"},
    RES / "track4_results_german.csv": {"German_OneVoice-MSD26_read"},
    RES / "track4_results_pcgita_msd.csv": {"PC-GITA_OneVoice-MSD26"},
}
PROS_FILES = [
    RES / "track4_prosodic_results_czech.csv",
    RES / "track4_prosodic_results_german.csv",
    RES / "track4_prosodic_results.csv",  # global, contains PC-GITA-MSD subset
]
VQ_FILE = RES / "track4_voice_quality.csv"

DS_RENAME = {
    "Czech_OneVoice-MSD26_read": "Czech_OneVoice-MSD26",
    "German_OneVoice-MSD26_read": "German_OneVoice-MSD26",
    "PC-GITA_OneVoice-MSD26": "PC-GITA_OneVoice-MSD26",
}

# Final schema (segmental + prosodic + voice quality)
SEG_COLS = [
    "dataset", "speaker_id", "language", "aetiology", "severity_label",
    "severity_numeric", "is_control", "n_phones",
    "back_dprime", "boundary_sharpness", "cross_position_cosim",
    "high_dprime", "low_dprime", "manner_dprime", "nasal_dprime",
    "round_dprime", "sonorant_dprime", "strident_dprime",
    "voicing_dprime", "vowel_triangle_area",
]
PROS_COLS = [
    "speech_rate", "pause_rate", "vowel_duration_cv",
    "f0_std_semitones", "f0_range_semitones", "f0_mean_hz",
    "n_utterances", "total_phones",
]
VQ_COLS = [
    "jitter_local", "shimmer_local", "hnr_mean", "f0_cv",
    "tremor_ratio", "n_files_vq",
]
ALL_COLS = SEG_COLS + PROS_COLS + VQ_COLS

# Column rename rule for Czech segmental (nasality_dprime -> nasal_dprime)
SEG_COL_ALIASES = {"nasality_dprime": "nasal_dprime"}

# Re-derive severity_numeric from severity_label. Czech and German source CSVs
# (built via the Vast pipeline) write severity_numeric=-1 uniformly even when
# the label is correct, so we always recompute from the label here.
SEV_NUMERIC = {
    "control": "0", "mild": "1", "moderate": "2", "severe": "3",
    "unknown": "-1", "": "-1",
}


def load_segmental():
    """Return dict (renamed_dataset, speaker_id) -> dict of segmental cols."""
    out = {}
    for path, allowed in SEG_FILES.items():
        if not path.exists():
            print(f"  SKIP missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            n = 0
            for row in r:
                if row["dataset"] not in allowed:
                    continue
                # Apply column aliases
                for old, new in SEG_COL_ALIASES.items():
                    if old in row and new not in row:
                        row[new] = row[old]
                # Apply dataset rename
                ds_renamed = DS_RENAME.get(row["dataset"], row["dataset"])
                row["dataset"] = ds_renamed
                # Re-derive severity_numeric from severity_label (the source
                # Czech/German CSVs ship -1 uniformly even when label is set)
                lab = (row.get("severity_label") or "").strip().lower()
                row["severity_numeric"] = SEV_NUMERIC.get(lab, "-1")
                row["is_control"] = "True" if lab == "control" else "False"
                key = (ds_renamed, row["speaker_id"])
                out[key] = {c: row.get(c, "") for c in SEG_COLS}
                n += 1
            print(f"  segmental {path.name}: {n} rows kept")
    return out


def load_aux(path: Path, cols, rename_map: dict, label: str):
    """Load aux CSV (prosodic / voice quality) and key by (renamed_ds, spk)."""
    if not path.exists():
        print(f"  SKIP missing {path}")
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        n = 0
        for row in r:
            ds = row.get("dataset", "")
            ds_r = rename_map.get(ds, ds)
            if ds_r not in {"Czech_OneVoice-MSD26", "German_OneVoice-MSD26",
                            "PC-GITA_OneVoice-MSD26"}:
                continue
            key = (ds_r, row["speaker_id"])
            sub = {c: row.get(c, "") for c in cols if c != "n_files_vq"}
            # rename voice-quality n_files -> n_files_vq to avoid collision
            if "n_files" in row and "n_files_vq" in cols:
                sub["n_files_vq"] = row.get("n_files", "")
            out[key] = sub
            n += 1
        print(f"  {label} {path.name}: {n} rows kept")
    return out


def main():
    print("Building MSD master CSV ...\n")

    seg = load_segmental()
    pros = {}
    for pf in PROS_FILES:
        pros.update(load_aux(pf, PROS_COLS, DS_RENAME, "prosodic"))
    vq = load_aux(VQ_FILE, VQ_COLS, DS_RENAME, "voice quality")

    print(f"\n  segmental keys: {len(seg)}")
    print(f"  prosodic keys:  {len(pros)}")
    print(f"  vq keys:        {len(vq)}")

    # Build merged rows (anchor on segmental — those are the rows worth keeping)
    merged = []
    pros_hit = 0
    vq_hit = 0
    for key, srow in seg.items():
        out_row = {c: "" for c in ALL_COLS}
        out_row.update(srow)
        if key in pros:
            out_row.update(pros[key])
            pros_hit += 1
        if key in vq:
            out_row.update(vq[key])
            vq_hit += 1
        merged.append(out_row)

    # Stats for sanity
    by_ds = defaultdict(lambda: {"n": 0, "pros": 0, "vq": 0,
                                  "sev": defaultdict(int)})
    for k, srow in seg.items():
        ds = k[0]
        by_ds[ds]["n"] += 1
        by_ds[ds]["sev"][srow.get("severity_label", "?")] += 1
        if k in pros:
            by_ds[ds]["pros"] += 1
        if k in vq:
            by_ds[ds]["vq"] += 1

    print(f"\n  prosodic hits: {pros_hit}/{len(seg)}")
    print(f"  voice-quality hits: {vq_hit}/{len(seg)}")
    print(f"\n  per-dataset breakdown:")
    for ds in sorted(by_ds.keys()):
        s = by_ds[ds]
        print(f"    {ds:<28} n={s['n']:3d}  pros={s['pros']:3d}  vq={s['vq']:3d}")
        for sev in ("control", "mild", "moderate", "severe", "unknown"):
            if sev in s["sev"]:
                print(f"      {sev:<10} {s['sev'][sev]}")

    # Write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS)
        w.writeheader()
        w.writerows(merged)
    print(f"\n  wrote {len(merged)} rows -> {OUT}")


if __name__ == "__main__":
    main()
