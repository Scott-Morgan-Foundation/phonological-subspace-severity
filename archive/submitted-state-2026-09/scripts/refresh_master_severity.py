#!/usr/bin/env python3
"""
Refresh severity columns in track4_master.csv from speaker_inventory.csv.

Why: master CSV (Apr 17 Windows / Apr 19 DGX) is older than the inventory
(Apr 24), which now reflects Lívia Ivaskó's Hungarian severity labels (17
of 47 Hungarian speakers labelled). Master still shows the pre-batch state
(13 labelled). Other corpora may also have updated labels via Track 4
pseudo-labelling. d-prime features depend on MFA alignments and do NOT
change — refreshing them would require a full HuBERT extraction re-run.

What this script does:
  - Reads speaker_inventory.csv (authoritative severity source).
  - Reads existing track4_master.csv (preserves all d-prime + prosodic
    + voice quality columns verbatim).
  - For each (dataset, speaker_id) pair in master, looks up the inventory
    and overwrites only: severity_label, severity_numeric, is_control.
  - Reports which rows changed (dataset, speaker, old → new).
  - Writes refreshed master to <input>.refreshed.csv (caller renames).

Usage (Windows, paper2 master):
    python papers/Published papers/Paper 2/scripts/refresh_master_severity.py \
        --inventory <path-to-inventory> \
        --master    papers/Published papers/Paper 2/results/track4_master.csv

By default, fetches the latest inventory from DGX over SSH if --inventory
is omitted.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

DGX_HOST = "<user>@<workstation-host>"
DGX_KEY = "<path-to-ssh-key>"
DGX_INVENTORY = "<remote-dysarthria-root>/results/track4/speaker_inventory.csv"

SEV_NUMERIC = {"control": 0, "mild": 1, "moderate": 2, "severe": 3,
               "unknown": -1, "": -1}


def fetch_inventory_from_dgx(dest: Path) -> Path:
    print(f"  scp from DGX: {DGX_INVENTORY} -> {dest}")
    cmd = ["scp", "-i", DGX_KEY, "-o", "StrictHostKeyChecking=no",
           f"{DGX_HOST}:{DGX_INVENTORY}", str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def load_inventory(inv_path: Path) -> dict:
    """Return {(dataset, speaker_id): row_dict}."""
    out = {}
    with open(inv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["speaker_id"])
            out[key] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=None,
                    help="speaker_inventory.csv (defaults to fetching from DGX)")
    ap.add_argument("--master", type=Path, required=True,
                    help="track4_master.csv to refresh")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <master>.refreshed.csv)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite master directly (creates .bak alongside)")
    args = ap.parse_args()

    if args.inventory is None:
        tmp = args.master.parent / "speaker_inventory.dgx.csv"
        args.inventory = fetch_inventory_from_dgx(tmp)

    if not args.master.exists():
        sys.exit(f"master not found: {args.master}")
    if not args.inventory.exists():
        sys.exit(f"inventory not found: {args.inventory}")

    print(f"  inventory: {args.inventory}")
    print(f"  master:    {args.master}")

    inv = load_inventory(args.inventory)
    print(f"  inventory rows: {len(inv)}")

    # Read full master, update severity columns in place
    with open(args.master, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        rows = list(r)

    required_cols = {"dataset", "speaker_id", "severity_label",
                     "severity_numeric", "is_control"}
    missing = required_cols - set(fieldnames)
    if missing:
        sys.exit(f"master missing required columns: {missing}")

    REAL_LABELS = {"control", "mild", "moderate", "severe"}

    changes = []          # applied
    conflicts = []        # real -> different real, NOT applied (need --allow-relabel)
    skipped_unknown = []  # inventory has 'unknown' but master has real label, kept master
    no_inv = []
    for row in rows:
        key = (row["dataset"], row["speaker_id"])
        inv_row = inv.get(key)
        if inv_row is None:
            no_inv.append(key)
            continue
        new_label = (inv_row.get("severity_label") or "unknown").strip() or "unknown"
        new_numeric = inv_row.get("severity_numeric", "")
        if new_numeric in (None, ""):
            new_numeric = SEV_NUMERIC.get(new_label, -1)
        new_is_control = inv_row.get("is_control", "")
        if new_is_control in (None, ""):
            new_is_control = "True" if new_label == "control" else "False"

        old_label = (row.get("severity_label") or "").strip()
        old_numeric = row.get("severity_numeric", "")
        old_is_control = row.get("is_control", "")

        old_is_real = old_label in REAL_LABELS
        new_is_real = new_label in REAL_LABELS

        # Policy:
        #   - If new is real and old is unknown/empty       -> APPLY (upgrade)
        #   - If new is real and old is the same real label -> normalize numeric/is_control if needed
        #   - If new is real and old is a DIFFERENT real    -> CONFLICT, don't apply
        #   - If new is unknown                              -> SKIP (preserve master)
        if not new_is_real:
            if old_is_real:
                skipped_unknown.append((row["dataset"], row["speaker_id"], old_label))
            continue

        if not old_is_real:
            # Upgrade unknown -> real
            changes.append((row["dataset"], row["speaker_id"],
                            f"{old_label or 'unknown'}/{old_numeric}/{old_is_control}",
                            f"{new_label}/{new_numeric}/{new_is_control}",
                            "upgrade"))
            row["severity_label"] = new_label
            row["severity_numeric"] = str(new_numeric)
            row["is_control"] = str(new_is_control)
            continue

        if old_label == new_label:
            # Same real label — normalize numeric/is_control if they drifted
            if (str(old_numeric) != str(new_numeric) or
                    str(old_is_control) != str(new_is_control)):
                changes.append((row["dataset"], row["speaker_id"],
                                f"{old_label}/{old_numeric}/{old_is_control}",
                                f"{new_label}/{new_numeric}/{new_is_control}",
                                "normalize"))
                row["severity_numeric"] = str(new_numeric)
                row["is_control"] = str(new_is_control)
            continue

        # Real -> different real: conflict
        conflicts.append((row["dataset"], row["speaker_id"], old_label, new_label))

    print(f"\n  rows in master:           {len(rows)}")
    print(f"  applied changes:          {len(changes)}")
    print(f"  conflicts (NOT applied):  {len(conflicts)}")
    print(f"  inv-says-unknown skipped: {len(skipped_unknown)} (kept master's real label)")
    print(f"  no inv lookup:            {len(no_inv)}")

    by_ds = Counter(c[0] for c in changes)
    if by_ds:
        print("\n  applied changes by dataset:")
        for ds, n in sorted(by_ds.items(), key=lambda x: -x[1]):
            print(f"    {ds:<28} {n:>4}")

    if changes:
        print("\n  sample applied changes (first 20):")
        for c in changes[:20]:
            ds, spk, old, new, kind = c
            print(f"    [{kind:<9}] {ds:<28} {spk:<20} {old}  ->  {new}")

    if conflicts:
        by_ds_c = Counter(c[0] for c in conflicts)
        print("\n  conflicts (real -> different real, NOT applied) by dataset:")
        for ds, n in sorted(by_ds_c.items(), key=lambda x: -x[1]):
            print(f"    {ds:<28} {n:>4}")
        print("  sample conflicts (first 10):")
        for ds, spk, old, new in conflicts[:10]:
            print(f"    {ds:<28} {spk:<20} master={old:<10} inv={new}")

    if skipped_unknown:
        by_ds_s = Counter(s[0] for s in skipped_unknown)
        print("\n  inv-unknown-skipped by dataset (master kept its real label):")
        for ds, n in sorted(by_ds_s.items(), key=lambda x: -x[1])[:15]:
            print(f"    {ds:<28} {n:>4}")

    if no_inv:
        ds_no = Counter(k[0] for k in no_inv)
        print("\n  master rows with no inventory match (kept unchanged):")
        for ds, n in sorted(ds_no.items(), key=lambda x: -x[1])[:15]:
            print(f"    {ds:<28} {n:>4}")

    # Write output
    if args.in_place:
        bak = args.master.with_suffix(args.master.suffix + ".bak")
        shutil.copy2(args.master, bak)
        out_path = args.master
        print(f"\n  backup: {bak}")
    else:
        out_path = args.out or args.master.with_suffix(".refreshed.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote: {out_path}")


if __name__ == "__main__":
    main()
