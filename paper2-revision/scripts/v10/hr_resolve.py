#!/usr/bin/env python3
"""Resolve a high-risk reference (see scripts/v10/hr_spec.py) to the printed string it must equal. Values are read from
the saved outputs at full precision and rounded once."""
import json
from pathlib import Path

REV = Path(__file__).resolve().parents[2]
_J = {}


def _json(rel):
    if rel not in _J:
        _J[rel] = json.loads((REV / rel).read_text(encoding="utf-8"))
    return _J[rel]


def _dig(obj, dotted):
    for k in dotted.split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


def _fmt(fmt, x, rule):
    if rule.startswith("lt="):
        v = rule[3:]
        return v if float(x) < float(v) else f"NOT<{v}({x})"
    if rule.startswith("gt="):
        v = rule[3:]
        return v if float(x) > float(v) else f"NOT>{v}({x})"
    return fmt(x, rule)


def resolve(ref, F, fmt):
    """F: registry id -> printed string (scripts/v10/format_numbers.py); fmt: its formatter."""
    kind, rest = ref.split(":", 1)
    body, _, mod = rest.partition("~")
    if kind == "R":
        v = F[body]
        if mod == "abs":
            v = v.lstrip("-−")
        elif mod:
            raise ValueError(f"unknown modifier {mod}")
        return v, {"src": "registry", "id": body}
    if kind == "H":
        x = _dig(_json("results/v10/v10_hr_values.json")["values"], body.split(".", 1)[0])["value"]
        if "." in body:
            x = _dig(x, body.split(".", 1)[1])
        return _fmt(fmt, x, mod), {"src": "results/v10/v10_hr_values.json", "path": body}
    if kind == "X":
        x = _json("results/v10/v10_external_descriptors.json")["values"][body]["value"]
        return _fmt(fmt, x, mod), {"src": "results/v10/v10_external_descriptors.json", "path": body}
    if kind == "J":
        f, _, path = body.partition("#")
        return _fmt(fmt, _dig(_json(f), path), mod), {"src": f, "path": path}
    raise ValueError(ref)
