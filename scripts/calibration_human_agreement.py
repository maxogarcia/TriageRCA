#!/usr/bin/env python3
"""Human vs. human agreement on the grounding calibration set (Grounding Metric Spec v1.1, 7.1).

Takes two or more labeled CSVs (from the review page, one per labeler) and reports:
  - pairwise Cohen's kappa (3 labels), percent agreement, label counts, 3x3 confusion, for every pair
  - Fleiss' kappa across all labelers (with 2 labelers this equals Scott's pi, close to Cohen's)
  - agreement by claim type, and claims where labelers split
Writes every claim with all labels to a per-claim CSV.

Does not read the judge key: this is only about whether humans agree with each other.
Read-only on its inputs. Labeler names come from the file name (calibration_labeled_<name>.csv;
plain calibration_labeled.csv -> "aeon"), or pass --names in the same order as --labeled.

Note: kappa is unstable with few claims and when one label dominates; read it with % agreement.

Example:
  python scripts/calibration_human_agreement.py \\
      --labeled results/grounding_dev/calibration_labeled.csv \\
                results/grounding_dev/calibration_labeled_daniel.csv \\
      --out results/grounding_dev/calibration_human_agreement.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

LABELS = ["Supported", "Partially Supported", "Unsupported"]
SHORT = {"Supported": "S", "Partially Supported": "P", "Unsupported": "U"}
KEY = ("row", "case_id", "claim_id")


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def name_from(path: Path) -> str:
    stem = path.stem
    if stem == "calibration_labeled":
        return "aeon"
    prefix = "calibration_labeled_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def cohen(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fleiss(items: list[list[str]]) -> float | None:
    """items: one list of labels per claim, same number of raters for every claim."""
    N = len(items)
    if not N:
        return None
    n = len(items[0])
    counts = [Counter(it) for it in items]
    p_bar = sum((sum(c[l] ** 2 for l in LABELS) - n) / (n * (n - 1)) for c in counts) / N
    p_j = [sum(c[l] for c in counts) / (N * n) for l in LABELS]
    pe = sum(p ** 2 for p in p_j)
    return None if pe == 1 else (p_bar - pe) / (1 - pe)


def fmt(k: float | None) -> str:
    return "n/a (all one label)" if k is None else f"{k:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", nargs="+", required=True, help="two or more human-labeled CSVs")
    ap.add_argument("--names", nargs="+", help="labeler names, same order as --labeled")
    ap.add_argument("--out", help="per-claim CSV (default: calibration_human_agreement.csv next to the first file)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.labeled]
    if len(paths) < 2:
        sys.exit("need at least two labeled CSVs")
    names = args.names or [name_from(p) for p in paths]
    if len(names) != len(paths):
        sys.exit("--names must have one name per --labeled file")
    if len(set(names)) != len(names):
        sys.exit(f"labeler names must be unique, got {names} (use --names)")

    out = Path(args.out) if args.out else paths[0].with_name("calibration_human_agreement.csv")
    if out.resolve() in {p.resolve() for p in paths}:
        sys.exit(f"--out {out} would overwrite an input file")
    if out.name in {"calibration_agreement.csv", "calibration_key.csv"}:
        sys.exit(f"--out {out.name} is reserved for the judge scripts; pick another name")

    data: dict[str, dict[tuple, dict]] = {}
    for nm, p in zip(names, paths):
        rows = read(p)
        bad = [r.get("claim_id") for r in rows if r.get("human_label") not in LABELS]
        if bad:
            sys.exit(f"{p}: {len(bad)} claims have no valid human_label, e.g. {bad[:3]}")
        by_key = {tuple(r[k] for k in KEY): r for r in rows}
        if len(by_key) != len(rows):
            sys.exit(f"{p}: duplicate (row, case_id, claim_id) entries")
        data[nm] = by_key

    keysets = [set(d) for d in data.values()]
    common = set.intersection(*keysets)
    if any(ks != common for ks in keysets):
        for nm in names:
            extra = set(data[nm]) - common
            if extra:
                print(f"note: {nm} has {len(extra)} claims not labeled by everyone, ignored, e.g. {sorted(extra)[:2]}")
    if not common:
        sys.exit("no claims labeled by every labeler")
    keys = sorted(common, key=lambda k: (k[0], k[1], int(k[2]) if k[2].isdigit() else k[2]))
    first = data[names[0]]

    print(f"{len(keys)} claims, {len({k[:2] for k in keys})} cases, labelers: {', '.join(names)}")
    for nm in names:
        c = Counter(data[nm][k]["human_label"] for k in keys)
        print(f"   {nm:<10} " + " ".join(f"{SHORT[l]}={c[l]}" for l in LABELS))

    for a, b in combinations(names, 2):
        pairs = [(data[a][k]["human_label"], data[b][k]["human_label"]) for k in keys]
        agree = sum(x == y for x, y in pairs)
        print(f"\n== {a} vs {b}: cohen kappa={fmt(cohen(pairs))}  agreement={agree}/{len(pairs)} ({agree / len(pairs):.0%})")
        print(f"   confusion (rows={a}, cols={b})   S  P  U")
        cm = Counter(pairs)
        for h in LABELS:
            print(f"   {h:<35}" + "".join(f"{cm[(h, j)]:>3}" for j in LABELS))

    items = [[data[nm][k]["human_label"] for nm in names] for k in keys]
    all_agree = sum(len(set(it)) == 1 for it in items)
    print(f"\n== all {len(names)} labelers: fleiss kappa={fmt(fleiss(items))}  "
          f"all agree={all_agree}/{len(keys)} ({all_agree / len(keys):.0%})")

    by_t = defaultdict(list)
    for k, it in zip(keys, items):
        by_t[first[k]["claim_type"]].append(it)
    print("\n== by claim type (all labelers)")
    for t, its in sorted(by_t.items()):
        a = sum(len(set(it)) == 1 for it in its)
        print(f"   {t:<14} all agree={a}/{len(its)}  fleiss={fmt(fleiss(its))}")

    split = [(k, it) for k, it in zip(keys, items) if len(set(it)) > 1]
    print(f"\n== claims where labelers split: {len(split)}")
    for k, it in split:
        labs = " ".join(f"{nm}={SHORT[l]}" for nm, l in zip(names, it))
        print(f"   {k[1]} #{k[2]} [{first[k]['claim_type']}] {labs}")

    out_rows = []
    for k, it in zip(keys, items):
        r = {**dict(zip(KEY, k)), "claim_type": first[k]["claim_type"], "claim": first[k]["claim"]}
        for nm in names:
            src = data[nm][k]
            r[f"{nm}_label"] = src["human_label"]
            r[f"{nm}_evidence"] = src.get("human_evidence", "")
            r[f"{nm}_note"] = src.get("human_note", "")
        r["all_agree"] = int(len(set(it)) == 1)
        out_rows.append(r)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nper-claim table -> {out}")


if __name__ == "__main__":
    main()
