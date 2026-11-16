#!/usr/bin/env python3
"""Paired significance tests for grounding (Spec Addendum A, section A11).

Reads grounding_secondary_cases.csv (from grounding_secondary.py) and, for each comparison, runs a
two-sided paired Wilcoxon signed-rank test over the cases scored in both rows (matched by case_id):
  Family 1 (vs react):     hybrid, hybrid_7b, llm_only, code_rule, no_critic
  Family 2 (vs no_critic): hybrid, hybrid_7b, llm_only, code_rule
Holm correction within each family. Primary metric grounding_expl; secondary supported_rate_expl
(Holm separately). Zero differences are dropped. Exact p-value when there are no ties among |d| and
at most 50 nonzero pairs; otherwise normal approximation with tie correction (no continuity
correction). Pure Python, no scipy needed. Read-only; writes grounding_tests.csv in --save-dir.

Example:
  python scripts/grounding_tests.py --save-dir results/grounding_heldout
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

FAMILIES = {
    "vs_react": ("react", ["hybrid", "hybrid_7b", "llm_only", "code_rule", "no_critic"]),
    "vs_no_critic": ("no_critic", ["hybrid", "hybrid_7b", "llm_only", "code_rule"]),
}
METRICS = [("grounding_expl", "primary"), ("supported_rate_expl", "secondary")]
ALPHA = 0.05
OUT = "grounding_tests.csv"


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def exact_p(w_plus: float, n: int) -> float:
    """Two-sided exact p for W+ with ranks 1..n (no ties)."""
    counts = [1] + [0] * (n * (n + 1) // 2)
    for k in range(1, n + 1):
        for s in range(len(counts) - 1, k - 1, -1):
            counts[s] += counts[s - k]
    total = 2 ** n
    w = int(round(w_plus))
    lo = sum(counts[: w + 1]) / total
    hi = sum(counts[w:]) / total
    return min(1.0, 2 * min(lo, hi))


def wilcoxon(d: list[float]) -> tuple[float | None, float | None, str, int]:
    nz = [x for x in d if x != 0]
    n = len(nz)
    if n == 0:
        return None, None, "all zero", 0
    a = [abs(x) for x in nz]
    r = ranks(a)
    w_plus = sum(ri for ri, x in zip(r, nz) if x > 0)
    ties = Counter(a)
    if n <= 50 and all(c == 1 for c in ties.values()):
        return w_plus, exact_p(w_plus, n), "exact", n
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
    if var <= 0:
        return w_plus, None, "no variance", n
    z = (w_plus - mean) / math.sqrt(var)
    return w_plus, math.erfc(abs(z) / math.sqrt(2)), "normal", n


def holm(ps: list[float | None]) -> list[float | None]:
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    m = len(idx)
    adj: list[float | None] = [None] * len(ps)
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-dir", required=True, help="dir with grounding_secondary_cases.csv")
    args = ap.parse_args()
    d = Path(args.save_dir)
    src = d / "grounding_secondary_cases.csv"
    if not src.exists():
        sys.exit(f"missing {src} (run grounding_secondary.py first)")
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, dict[str, dict]] = {}
    for r in rows:
        data.setdefault(r["row"], {})
        if r["case_id"] in data[r["row"]]:
            sys.exit(f"duplicate case {r['case_id']} in row {r['row']}")
        data[r["row"]][r["case_id"]] = r
    needed = {ref for ref, _ in FAMILIES.values()} | {x for _, xs in FAMILIES.values() for x in xs}
    missing = sorted(needed - set(data))
    if missing:
        sys.exit(f"rows not found in {src.name}: {missing}")

    out = []
    for metric, role in METRICS:
        for fam, (ref, others) in FAMILIES.items():
            res = []
            for row in others:
                common = sorted(set(data[row]) & set(data[ref]))
                pairs = [(float(data[row][c][metric]), float(data[ref][c][metric])) for c in common
                         if data[row][c][metric] != "" and data[ref][c][metric] != ""]
                diffs = [a - b for a, b in pairs]
                w, p, method, n_nz = wilcoxon(diffs)
                res.append({"metric": metric, "role": role, "family": fam, "row": row, "reference": ref,
                            "n_pairs": len(diffs), "n_zero": len(diffs) - n_nz,
                            "n_row_higher": sum(x > 0 for x in diffs), "n_ref_higher": sum(x < 0 for x in diffs),
                            "mean_row": round(statistics.mean(a for a, _ in pairs), 4) if pairs else "",
                            "mean_ref": round(statistics.mean(b for _, b in pairs), 4) if pairs else "",
                            "mean_diff": round(statistics.mean(diffs), 4) if diffs else "",
                            "median_diff": round(statistics.median(diffs), 4) if diffs else "",
                            "W_plus": w if w is not None else "", "method": method,
                            "p": round(p, 5) if p is not None else ""})
            adj = holm([r["p"] if r["p"] != "" else None for r in res])
            for r, a in zip(res, adj):
                r["p_holm"] = round(a, 5) if a is not None else ""
                r["significant"] = int(a is not None and a < ALPHA)
            out += res

    with open(d / OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    for metric, role in METRICS:
        print(f"\n== {metric} ({role}); Holm within family, alpha {ALPHA}")
        print(f"   {'row':<10} {'ref':<10} {'pairs':>5} {'zero':>4} {'+':>3} {'-':>3} "
              f"{'mean_diff':>9} {'med_diff':>8} {'p':>8} {'p_holm':>8}  method")
        for r in out:
            if r["metric"] == metric:
                star = " *" if r["significant"] else ""
                print(f"   {r['row']:<10} {r['reference']:<10} {r['n_pairs']:>5} {r['n_zero']:>4} "
                      f"{r['n_row_higher']:>3} {r['n_ref_higher']:>3} {r['mean_diff']!s:>9} {r['median_diff']!s:>8} "
                      f"{r['p']!s:>8} {r['p_holm']!s:>8}  {r['method']}{star}")
    print(f"\n-> {d / OUT}")


if __name__ == "__main__":
    main()
