#!/usr/bin/env python3
"""AC@1 x grounding 2x2 per row (Spec v1.1 4.2, Addendum A12). Descriptive only, no tests.

High grounding = grounding_expl >= 0.75 (fixed in A12, not a CLI option). Primary table uses ok
cases with a score; excluded cases are counted per row by reason. A10 check: every
parse_fail_label case with explanation claims is classified at its lower and upper bound
(high at both / low at both / undetermined), split by AC@1.

Inputs (read-only):
  --ac        one or more scores_cases.csv from score_ablation.py (columns row, case_id, system, correct)
  --save-dir  judge dir holding grounding_secondary_cases.csv (from grounding_secondary.py)
Writes grounding_2x2_cases.csv and grounding_2x2_summary.csv into --save-dir.

The script stops (writes nothing) if a (row, case_id) is on one side only, appears twice,
has a different system on the two sides, or has an unexpected value.

Example:
  python scripts/grounding_2x2.py --ac runs/heldout/scores_cases.csv runs/heldout_baselines/scores_cases.csv \\
      --save-dir results/grounding_heldout
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

THRESHOLD = 0.75          # A12: fixed before the join
EPS = 1e-9                # scores are rounded to 4 decimals; guard float compare
IN_GROUNDING = "grounding_secondary_cases.csv"
OUT_CASES = "grounding_2x2_cases.csv"
OUT_SUMMARY = "grounding_2x2_summary.csv"
ROW_ORDER = ["react", "retrieval_only", "no_critic", "code_rule", "hybrid_7b", "llm_only", "hybrid"]
STATUSES = {"ok", "parse_fail_label", "no_report"}  # anything else stops the script


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty")
    return rows


def need(rows: list[dict], cols: list[str], path) -> None:
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing columns {missing}")


def index(rows: list[dict], path) -> dict:
    out = {}
    for r in rows:
        k = (r["row"], r["case_id"])
        if k in out:
            sys.exit(f"duplicate (row, case_id) {k} in {path}")
        out[k] = r
    return out


def high(x: str) -> bool:
    return float(x) >= THRESHOLD - EPS


def classify(g: dict) -> tuple[str, str, str]:
    """-> (reason, grounding level for primary or '', A10 class or '')."""
    st = g["status"]
    if st not in STATUSES:
        sys.exit(f"{g['row']} {g['case_id']}: unexpected status {st!r}")
    if st == "ok":
        if g["grounding_expl"] == "":
            if int(g["n_claims_expl"]) != 0:
                sys.exit(f"{g['row']} {g['case_id']}: ok with explanation claims but blank grounding_expl")
            return "zero_expl_claims", "", ""
        return "scored", "high" if high(g["grounding_expl"]) else "low", ""
    if st == "parse_fail_label":
        lo, hi = g["grounding_expl_lo"], g["grounding_expl_hi"]
        if lo == "" or hi == "":
            return "parse_fail_label_no_expl", "", ""
        if float(lo) > float(hi) + EPS:
            sys.exit(f"{g['row']} {g['case_id']}: lower bound {lo} > upper bound {hi}")
        a10 = "high_both" if high(lo) else ("low_both" if not high(hi) else "undetermined")
        return "parse_fail_label", "", a10
    return "no_report", "", ""


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ac", nargs="+", required=True, help="scores_cases.csv file(s) with per-case AC@1")
    ap.add_argument("--save-dir", required=True, help="dir with grounding_secondary_cases.csv; outputs go here")
    args = ap.parse_args()
    d = Path(args.save_dir)

    ac_rows = []
    for p in args.ac:
        rs = read(Path(p))
        need(rs, ["row", "case_id", "system", "correct"], p)
        ac_rows += rs
    ac = index(ac_rows, "the --ac files (combined)")
    g_path = d / IN_GROUNDING
    g_rows = read(g_path)
    need(g_rows, ["row", "case_id", "system", "status", "n_claims_expl", "grounding_expl",
                  "grounding_expl_lo", "grounding_expl_hi"], g_path)
    gr = index(g_rows, g_path)

    only_ac, only_g = sorted(set(ac) - set(gr)), sorted(set(gr) - set(ac))
    if only_ac or only_g:
        sys.exit(f"case sets differ: {len(only_ac)} only in AC (e.g. {only_ac[:2]}), "
                 f"{len(only_g)} only in grounding (e.g. {only_g[:2]}); nothing written")

    cases = []
    for k in sorted(gr):
        a, g = ac[k], gr[k]
        if a["system"] != g["system"]:
            sys.exit(f"{k}: system {a['system']!r} in AC vs {g['system']!r} in grounding")
        if a["correct"] not in ("True", "False"):
            sys.exit(f"{k}: unexpected correct value {a['correct']!r}")
        reason, level, a10 = classify(g)
        cases.append({"row": k[0], "case_id": k[1], "system": g["system"],
                      "ac1_correct": a["correct"] == "True", "status": g["status"],
                      "n_claims_expl": g["n_claims_expl"], "grounding_expl": g["grounding_expl"],
                      "grounding_expl_lo": g["grounding_expl_lo"], "grounding_expl_hi": g["grounding_expl_hi"],
                      "reason": reason, "grounding_level": level, "a10_class": a10})

    by_row = defaultdict(list)
    for c in cases:
        by_row[c["row"]].append(c)
    order = [r for r in ROW_ORDER if r in by_row] + sorted(r for r in by_row if r not in ROW_ORDER)

    summary = []
    for row in order:
        cs = by_row[row]
        sc = [c for c in cs if c["reason"] == "scored"]
        cell = Counter((c["ac1_correct"], c["grounding_level"]) for c in sc)
        rsn = Counter(c["reason"] for c in cs)
        a10 = Counter((c["ac1_correct"], c["a10_class"]) for c in cs if c["a10_class"])
        n = len(sc)
        s = {"row": row, "n_cases": len(cs), "n_scored": n,
             "excl_parse_fail_label": rsn["parse_fail_label"],
             "excl_parse_fail_label_no_expl": rsn["parse_fail_label_no_expl"],
             "excl_zero_expl_claims": rsn["zero_expl_claims"], "excl_no_report": rsn["no_report"],
             "correct_high": cell[(True, "high")], "correct_low": cell[(True, "low")],
             "wrong_high": cell[(False, "high")], "wrong_low": cell[(False, "low")]}
        for q in ("correct_high", "correct_low", "wrong_high", "wrong_low"):
            s[f"{q}_pct"] = pct(s[q], n)
        for ok_ in (True, False):
            tag = "correct" if ok_ else "wrong"
            for cl in ("high_both", "low_both", "undetermined"):
                s[f"a10_{tag}_{cl}"] = a10[(ok_, cl)]
        if s["correct_high"] + s["correct_low"] + s["wrong_high"] + s["wrong_low"] != n:
            sys.exit(f"{row}: cells do not add up to n_scored")
        if n + sum(rsn[r] for r in rsn if r != "scored") != len(cs):
            sys.exit(f"{row}: scored + excluded != n_cases")
        summary.append(s)

    for path, rows in ((d / OUT_CASES, cases), (d / OUT_SUMMARY, summary)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"AC@1 x grounding 2x2 (A12): high = grounding_expl >= {THRESHOLD}, ok cases only. "
          f"{len(cases)} cases joined.")
    print("C = AC@1 correct, W = wrong; H/L = high/low grounding; % of scored cases in the row\n")
    print(f"{'row':<15}{'n':>4}{'scored':>7}  {'C+H':>9}{'C+L':>9}{'W+H':>9}{'W+L':>9}   excluded (pf / pf0 / zero / no_rep)")
    for s in summary:
        q = lambda k: f"{s[k]} {s[k + '_pct']:>4}"
        print(f"{s['row']:<15}{s['n_cases']:>4}{s['n_scored']:>7}  {q('correct_high'):>9}{q('correct_low'):>9}"
              f"{q('wrong_high'):>9}{q('wrong_low'):>9}   {s['excl_parse_fail_label']} / "
              f"{s['excl_parse_fail_label_no_expl']} / {s['excl_zero_expl_claims']} / {s['excl_no_report']}")
    print("\nA10 check, parse_fail_label cases at their bounds (high both / low both / undetermined):")
    for s in summary:
        if s["excl_parse_fail_label"]:
            print(f"  {s['row']:<15} correct {s['a10_correct_high_both']}/{s['a10_correct_low_both']}/"
                  f"{s['a10_correct_undetermined']}   wrong {s['a10_wrong_high_both']}/"
                  f"{s['a10_wrong_low_both']}/{s['a10_wrong_undetermined']}")
    print(f"\n-> {d / OUT_CASES}\n-> {d / OUT_SUMMARY}")


if __name__ == "__main__":
    main()
