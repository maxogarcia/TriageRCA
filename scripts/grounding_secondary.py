#!/usr/bin/env python3
"""Secondary grounding numbers from a finished judge run (Spec Addendum A, sections A4 and A10).

Reads grounding_cases.csv and grounding_claims.csv written by judge_grounding.py and adds:
  - supported_rate_expl (A4): verified Supported explanation claims / explanation claims, per case.
  - bounds for parse_fail_label cases (A10): claims that parsed keep their verified label, failed
    explanation claims count as Unsupported (lower) or Supported (upper). Done for grounding_expl
    and supported_rate_expl. ok cases have lower = upper = their actual value.

Primary numbers (ok cases only) are recomputed from the claim rows and checked against the judge's
grounding_expl per case; the script stops if any case differs. No LLM calls; read-only on the judge
CSVs; writes only grounding_secondary_cases.csv and grounding_secondary_summary.csv in --save-dir.

Example:
  python scripts/grounding_secondary.py --save-dir results/grounding_heldout
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

WEIGHT = {"Supported": 1.0, "Partially Supported": 0.5, "Unsupported": 0.0}
FAIL = "PARSE_FAIL"
OUT_CASES = "grounding_secondary_cases.csv"
OUT_SUMMARY = "grounding_secondary_summary.csv"
TOL = 1e-4  # judge rounds per-case scores to 4 decimals


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path} (run judge_grounding.py first)")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]):
    return round(statistics.mean(xs), 4) if xs else ""


def per_case(case: dict, claims: list[dict]) -> dict:
    out = {k: case[k] for k in ("row", "case_id", "system", "fault", "status")}
    expl = [c for c in claims if c["origin"] == "explanation"]
    parsed = [c for c in expl if c["verified_label"] in WEIGHT]
    failed = [c for c in expl if c["verified_label"] == FAIL]
    unknown = [c for c in expl if c["verified_label"] not in WEIGHT and c["verified_label"] != FAIL]
    if unknown:
        sys.exit(f"{case['row']} {case['case_id']}: unexpected verified_label {unknown[0]['verified_label']!r}")
    n = len(expl)
    n_fail_all = sum(c["verified_label"] == FAIL for c in claims)
    out.update({"n_claims_expl": n, "n_fail_expl": len(failed), "n_fail_all": n_fail_all,
                "grounding_expl": "", "supported_rate_expl": "",
                "grounding_expl_lo": "", "grounding_expl_hi": "",
                "supported_rate_expl_lo": "", "supported_rate_expl_hi": ""})
    if case["status"] not in ("ok", "parse_fail_label") or n == 0:
        return out
    g = sum(WEIGHT[c["verified_label"]] for c in parsed)
    s = sum(c["verified_label"] == "Supported" for c in parsed)
    k = len(failed)
    if case["status"] == "ok":
        if k:
            sys.exit(f"{case['row']} {case['case_id']}: status ok but {k} PARSE_FAIL claims")
        ge, sr = round(g / n, 4), round(s / n, 4)
        judge = case.get("grounding_expl", "")
        if judge == "" or abs(float(judge) - ge) > TOL:
            sys.exit(f"{case['row']} {case['case_id']}: recomputed grounding_expl {ge} != judge {judge!r}; "
                     "claims CSV and cases CSV do not match, nothing written")
        out.update({"grounding_expl": ge, "supported_rate_expl": sr,
                    "grounding_expl_lo": ge, "grounding_expl_hi": ge,
                    "supported_rate_expl_lo": sr, "supported_rate_expl_hi": sr})
    else:
        if n_fail_all == 0:
            sys.exit(f"{case['row']} {case['case_id']}: parse_fail_label but no PARSE_FAIL claim rows")
        out.update({"grounding_expl_lo": round(g / n, 4), "grounding_expl_hi": round((g + k) / n, 4),
                    "supported_rate_expl_lo": round(s / n, 4), "supported_rate_expl_hi": round((s + k) / n, 4)})
    return out


def summarize(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[(r["row"], r["system"])].append(r)
        by[(r["row"], "ALL")].append(r)
    out = []
    for (row, system), rs in sorted(by.items()):
        ok = [r for r in rs if r["grounding_expl"] != ""]
        bnd = [r for r in rs if r["grounding_expl_lo"] != ""]
        st = Counter(r["status"] for r in rs)
        out.append({
            "row": row, "system": system, "n_cases": len(rs), "n_ok": st["ok"],
            "n_parse_fail_label": st["parse_fail_label"],
            "n_other_fail": len(rs) - st["ok"] - st["parse_fail_label"],
            "n_scored_expl": len(ok), "n_bounded_expl": len(bnd),
            "mean_expl": mean([r["grounding_expl"] for r in ok]),
            "mean_expl_lo": mean([r["grounding_expl_lo"] for r in bnd]),
            "mean_expl_hi": mean([r["grounding_expl_hi"] for r in bnd]),
            "mean_supported_rate": mean([r["supported_rate_expl"] for r in ok]),
            "mean_supported_rate_lo": mean([r["supported_rate_expl_lo"] for r in bnd]),
            "mean_supported_rate_hi": mean([r["supported_rate_expl_hi"] for r in bnd]),
        })
    return out


def write(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-dir", required=True, help="judge output dir with grounding_cases.csv / grounding_claims.csv")
    args = ap.parse_args()
    d = Path(args.save_dir)
    cases = read(d / "grounding_cases.csv")
    claims = read(d / "grounding_claims.csv")
    if not cases:
        sys.exit("grounding_cases.csv is empty")
    versions = {c.get("scoring_version", "") for c in cases}
    if versions != {"expl-primary-v1"}:
        sys.exit(f"expected scoring_version expl-primary-v1, found {sorted(versions)}")

    by_case = defaultdict(list)
    for c in claims:
        by_case[(c["row"], c["case_id"])].append(c)
    keys = [(c["row"], c["case_id"]) for c in cases]
    if len(set(keys)) != len(keys):
        sys.exit("duplicate (row, case_id) in grounding_cases.csv")
    orphan = set(by_case) - set(keys)
    if orphan:
        sys.exit(f"{len(orphan)} claim groups have no case row, e.g. {sorted(orphan)[:2]}")

    rows = [per_case(c, by_case.get((c["row"], c["case_id"]), [])) for c in cases]
    summary = summarize(rows)

    # cross-check vs the judge's own summary means (same rounding chain)
    judge_summary = d / "grounding_summary.csv"
    if judge_summary.exists():
        js = {(r["row"], r["system"]): r for r in read(judge_summary)}
        for s in summary:
            j = js.get((s["row"], s["system"]))
            if j and j.get("mean_expl", "") != "" and s["mean_expl"] != "":
                if abs(float(j["mean_expl"]) - s["mean_expl"]) > TOL:
                    sys.exit(f"{s['row']} {s['system']}: mean_expl {s['mean_expl']} != judge summary {j['mean_expl']}")
        print("check: per-case and summary mean_expl match the judge")

    write(d / OUT_CASES, rows)
    write(d / OUT_SUMMARY, summary)

    fail_claims = [c for c in claims if c["verified_label"] == FAIL]
    print(f"{len(rows)} cases, {len(fail_claims)} PARSE_FAIL claims "
          f"({sum(c['origin'] == 'explanation' for c in fail_claims)} explanation, "
          f"{sum(c['origin'] == 'structured' for c in fail_claims)} structured); "
          f"by system: {dict(Counter(c['system'] for c in fail_claims))}")
    print(f"\n{'row':<16}{'ok':>4}{'fail':>5}  {'expl':>6} [{'lo':>6} {'hi':>6}]   {'S-rate':>6} [{'lo':>6} {'hi':>6}]")
    for s in summary:
        if s["system"] == "ALL":
            print(f"{s['row']:<16}{s['n_ok']:>4}{s['n_parse_fail_label']:>5}  {s['mean_expl']!s:>6} "
                  f"[{s['mean_expl_lo']!s:>6} {s['mean_expl_hi']!s:>6}]   {s['mean_supported_rate']!s:>6} "
                  f"[{s['mean_supported_rate_lo']!s:>6} {s['mean_supported_rate_hi']!s:>6}]")
    print(f"\n-> {d / OUT_CASES}\n-> {d / OUT_SUMMARY}")


if __name__ == "__main__":
    main()
