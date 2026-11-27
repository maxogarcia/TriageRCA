#!/usr/bin/env python3
"""Round-1 query targets (post hoc, descriptive, Addendum A15.4).

Read-only. No LLM calls. For each case: which services did the round-1 Executor query (no_critic ledger,
= round 1 of every critic row), did it ever query E001's top service, and was its first query the
service it then named as root?

Inputs:  --cases results/diagnosis/cases.csv (from scripts/diagnosis.py)
         --runs  runs/heldout runs/heldout_baselines
Output:  --out   results/diagnosis_r1 (must be new or empty)

Example: python scripts/diagnosis_r1_queries.py --out results/diagnosis_r1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SCREEN_CAT = "screen (= H_E001a)"


def die(m):
    sys.exit(f"ERROR: {m}")


def norm(s):
    return str(s).strip().lower() if s is not None else ""


def find_json(run_dirs, row, c):
    hits = [p for d in run_dirs for p in (d / row).glob(f"*/{c}.json") if p.parent.name != "llm_cache"]
    if len(hits) != 1:
        die(f"expected 1 JSON for {row}/{c}, found {len(hits)}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="results/diagnosis/cases.csv")
    ap.add_argument("--runs", nargs="+", default=["runs/heldout", "runs/heldout_baselines"])
    ap.add_argument("--out", default="results/diagnosis_r1")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    p = Path(a.cases)
    if not p.exists():
        die(f"missing {p}")
    cases = list(csv.DictReader(open(p, newline="")))
    run_dirs = [Path(x) for x in a.runs]

    rows, bad = [], []
    for r in cases:
        c = r["case_id"]
        d = json.loads(find_json(run_dirs, "no_critic", c).read_text())
        led = d.get("evidence_ledger") or []
        if not led or led[0].get("id") != "E001":
            die(f"no_critic/{c}: first item is not E001")
        hyps = d.get("hypotheses") or []
        root = ((hyps[0].get("ranked_services") or [""])[0]) if hyps else ""
        if norm(root) != norm(r["r1_root"]):
            bad.append(f"{c}: json {root!r} vs cases.csv {r['r1_root']!r}")
        q = [(it.get("source"), norm((it.get("args") or {}).get("service"))) for it in led[1:]
             if (it.get("args") or {}).get("service")]
        first_metric = next((s for src, s in q if src == "metric_query"), "")
        rows.append({
            "case_id": c, "system": r["system"], "group": r["group"], "r1_category": r["r1_category"],
            "r1_root": root, "screen": r["screen"], "truth": r["truth"],
            "n_queries": len(q), "n_services_queried": len({s for _, s in q}),
            "screen_queried": norm(r["screen"]) in {s for _, s in q},
            "root_queried": norm(root) in {s for _, s in q} if root else False,
            "first_query_is_root": bool(q) and bool(root) and q[0][1] == norm(root),
            "first_metric_is_root": bool(first_metric) and first_metric == norm(root),
            "first_query_service": q[0][1] if q else "",
            "services_queried": " ".join(dict.fromkeys(s for _, s in q))})
    if bad:
        die(f"round-1 root mismatch in {len(bad)} cases, e.g. {bad[:3]}")
    n_scr = sum(r["r1_category"] == SCREEN_CAT for r in rows)
    print(f"checks: {len(rows)} cases (expect 60), screen {n_scr} / non-screen {len(rows) - n_scr} (expect 15 / 45), "
          f"round-1 roots match cases.csv")
    if len(rows) != 60 or n_scr != 15:
        die("case count or 15/45 split differs; nothing written")

    keys = ["screen_queried", "root_queried", "first_query_is_root", "first_metric_is_root"]
    sets = defaultdict(list)
    for r in rows:
        sets["non-screen (45)" if r["r1_category"] != SCREEN_CAT else "screen (15)"].append(r)
        sets[f"cat: {r['r1_category']}"].append(r)
    summ_rows = []
    for name in ["non-screen (45)", "screen (15)"] + sorted(k for k in sets if k.startswith("cat:")):
        rs = sets[name]
        line = {"set": name, "n": len(rs), **{k: sum(x[k] for x in rs) for k in keys},
                "mean_services_queried": round(sum(x["n_services_queried"] for x in rs) / len(rs), 2)}
        summ_rows.append(line)
    L = ["# Round-1 query targets (post hoc, descriptive, A15.4)", "",
         "| set | n | screen queried | root queried | first query = root | first metric_query = root | mean services queried |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summ_rows:
        L.append(f"| {s['set']} | {s['n']} | {s['screen_queried']} | {s['root_queried']} | {s['first_query_is_root']} | "
                 f"{s['first_metric_is_root']} | {s['mean_services_queried']} |")
    summ = "\n".join(L) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("r1_queries_cases.csv", rows), ("r1_queries_summary.csv", summ_rows)]:
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    (out / "summary.md").write_text(summ)
    print("\n" + summ + f"\nwrote {out}/ (2 CSVs + summary.md)")


if __name__ == "__main__":
    main()
