#!/usr/bin/env python3
"""Blind-safe audit of the Phase 3b prep inputs (A13.3). Prints COUNTS ONLY.

Checks that the claims, run JSONs, manifest and case data the prep script will use are the
held-out files the frozen judge evaluated, without showing any claim text, judge label,
ground truth or case ID. Read-only, no LLM calls, writes nothing.

Checks:
  A. claims file: rows, (row, case) per row, prompt_version, origin values
  B. run_path: under runs/heldout/ or runs/heldout_baselines/, row folder = row column,
     file stem = case_id, file exists, JSON case_id = case_id, JSON has a report + ledger,
     run file not modified after the claims file was written
  C. manifest: every master case is in it, system matches, root service matches the truth column,
     case data (metrics.parquet, inject_time.txt) present and not modified after the claims file,
     incident query (INPUT) rebuilds for every case with the judge's own function
  D. master vs claims: same (row, case) set (master lines without claims = no-report runs only)
  E. evidence the judge saw: for every parsed claim, the judge's quote check is recomputed with the
     judge's own quote_ok() from the run JSON ledger on disk + rebuilt INPUT, and must equal the
     stored quote_ok; cited evidence IDs must exist in that ledger; verified_label must follow
     from label + quote_ok. If the ledger or INPUT on disk differed from what the judge saw,
     this would fail.

Exit code 0 = all checks passed, 1 = something failed (counts printed, no values).

Example (repo root):  python scripts/audit_3b_inputs.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_grounding as jg  # noqa: E402

MASTER = "results/critic_attribution/master_cases.csv"
CLAIMS = "results/grounding_heldout/grounding_claims.csv"
ROWS = ["hybrid", "llm_only", "hybrid_7b", "code_rule", "no_critic", "react", "retrieval_only"]
BASES = ("runs/heldout/", "runs/heldout_baselines/")
LABELS = ["Supported", "Partially Supported", "Unsupported"]
FAIL = "PARSE_FAIL"
EXPECT_RC_WITH_CLAIMS = 419  # 420 judged runs minus 1 no_report (A7)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def rd(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    root = jg.REPO_ROOT
    mpath, cpath = jg.repo_path(MASTER), jg.repo_path(CLAIMS)
    for p in (mpath, cpath):
        if not p.exists():
            print(f"missing {p.relative_to(root)}")
            return 1
    claims_mtime = cpath.stat().st_mtime
    master = rd(mpath)
    claims = rd(cpath)
    manifest = jg.load_manifest(jg.repo_path(jg.DEFAULT_MANIFEST))
    print(f"judge_grounding: PROMPT_VERSION={jg.PROMPT_VERSION} "
          f"SCORING_VERSION={getattr(jg, 'SCORING_VERSION', 'MISSING')}")
    print(f"inputs: master {len(master)} lines, claims {len(claims)} lines, manifest {len(manifest)} cases")

    # ---- A. claims file
    rows = sorted({c["row"] for c in claims})
    check("A1 claims rows = the 7 held-out rows", rows == sorted(ROWS), f"{len(rows)} rows")
    rc = defaultdict(set)
    for c in claims:
        rc[c["row"]].add(c["case_id"])
    per_row = {r: len(rc[r]) for r in ROWS}
    check("A2 (row, case) with claims = 419", sum(per_row.values()) == EXPECT_RC_WITH_CLAIMS,
          " ".join(f"{r}={n}" for r, n in per_row.items()))
    pv = Counter(c["prompt_version"] for c in claims)
    check("A3 every claim prompt_version = frozen PROMPT_VERSION", set(pv) == {jg.PROMPT_VERSION},
          f"{len(pv)} distinct value(s)")
    check("A4 origin values", {c["origin"] for c in claims} <= {"structured", "explanation"})
    keys = Counter((c["row"], c["case_id"], c["claim_id"]) for c in claims)
    check("A5 no duplicate claims", max(keys.values()) == 1)

    # ---- B. run paths and run JSONs
    bad = Counter()
    runs = {}
    ckeys_all = {(c["row"], c["case_id"]) for c in claims}
    for (row, cid) in {(c["row"], c["case_id"]) for c in claims}:
        paths = {c["run_path"] for c in claims if c["row"] == row and c["case_id"] == cid}
        if len(paths) != 1:
            bad["one run_path per (row, case)"] += 1
            continue
        rp = paths.pop()
        norm = rp.replace("\\", "/")
        if not norm.startswith(BASES):
            bad["run_path under runs/heldout/ or runs/heldout_baselines/"] += 1
        parts = Path(norm).parts
        if len(parts) < 4 or parts[2] != row:
            bad["row folder in run_path = row column"] += 1
        if Path(norm).stem != cid:
            bad["run file name = case_id"] += 1
        p = Path(rp) if Path(rp).is_absolute() else root / rp
        if not p.exists():
            bad["run file exists"] += 1
            continue
        if p.stat().st_mtime > claims_mtime:
            bad["run file not modified after claims file"] += 1
        try:
            run = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            bad["run JSON readable"] += 1
            continue
        if run.get("case_id", p.stem) != cid:
            bad["run JSON case_id = case_id"] += 1
        if not run.get("report") or not isinstance(run.get("evidence_ledger"), list):
            bad["run JSON has report + ledger"] += 1
        runs[(row, cid)] = run
    for name in ["one run_path per (row, case)", "run_path under runs/heldout/ or runs/heldout_baselines/",
                 "row folder in run_path = row column", "run file name = case_id", "run file exists",
                 "run file not modified after claims file", "run JSON readable", "run JSON case_id = case_id",
                 "run JSON has report + ledger"]:
        check(f"B {name}", bad[name] == 0, f"{bad[name]} failing of {len(ckeys_all)}")
    base_by_row = defaultdict(Counter)
    for c in claims:
        base_by_row[c["row"]]["/".join(Path(c["run_path"].replace('\\', '/')).parts[:2])] += 1
    print("run folders per row: " + "; ".join(f"{r}: {', '.join(sorted(base_by_row[r]))}" for r in ROWS))

    # ---- C. manifest and case data
    mcases = {}
    for m in master:
        mcases.setdefault(m["case_id"], m)
    check("C1 master has 60 cases", len(mcases) == 60, f"{len(mcases)}")
    check("C2 every master case in manifest", all(c in manifest for c in mcases),
          f"{sum(c not in manifest for c in mcases)} missing")
    sys_alias = {"ss": "ss", "tt": "tt", "sock_shop": "ss", "train_ticket": "tt"}
    check("C3 manifest system = master system",
          all(sys_alias.get(manifest[c]["system"].lower()) == sys_alias.get(m["system"].lower())
              for c, m in mcases.items() if c in manifest), "")
    if "root_cause_service" in next(iter(manifest.values())):
        n_bad = sum(manifest[c]["root_cause_service"].strip().lower() != m["truth"].strip().lower()
                    for c, m in mcases.items() if c in manifest)
        check("C4 manifest root_cause_service = master truth", n_bad == 0, f"{n_bad} differ")
    newer, missing, qfail = 0, 0, 0
    for c in mcases:
        if c not in manifest:
            continue
        d = jg.repo_path(manifest[c]["case_dir"])
        for f in ("metrics.parquet", "inject_time.txt"):
            q = d / f
            if not q.exists():
                missing += 1
            elif q.stat().st_mtime > claims_mtime:
                newer += 1
        try:
            jg.incident_query(manifest[c])
        except Exception:  # noqa: BLE001
            qfail += 1
    check("C5 case data files present", missing == 0, f"{missing} missing")
    check("C6 case data not modified after claims file", newer == 0, f"{newer} newer")
    check("C7 INPUT rebuilds for all 60 cases", qfail == 0, f"{qfail} failed")

    # ---- D. master vs claims
    mkeys = {(m["row"], m["case_id"]) for m in master if m["row"] in ROWS}
    ckeys = {(c["row"], c["case_id"]) for c in claims}
    extra = ckeys - mkeys
    no_claims = mkeys - ckeys
    no_rep = {(m["row"], m["case_id"]) for m in master if m.get("judge_status") == "no_report"}
    check("D1 every claims (row, case) is in master", not extra, f"{len(extra)} extra")
    check("D2 master (row, case) without claims are no_report only", no_claims <= no_rep,
          f"{len(no_claims)} without claims, {len(no_claims - no_rep)} unexplained")

    # ---- E. evidence the judge saw
    n, q_mis, cite_bad, vl_bad, skipped = 0, 0, 0, 0, 0
    for c in claims:
        if c["verified_label"] == FAIL:
            skipped += 1
            continue
        run = runs.get((c["row"], c["case_id"]))
        if run is None or c["case_id"] not in manifest:
            q_mis += 1
            continue
        ledger = run.get("evidence_ledger") or []
        texts = {str(i.get("id")): f"{json.dumps(i.get('args'), sort_keys=True, default=str)} {i.get('content')}"
                 for i in ledger}
        texts["INPUT"] = jg.incident_query(manifest[c["case_id"]])
        ev = c["evidence"].split()
        q = jg.quote_ok(c["evidence_says"], ev, texts)
        n += 1
        if str(q) != c["quote_ok"]:
            q_mis += 1
        if c["match"] in ("yes", "partly") and any(e not in texts for e in ev):
            cite_bad += 1
        exp_v = "Unsupported" if str(q) == "0" else c["label"]
        if c["verified_label"] != exp_v or c["label"] not in LABELS:
            vl_bad += 1
    check("E1 quote check recomputed from disk ledger = stored quote_ok", q_mis == 0,
          f"{q_mis} mismatch of {n} parsed claims ({skipped} PARSE_FAIL skipped)")
    check("E2 cited evidence IDs exist in the ledger (yes/partly)", cite_bad == 0,
          f"{cite_bad} claims cite a missing ID (judge counts these as bad_cite; informational if > 0)")
    check("E3 verified_label = label, or Unsupported when quote_ok = 0", vl_bad == 0, f"{vl_bad} differ")

    print()
    fails = 0
    for name, ok, detail in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
        if not ok and not name.startswith("E2"):
            fails += 1
    print(f"\n{len(results) - fails}/{len(results)} checks ok" + (" (E2 is informational)" if any(
        n.startswith("E2") and not o for n, o, _ in results) else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
