#!/usr/bin/env python3
"""Critic revisions + budget (post hoc, descriptive, Addendum A13.1).

Read-only. No LLM calls. Reads the saved held-out runs and the Phase 1 master table and writes
CSVs + summary.md to a NEW --out.

Questions (fix plan Phase 2):
  - Do revision rounds gather new evidence, or only reword / follow the onset hint?
  - What does each row cost (rounds, tool calls, LLM calls, tokens, LLM time)?
  - Why does the code rule skip (code_rule skip cases)?

Inputs (repo-relative defaults):
  --runs      experiment folders (default runs/heldout runs/heldout_baselines), layout as in Phase 1
  --master    results/critic_attribution/master_cases.csv (Phase 1 source of truth)
  --manifest  data/manifests/sample_manifest.csv (optional: service list per case from metrics.parquet)

Step 0 (before any number): every (row, case) is rebuilt from the raw JSON and compared with
master_cases.csv field by field; AC@1 per row and the known Phase 1 counts are checked. Any
mismatch stops the script with nothing written.

Step 1: structure report (roles seen, verdict fields, where onset hints appear, round split
calibration). Printed and saved as structure.txt so assumptions can be audited.

Outputs (--out, must be new or empty):
  r1_identity.csv        one line per (critic row, case): round 1 equals no_critic?
  revision_rounds.csv    one line per (critic row, case, round >= 2)
  revision_cases.csv     one line per (critic row, case)
  revision_by_row.csv    per-row summary of rounds >= 2
  hint_by_row.csv        onset-hint following per row
  pushed_back_wrong.csv  correct root pushed back, final answer wrong (all critic rows)
  budget.csv             cost per row (7 rows + H_E001a), mean and median
  skip_cause.csv         code_rule skip cases, one line each, with a cause class
  structure.txt, summary.md

Example (from repo root):
  python scripts/critic_revisions.py --out results/critic_revisions
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evidence import EvidenceItem, EvidenceLedger, onset_shift_ranking  # noqa: E402

LOOP_ROWS = ["hybrid", "llm_only", "hybrid_7b", "code_rule"]
LLM_CRITIC_ROWS = ["hybrid", "llm_only", "hybrid_7b"]   # rows with critic-role LLM calls
ROWS = LOOP_ROWS + ["no_critic", "react", "retrieval_only"]
BASELINE_ROWS = {"react", "retrieval_only"}
SYSTEMS = ["SS", "TT"]
PUSH_BACK = {"reject", "need_more_evidence"}
RULE_SKIP = "code onset rule skipped:"
HINT_RE = re.compile(r"\binvestigate\s+([A-Za-z0-9][A-Za-z0-9\-]*)\s*:", re.I)
EXPECTED_AC = {"hybrid": 48, "llm_only": 42, "hybrid_7b": 42, "code_rule": 37, "no_critic": 15,
               "react": 36, "retrieval_only": 5}
EXPECTED = {  # Phase 1 numbers (fix plan, Phase 1 Results / reconcile)
    "correct_pushed_back_final_wrong/hybrid": 4, "correct_pushed_back_final_wrong/llm_only": 4,
    "correct_pushed_back_final_wrong/hybrid_7b": 4, "rule_skip_cases/code_rule": 14,
    "r1_pushback/hybrid": 51, "r1_pushback/code_rule": 31, "h_e001a_correct/all": 48,
}
# master_cases.csv field -> kind
MASTER_FIELDS = [("system", "str"), ("truth", "norm"), ("final_root", "str"), ("correct", "bool"),
                 ("rounds", "int"), ("n_ledger", "int"), ("n_calls", "int"), ("llm_time_s", "float"),
                 ("critic_time_s", "float"), ("r1_root", "str"), ("r2_root", "str"), ("r3_root", "str"),
                 ("r1_decision", "str"), ("r2_decision", "str"), ("r3_decision", "str")]
MASTER_LOOP_FIELDS = [("correct_pushed_back", "bool"), ("rule_skip", "bool"), ("r1_pushback", "bool")]


# ------------------------------------------------------------------ helpers

def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def norm(s) -> str:
    """AC@1 scorer rule: strip + lowercase, exact match."""
    return str(s).strip().lower() if s is not None else ""


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        die(f"missing {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die(f"{path} is empty")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        rows = [{"note": "no rows"}]
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else ""


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else ""


def root_of(h) -> str:
    if isinstance(h, dict):
        rs = h.get("ranked_services") or []
        return str(rs[0]) if rs else ""
    return ""


def item_key(d: dict) -> tuple:
    return (d.get("id"), d.get("source"), json.dumps(d.get("args"), sort_keys=True, default=str),
            d.get("content"))


def call_key(d: dict) -> tuple:
    return (d.get("source"), json.dumps(d.get("args"), sort_keys=True, default=str))


def strings_in(obj, path=""):
    """Yield (field path, string) for every string inside a verdict."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings_in(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from strings_in(v, f"{path}[]")


def hint_of(verdict) -> tuple[str, str]:
    """-> (hint service, field path) from the first 'investigate <svc>:' in the verdict."""
    for p, s in strings_in(verdict):
        m = HINT_RE.search(s)
        if m:
            return m.group(1), p
    return "", ""


def to_ledger(items: list[dict]) -> EvidenceLedger:
    return EvidenceLedger(items=[EvidenceItem(id=d["id"], source=d["source"], args=d.get("args") or {},
                                              content=d["content"], created_at=d.get("created_at", 0.0))
                                 for d in items])


def ranking(items: list[dict]) -> list[str]:
    return [r[0] for r in onset_shift_ranking(to_ledger(items))] if items else []


_TOP_LINE = re.compile(r"^\s*-\s*([A-Za-z0-9\-]+):")


def e001_top12(content: str) -> list[str]:
    """Services listed in the contiguous block under 'Top 12 services' (same rule as Phase 1)."""
    lines = content.splitlines()
    i = next((k for k, l in enumerate(lines) if l.strip().startswith("Top 12 services")), None)
    out = []
    for l in lines[i + 1:] if i is not None else []:
        m = _TOP_LINE.match(l)
        if not m:
            break
        out.append(m.group(1))
    return out


def parse_bool(s: str) -> bool:
    if s not in ("True", "False"):
        raise ValueError(f"not a bool: {s!r}")
    return s == "True"


# ------------------------------------------------------------------ loading

def load_experiments(run_dirs: list[Path]):
    rows_at, case_list = {}, None
    for d in run_dirs:
        p = d / "experiment.json"
        if not p.exists():
            die(f"{p} not found")
        exp = json.loads(p.read_text())
        cases = list(exp["cases"])
        if case_list is None:
            case_list = cases
        elif sorted(cases) != sorted(case_list):
            die(f"case list in {p} differs from the first --runs folder; not paired")
        for row in exp["rows"]:
            if (d / row).is_dir():
                if row in rows_at:
                    die(f"row {row} has a folder in both {rows_at[row]} and {d}")
                rows_at[row] = d
    missing = [r for r in ROWS if r not in rows_at]
    if missing:
        die(f"rows missing from --runs: {missing}")
    return rows_at, sorted(case_list)


def find_json(row_dir: Path, case_id: str) -> Path:
    hits = [p for p in row_dir.glob(f"*/{case_id}.json") if p.parent.name != "llm_cache"]
    if len(hits) != 1:
        die(f"expected 1 JSON for {case_id} in {row_dir}, found {len(hits)}")
    return hits[0]


def service_lists(manifest: Path) -> tuple[dict, str]:
    """case -> set of services from metrics.parquet columns (same rule as the judge's query)."""
    if not manifest.exists():
        return {}, f"manifest {manifest} not found; valid_service = n/a"
    try:
        import pandas as pd
    except ImportError:
        return {}, "pandas not importable; valid_service = n/a"
    out, bad = {}, 0
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            cd = Path(r.get("case_dir", ""))
            cd = cd if cd.is_absolute() else REPO / cd
            try:
                cols = pd.read_parquet(cd / "metrics.parquet").columns
            except Exception:
                bad += 1
                continue
            out[r["case"]] = {norm(c.rsplit("_", 1)[0]) for c in cols if c != "time"}
    return out, (f"{bad} manifest cases without a readable metrics.parquet" if bad else "")


# ------------------------------------------------------------------ per record

def segments(calls: list[dict], critic: str) -> list[list[dict]]:
    """Cut llm_calls into rounds: a round ends with its critic call. Calls after the last critic call
    are returned separately as trailing (not a round); with no critic call at all, everything is one
    segment (no_critic, code_rule)."""
    segs: list[list[dict]] = [[]]
    for c in calls:
        segs[-1].append(c)
        if c.get("role") == critic:
            segs.append([])
    trailing = segs.pop() if len(segs) > 1 else []
    return segs, trailing


def build(row, case_id, rec, m, nc_ledger, nc_root, roles, a) -> dict:
    hyps = rec.get("hypotheses") or []
    verds = rec.get("critic_verdicts") or []
    ledger = rec.get("evidence_ledger") or []
    calls = rec.get("llm_calls") or []
    rounds = rec.get("iteration_count") or 0
    loop = row in LOOP_ROWS
    out = {"row": row, "case_id": case_id, "system": m["system"], "truth": m["truth"],
           "rounds": rounds, "hyps": hyps, "verds": verds, "ledger": ledger, "calls": calls}
    # round-1 identity
    if loop:
        n = len(nc_ledger)
        why = []
        if len(ledger) < n:
            why.append(f"ledger shorter than no_critic ({len(ledger)} < {n})")
        elif [item_key(d) for d in ledger[:n]] != [item_key(d) for d in nc_ledger]:
            why.append("first N items differ from no_critic")
        if norm(root_of(hyps[0]) if hyps else "") != norm(nc_root):
            why.append("round-1 root differs from no_critic final root")
        if rounds == 1 and len(ledger) != n:
            why.append(f"1-round case with {len(ledger)} items != N {n}")
        out["N"] = n
        out["r1_ok"] = not why
        out["r1_why"] = "; ".join(why)
    # segments ending at each critic call (LLM-critic rows only; code_rule has no critic calls)
    segs, trailing = segments(calls, a.critic_role)
    out["n_segments"] = len(segs)
    out["n_trailing_calls"] = len(trailing)
    out["exec_per_round"] = [sum(c.get("role") != a.critic_role for c in s) for s in segs]
    for c in calls:
        roles[row][str(c.get("role"))] += 1
    return out


# ------------------------------------------------------------------ main analysis

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs/heldout", "runs/heldout_baselines"])
    ap.add_argument("--master", default="results/critic_attribution/master_cases.csv")
    ap.add_argument("--manifest", default="data/manifests/sample_manifest.csv")
    ap.add_argument("--out", default="results/critic_revisions")
    ap.add_argument("--critic-role", default="critic")
    ap.add_argument("--allow-r1-mismatch", action="store_true",
                    help="go on when round 1 differs from no_critic; those cases get no per-round counts")
    ap.add_argument("--expected-ac", default=None, help=argparse.SUPPRESS)  # tests: JSON replacing AC + EXPECTED
    a = ap.parse_args()

    global EXPECTED_AC, EXPECTED
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    if a.expected_ac:
        e = json.loads(Path(a.expected_ac).read_text())
        EXPECTED_AC, EXPECTED = e["ac"], e["expected"]
        print(f"NOTE: expected values from {a.expected_ac} (test mode), not the held-out numbers")

    run_dirs = [Path(p) for p in a.runs]
    rows_at, cases = load_experiments(run_dirs)
    master_rows = read_csv(Path(a.master))
    master = {}
    for r in master_rows:
        k = (r["row"], r["case_id"])
        if k in master:
            die(f"duplicate {k} in {a.master}")
        master[k] = r
    want = {(r, c) for r in ROWS for c in cases}
    if set(master) != want:
        die(f"{a.master}: (row, case) set differs from the runs ({len(set(master) ^ want)} differ, "
            f"e.g. {sorted(set(master) ^ want)[:2]})")
    print(f"{len(cases)} cases, rows: " + ", ".join(f"{r} ({rows_at[r]})" for r in ROWS))

    # ---------- load + Step 0: rebuild vs master
    roles = defaultdict(Counter)
    recs, bad = {}, defaultdict(list)
    for c in cases:
        nc = json.loads(find_json(rows_at["no_critic"] / "no_critic", c).read_text())
        nc_ledger = nc.get("evidence_ledger") or []
        nc_root = ((nc.get("report") or {}).get("ranked_services") or [""])[0]
        for row in ROWS:
            p = find_json(rows_at[row] / row, c)
            rec = json.loads(p.read_text())
            if rec.get("case_id") != c:
                die(f"{p} has case_id {rec.get('case_id')!r}")
            m = master[(row, c)]
            x = build(row, c, rec, m, nc_ledger, nc_root, roles, a)
            ranked = (rec.get("report") or {}).get("ranked_services") or []
            calls = x["calls"]
            got = {"system": m["system"], "truth": m["truth"],
                   "final_root": ranked[0] if ranked else "",
                   "correct": bool(ranked) and norm(ranked[0]) == norm(m["truth"]),
                   "rounds": x["rounds"], "n_ledger": len(x["ledger"]), "n_calls": len(calls),
                   "llm_time_s": round(sum(float(k.get("latency_s") or 0) for k in calls), 3),
                   "critic_time_s": round(sum(float(k.get("latency_s") or 0) for k in calls
                                              if k.get("role") == a.critic_role), 3)}
            roots = [root_of(h) for h in x["hyps"]]
            decs = [str(v.get("decision", "")) for v in x["verds"]]
            for i in range(3):
                got[f"r{i + 1}_root"] = roots[i] if i < len(roots) else ""
                got[f"r{i + 1}_decision"] = decs[i] if i < len(decs) else ""
            if row in LOOP_ROWS:
                got["correct_pushed_back"] = any(i < len(roots) and norm(roots[i]) == norm(m["truth"])
                                                 and d != "accept" for i, d in enumerate(decs))
                got["rule_skip"] = any(RULE_SKIP in str(r) for v in x["verds"] for r in (v.get("reasons") or []))
                got["r1_pushback"] = bool(decs) and decs[0] in PUSH_BACK
            fl = MASTER_FIELDS + (MASTER_LOOP_FIELDS if row in LOOP_ROWS else [])
            for f, kind in fl:
                if f not in m:
                    die(f"{a.master} has no column {f}")
                mv, gv = m[f], got[f]
                try:
                    same = {"bool": lambda: parse_bool(mv) == gv,
                            "int": lambda: int(float(mv)) == int(gv),
                            "float": lambda: abs(float(mv) - float(gv)) <= 0.0011,
                            "norm": lambda: norm(mv) == norm(gv),
                            "str": lambda: mv == str(gv)}[kind]()
                except ValueError:
                    same = False
                if not same:
                    bad[f].append(f"{row}/{c} master={mv!r} json={gv!r}")
            x.update(got)
            x["roots"], x["decs"] = roots, decs
            x["e001_top_a"] = m.get("e001_top_a", "")
            x["h_e001a_correct"] = m.get("h_e001a_correct", "") == "True"
            recs[(row, c)] = x
    if bad:
        for f, ex in bad.items():
            print(f"  MISMATCH vs master, field {f}: {len(ex)} cases, e.g. {ex[:3]}")
        die("rebuilt per-case values differ from master_cases.csv; nothing written")
    checks = []
    for row, n in EXPECTED_AC.items():
        g = sum(recs[(row, c)]["correct"] for c in cases)
        checks.append((f"ac1/{row}", n, g))
    for key, n in EXPECTED.items():
        what, row = key.split("/")
        if what == "correct_pushed_back_final_wrong":
            g = sum(recs[(row, c)]["correct_pushed_back"] and not recs[(row, c)]["correct"] for c in cases)
        elif what == "rule_skip_cases":
            g = sum(recs[(row, c)]["rule_skip"] for c in cases)
        elif what == "r1_pushback":
            g = sum(recs[(row, c)]["r1_pushback"] for c in cases)
        elif what == "h_e001a_correct":
            g = sum(recs[("hybrid", c)]["h_e001a_correct"] for c in cases)
        else:
            die(f"unknown expected key {key}")
        checks.append((key, n, g))
    failed = [(k, e, g) for k, e, g in checks if e != g]
    print(f"Step 0: master_cases.csv matches the raw JSONs in every field; known counts "
          f"{len(checks) - len(failed)}/{len(checks)} ok")
    if failed:
        for k, e, g in failed:
            print(f"  MISMATCH {k}: expected {e}, got {g}")
        die("known-count check failed; nothing written")

    # ---------- Step 1: structure report
    S = ["# Structure report (assumptions to audit)", ""]
    S.append("llm_calls roles per row: " + "; ".join(f"{r}: " + ", ".join(f"{k}={v}" for k, v in sorted(roles[r].items()))
                                                    for r in ROWS))
    for r in LLM_CRITIC_ROWS:
        if a.critic_role not in roles[r]:
            die(f"role {a.critic_role!r} never appears in {r} llm_calls (roles: {sorted(roles[r])}); pass --critic-role")
    vkeys, hint_fields, hint_n = Counter(), Counter(), Counter()
    for row in LOOP_ROWS:
        for c in cases:
            for v in recs[(row, c)]["verds"]:
                vkeys.update(v.keys() if isinstance(v, dict) else ["<not a dict>"])
                h, p = hint_of(v)
                if h:
                    hint_fields[p] += 1
                    hint_n[(row, str(v.get("decision")))] += 1
    S.append("verdict keys (critic rows): " + ", ".join(f"{k}={n}" for k, n in sorted(vkeys.items())))
    S.append("onset hint found in field: " + (", ".join(f"{k}={n}" for k, n in hint_fields.items()) or "NONE"))
    S.append("verdicts with a hint, by (row, decision): " + ", ".join(f"{r}/{d}={n}" for (r, d), n in sorted(hint_n.items())))
    if not hint_fields:
        print("WARNING: no 'investigate <svc>:' hint found in any verdict; hint columns will be empty")
    r1_bad = [(r, c) for r in LOOP_ROWS for c in cases if not recs[(r, c)]["r1_ok"]]
    S.append(f"round-1 identity: {len(LOOP_ROWS) * len(cases) - len(r1_bad)}/{len(LOOP_ROWS) * len(cases)} ok")
    if r1_bad:
        for r, c in r1_bad[:10]:
            S.append(f"  r1 mismatch {r}/{c}: {recs[(r, c)]['r1_why']}")
        if not a.allow_r1_mismatch:
            print("\n".join(S))
            die(f"{len(r1_bad)} round-1 identity mismatch(es); explain, then rerun with --allow-r1-mismatch")
    # segment count vs rounds; calibrate d on 2-round cases
    seg_bad = [(r, c) for r in LLM_CRITIC_ROWS for c in cases
               if recs[(r, c)]["n_segments"] != recs[(r, c)]["rounds"]]
    S.append(f"critic-call segments == iteration_count: {len(LLM_CRITIC_ROWS) * len(cases) - len(seg_bad)}/"
             f"{len(LLM_CRITIC_ROWS) * len(cases)} (LLM-critic rows; code_rule has no critic calls, so its "
             f"3-round cases get rounds-2+ totals only)")
    tr = Counter(recs[(r, c)]["n_trailing_calls"] for r in LLM_CRITIC_ROWS for c in cases)
    S.append(f"pipeline calls after the last critic call (not counted as a round), cases by count: {dict(tr)}")
    for r, c in seg_bad[:5]:
        S.append(f"  segments != rounds: {r}/{c} ({recs[(r, c)]['n_segments']} vs {recs[(r, c)]['rounds']})")
    ds = Counter()
    for r in LOOP_ROWS:
        for c in cases:
            x = recs[(r, c)]
            if x["rounds"] == 2 and x["r1_ok"] and x["n_segments"] == 2:
                ds[x["exec_per_round"][1] - (len(x["ledger"]) - x["N"])] += 1
    d = next(iter(ds)) if len(ds) == 1 else None
    S.append(f"calibration d (pipeline calls - items, round 2 of 2-round cases): {dict(ds)} -> "
             + (f"d = {d}, per-round split used where it checks out" if d is not None
                else "not constant, per-round split NOT used for 3-round cases"))

    # ---------- per-round table
    rounds_rows, case_rows = [], []
    for row in LOOP_ROWS:
        for c in cases:
            x = recs[(row, c)]
            N, L = x.get("N", 0), x["ledger"]
            ok = x["r1_ok"]
            beyond = len(L) - N if ok else None
            # per-round item counts
            per = None
            if ok and x["rounds"] >= 2:
                if x["rounds"] == 2:
                    per = {2: beyond}
                elif d is not None and x["n_segments"] == x["rounds"]:
                    est = {k: x["exec_per_round"][k - 1] - d for k in range(2, x["rounds"] + 1)}
                    if all(v >= 0 for v in est.values()) and sum(est.values()) == beyond:
                        per = est
            bounds, start = {}, N
            if per:
                for k in sorted(per):
                    bounds[k] = (start, start + per[k])
                    start += per[k]
            id_idx = {str(it.get("id")): i for i, it in enumerate(L)}
            seen_calls = {call_key(it) for it in L[:N]} if ok else set()
            earlier_cites = set(str(i) for i in ((x["hyps"][0].get("citations") or []) if x["hyps"] else []))
            case_new_unique = 0
            n_reword = 0
            for k in range(2, len(x["roots"]) + 1):
                prev, cur = x["roots"][k - 2], x["roots"][k - 1]
                pc, cc = norm(prev) == norm(x["truth"]), norm(cur) == norm(x["truth"])
                if pc and cc:
                    tr = "correct->correct"
                elif pc:
                    tr = "correct->wrong"
                elif cc:
                    tr = "wrong->correct"
                else:
                    tr = "wrong->wrong_same" if norm(prev) == norm(cur) else "wrong->wrong_diff"
                pv = x["verds"][k - 2] if k - 2 < len(x["verds"]) else {}
                hint, _ = hint_of(pv) if str(pv.get("decision")) in PUSH_BACK else ("", "")
                hyp = x["hyps"][k - 1]
                cites = [str(i) for i in (hyp.get("citations") or [])]
                new_k = uniq_k = ""
                if k in bounds:
                    lo, hi = bounds[k]
                    new_k = hi - lo
                    uniq_k = 0
                    for it in L[lo:hi]:
                        if call_key(it) not in seen_calls:
                            uniq_k += 1
                        seen_calls.add(call_key(it))
                    case_new_unique += uniq_k
                elif ok and beyond == 0:
                    new_k, uniq_k = 0, 0
                same_root = norm(prev) == norm(cur)
                reword = (same_root and new_k == 0) if new_k != "" else ""
                n_reword += reword is True
                rounds_rows.append({
                    "row": row, "case_id": c, "system": x["system"], "truth": x["truth"], "round": k,
                    "prev_root": prev, "root": cur, "prev_decision": str(pv.get("decision", "")),
                    "root_changed": not same_root, "transition": tr,
                    "hint": hint, "hint_eq_truth": (norm(hint) == norm(x["truth"])) if hint else "",
                    "hint_eq_h_e001a": (norm(hint) == norm(x["e001_top_a"])) if hint else "",
                    "root_eq_hint": (norm(cur) == norm(hint)) if hint else "",
                    "root_eq_h_e001a": norm(cur) == norm(x["e001_top_a"]),
                    "r1_identity_ok": ok, "split_ok": k in bounds or (ok and beyond == 0),
                    "new_tool_calls": new_k, "new_unique_items": uniq_k,
                    "n_cited": len(cites),
                    "cited_after_r1": sum(1 for i in cites if i in id_idx and id_idx[i] >= N) if ok else "",
                    "cited_this_round": (sum(1 for i in cites if i in id_idx and bounds[k][0] <= id_idx[i] < bounds[k][1])
                                         if k in bounds else ""),
                    "cited_not_cited_before": sum(1 for i in cites if i not in earlier_cites),
                    "bad_cites": sum(1 for i in cites if i not in id_idx),
                    "reword_only": reword,
                    "explanation_same_as_prev": (hyp.get("explanation") or "") == (x["hyps"][k - 2].get("explanation") or ""),
                    "decision": x["decs"][k - 1] if k - 1 < len(x["decs"]) else "",
                })
                earlier_cites |= set(cites)
            case_rows.append({
                "row": row, "case_id": c, "system": x["system"], "truth": x["truth"], "rounds": x["rounds"],
                "correct": x["correct"], "r1_correct": norm(x["r1_root"]) == norm(x["truth"]),
                "round_path": " -> ".join(f"{r_ or '?'} ({d_})" for r_, d_ in zip(x["roots"], x["decs"])),
                "N_r1_items": N, "n_ledger": len(L), "r1_identity_ok": ok,
                "new_items_rounds2plus": beyond if x["rounds"] >= 2 else (0 if ok else ""),
                "new_unique_rounds2plus": (sum(1 for it in L[N:] if call_key(it) not in {call_key(j) for j in L[:N]})
                                           if ok else ""),
                "per_round_split": json.dumps(per) if per else "",
                "root_changed_any": any(norm(x["roots"][i]) != norm(x["roots"][i - 1]) for i in range(1, len(x["roots"]))),
                "final_eq_h_e001a": norm(x["final_root"]) == norm(x["e001_top_a"]),
                "reword_only_rounds": n_reword,
            })

    # ---------- per-row summaries
    by_row, hint_rows = [], []
    for row in LOOP_ROWS:
        rr = [r for r in rounds_rows if r["row"] == row]
        cr = [r for r in case_rows if r["row"] == row]
        multi = [r for r in cr if r["rounds"] >= 2]
        nb = [r["new_items_rounds2plus"] for r in multi if r["new_items_rounds2plus"] != ""]
        tc = Counter(r["transition"] for r in rr)
        rs = [r for r in rr if r["new_tool_calls"] != ""]
        by_row.append({
            "row": row, "cases": len(cr), "multi_round_cases": len(multi),
            "rounds_ge2": len(rr), "root_changed": sum(r["root_changed"] for r in rr),
            **{t: tc[t] for t in ["correct->correct", "correct->wrong", "wrong->correct",
                                  "wrong->wrong_same", "wrong->wrong_diff"]},
            "rounds_with_split": len(rs),
            "reword_only_rounds": sum(r["reword_only"] is True for r in rr),
            "zero_new_tool_call_rounds": sum(r["new_tool_calls"] == 0 for r in rs),
            "same_root_rounds": sum(not r["root_changed"] for r in rr),
            "explanation_unchanged_rounds": sum(r["explanation_same_as_prev"] for r in rr),
            "new_items_r2plus_mean": mean(nb), "new_items_r2plus_median": median(nb),
            "new_unique_r2plus_mean": mean([r["new_unique_rounds2plus"] for r in multi if r["new_unique_rounds2plus"] != ""]),
            "multi_cases_zero_new_items": sum(1 for v in nb if v == 0),
            "rounds_citing_items_after_r1": sum(1 for r in rr if r["cited_after_r1"] not in ("", 0)),
            "rounds_root_eq_h_e001a": sum(r["root_eq_h_e001a"] for r in rr),
            "multi_cases_final_eq_h_e001a": sum(r["final_eq_h_e001a"] for r in multi),
            "multi_cases_correct": sum(r["correct"] for r in multi),
            "r1_identity_fail": sum(not r["r1_identity_ok"] for r in cr),
        })
        hr = [r for r in rr if r["hint"]]
        fol = [r for r in hr if r["root_eq_hint"] is True]
        nof = [r for r in hr if r["root_eq_hint"] is False]
        hint_rows.append({
            "row": row, "rounds_ge2": len(rr), "rounds_after_hint": len(hr),
            "followed": len(fol), "follow_rate": round(len(fol) / len(hr), 4) if hr else "",
            "followed_correct": sum(norm(r["root"]) == norm(r["truth"]) for r in fol),
            "not_followed": len(nof),
            "not_followed_correct": sum(norm(r["root"]) == norm(r["truth"]) for r in nof),
            "hint_eq_truth": sum(r["hint_eq_truth"] is True for r in hr),
            "hint_eq_h_e001a": sum(r["hint_eq_h_e001a"] is True for r in hr),
            "rounds_no_hint": len(rr) - len(hr),
            "no_hint_correct": sum(norm(r["root"]) == norm(r["truth"]) for r in rr if not r["hint"]),
        })

    # ---------- correct pushed back, final wrong
    pbw = []
    for row in LOOP_ROWS:
        for c in cases:
            x = recs[(row, c)]
            if x["correct_pushed_back"] and not x["correct"]:
                first = next(i for i, d_ in enumerate(x["decs"]) if norm(x["roots"][i]) == norm(x["truth"]) and d_ != "accept")
                h, _ = hint_of(x["verds"][first])
                pbw.append({"row": row, "case_id": c, "system": x["system"], "truth": x["truth"],
                            "final_root": x["final_root"], "pushed_back_round": first + 1,
                            "decision_then": x["decs"][first], "hint_then": h,
                            "round_path": " -> ".join(f"{r_ or '?'} ({d_})" for r_, d_ in zip(x["roots"], x["decs"])),
                            "e001_top_a": x["e001_top_a"],
                            "reasons_then": " | ".join(str(r_) for r_ in (x["verds"][first].get("reasons") or []))[:600]})

    # ---------- budget
    budget = []
    nc_correct = sum(recs[("no_critic", c)]["correct"] for c in cases)
    nc_rounds = sum(recs[("no_critic", c)]["rounds"] for c in cases)
    nc_time = sum(recs[("no_critic", c)]["llm_time_s"] for c in cases)
    tok_missing = Counter()
    for row in ROWS:
        xs = [recs[(row, c)] for c in cases]
        per = defaultdict(list)
        for x in xs:
            pin = pout = cin = cout = 0
            for k in x["calls"]:
                it, ot = k.get("input_tokens"), k.get("output_tokens")
                if it is None or ot is None:
                    tok_missing[row] += 1
                it, ot = int(it or 0), int(ot or 0)
                if k.get("role") == a.critic_role:
                    cin, cout = cin + it, cout + ot
                else:
                    pin, pout = pin + it, pout + ot
            for key, v in [("rounds", x["rounds"]), ("tool_calls", len(x["ledger"])), ("llm_calls", len(x["calls"])),
                           ("critic_calls", sum(k.get("role") == a.critic_role for k in x["calls"])),
                           ("in_tok_pipeline", pin), ("out_tok_pipeline", pout),
                           ("in_tok_critic", cin), ("out_tok_critic", cout),
                           ("llm_time_s", x["llm_time_s"]), ("critic_time_s", x["critic_time_s"])]:
                per[key].append(v)
        corr = sum(x["correct"] for x in xs)
        line = {"row": row, "n": len(xs), "ac1": corr}
        for key, v in per.items():
            line[f"{key}_mean"], line[f"{key}_median"] = mean(v), median(v)
        extra_r = sum(per["rounds"]) - nc_rounds
        extra_min = (sum(per["llm_time_s"]) - nc_time) / 60
        line["extra_correct_vs_no_critic"] = corr - nc_correct
        line["extra_rounds_total"] = extra_r
        line["extra_correct_per_extra_round"] = round((corr - nc_correct) / extra_r, 4) if row in LOOP_ROWS and extra_r else ""
        line["extra_correct_per_extra_llm_min"] = round((corr - nc_correct) / extra_min, 4) if row in LOOP_ROWS and extra_min > 0 else ""
        line["tokens_missing_calls"] = tok_missing[row]
        budget.append(line)
    budget.append({"row": "H_E001a (screen, post hoc)", "n": len(cases),
                   "ac1": sum(recs[("hybrid", c)]["h_e001a_correct"] for c in cases),
                   "rounds_mean": 0, "tool_calls_mean": 1, "llm_calls_mean": 0, "llm_time_s_mean": 0,
                   "critic_time_s_mean": 0})

    # ---------- skip cause (code_rule)
    svc_lists, svc_note = service_lists(Path(a.manifest))
    skip_rows = []
    for c in cases:
        x = recs[("code_rule", c)]
        if not x["rule_skip"]:
            continue
        idx = [i for i, v in enumerate(x["verds"]) if any(RULE_SKIP in str(r) for r in (v.get("reasons") or []))]
        if idx != [len(x["verds"]) - 1]:
            die(f"code_rule/{c}: skip verdict(s) at {idx}, not only the last verdict ({len(x['verds'])})")
        k = idx[0]
        root = x["roots"][k] if k < len(x["roots"]) else ""
        L = x["ledger"]
        hyp = x["hyps"][k]
        cited = {str(i) for i in (hyp.get("citations") or [])}
        rk_full = ranking(L)
        rk_cited = ranking([it for it in L if str(it.get("id")) in cited])
        text = "\n".join(str(it.get("content", "")) for it in L).lower()
        args = " ".join(json.dumps(it.get("args"), default=str) for it in L).lower()
        r_ = norm(root)
        e001 = L[0].get("content", "") if L else ""
        top12 = e001_top12(e001)
        in_text = bool(r_) and r_ in text
        in_args = bool(r_) and r_ in args
        underscore = bool(r_) and re.search(re.escape(r_) + r"_[A-Za-z0-9.\-]*_", text) is not None
        in_full = r_ in [norm(s) for s in rk_full]
        if in_full and len(rk_full) >= 2:
            cause = "ranked in full ledger (skip unexpected in full mode)"
        elif len(rk_full) < 2:
            cause = "fewer than 2 ranked services"
        elif not (in_text or in_args):
            cause = "claimed root never in the ledger"
        else:
            cause = "claimed root in ledger but not ranked" + (" (underscore-name parsing-gap candidate)" if underscore else "")
        valid = svc_lists.get(c)
        skip_rows.append({
            "case_id": c, "system": x["system"], "truth": x["truth"], "claimed_root": root, "skip_round": k + 1,
            "final_correct": x["correct"], "cause": cause,
            "skip_reason": " | ".join(str(r) for r in (x["verds"][k].get("reasons") or []) if RULE_SKIP in str(r))[:500],
            "valid_service": ("n/a" if valid is None else (r_ in valid)),
            "root_in_e001_top12": r_ in [norm(s) for s in top12], "e001_top_a": x["e001_top_a"],
            "root_in_ledger_text": in_text, "root_in_tool_args": in_args, "underscore_line": underscore,
            "n_ranked_full": len(rk_full), "root_in_ranking_full": in_full,
            "rank_full_top5": " ".join(rk_full[:5]),
            "n_cited": len(cited), "n_ranked_cited": len(rk_cited),
            "root_in_ranking_cited": r_ in [norm(s) for s in rk_cited],
            "rank_cited_top5": " ".join(rk_cited[:5]), "n_ledger": len(L),
        })
    if svc_note:
        S.append(f"service lists: {svc_note}")

    # ---------- summary
    L_ = ["# Critic revisions + budget (post hoc, descriptive, A13.1): summary", "",
          f"Cases: {len(cases)} (all SS+TT). Generated by scripts/critic_revisions.py. "
          "Step 0: every (row, case) field matches master_cases.csv; AC@1 and Phase 1 counts reproduced.", ""]
    L_ += S[2:] + [""]
    L_.append("## Revision rounds (rounds >= 2, critic rows)")
    L_.append("| row | multi-round cases | rounds >= 2 | root changed | c->w | w->c | w->w same | w->w diff | c->c "
              "| reword-only | 0 new tool calls (of split) | new items r2+ mean / median | multi cases with 0 new items "
              "| multi final = H_E001a |")
    L_.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in by_row:
        L_.append(f"| {r['row']} | {r['multi_round_cases']} | {r['rounds_ge2']} | {r['root_changed']} | {r['correct->wrong']} "
                  f"| {r['wrong->correct']} | {r['wrong->wrong_same']} | {r['wrong->wrong_diff']} | {r['correct->correct']} "
                  f"| {r['reword_only_rounds']} | {r['zero_new_tool_call_rounds']}/{r['rounds_with_split']} "
                  f"| {r['new_items_r2plus_mean']} / {r['new_items_r2plus_median']} | {r['multi_cases_zero_new_items']} "
                  f"| {r['multi_cases_final_eq_h_e001a']}/{r['multi_round_cases']} |")
    L_.append("\n## Onset hint following (round k root vs hint in verdict k-1)")
    L_.append("| row | rounds after a hint | followed | root correct when followed | not followed | correct when not followed "
              "| hint = truth | hint = H_E001a | rounds without hint (correct) |")
    L_.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in hint_rows:
        L_.append(f"| {r['row']} | {r['rounds_after_hint']} | {r['followed']} | {r['followed_correct']}/{r['followed']} "
                  f"| {r['not_followed']} | {r['not_followed_correct']}/{r['not_followed']} | {r['hint_eq_truth']} "
                  f"| {r['hint_eq_h_e001a']} | {r['rounds_no_hint']} ({r['no_hint_correct']}) |")
    L_.append("\n## Correct root pushed back, final answer wrong")
    for row in LOOP_ROWS:
        ps = [p for p in pbw if p["row"] == row]
        L_.append(f"**{row} ({len(ps)})**" + ("" if ps else ": none"))
        for p in ps:
            L_.append(f"- {p['case_id']}: pushed back in r{p['pushed_back_round']} ({p['decision_then']}"
                      + (f", hint {p['hint_then']}" if p["hint_then"] else ", no hint") + f"); {p['round_path']}")
    L_.append(f"\n## Budget (mean / median per case, all {len(cases)} cases)")
    L_.append("| row | AC@1 | rounds | tool calls | LLM calls | in tok pipeline | out tok pipeline | in tok critic "
              "| out tok critic | LLM time s | critic time s | extra correct per extra round | per extra LLM min |")
    L_.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in budget:
        def mm(k):
            return f"{b.get(k + '_mean', '')} / {b.get(k + '_median', '')}" if k + "_median" in b else f"{b.get(k + '_mean', '')}"
        L_.append(f"| {b['row']} | {b['ac1']} | {mm('rounds')} | {mm('tool_calls')} | {mm('llm_calls')} | {mm('in_tok_pipeline')} "
                  f"| {mm('out_tok_pipeline')} | {mm('in_tok_critic')} | {mm('out_tok_critic')} | {mm('llm_time_s')} "
                  f"| {mm('critic_time_s')} | {b.get('extra_correct_per_extra_round', '')} | {b.get('extra_correct_per_extra_llm_min', '')} |")
    if any(tok_missing.values()):
        L_.append(f"\nCalls without token counts (counted as 0): {dict(tok_missing)}")
    L_.append(f"\n## code_rule skip cause ({len(skip_rows)} cases)")
    for k_, n_ in Counter(r["cause"] for r in skip_rows).most_common():
        L_.append(f"- {k_}: {n_}")
    for r in skip_rows:
        L_.append(f"- {r['case_id']}: claimed {r['claimed_root']} (truth {r['truth']}); in E001 top12 {r['root_in_e001_top12']}, "
                  f"valid service {r['valid_service']}, ranked full {r['root_in_ranking_full']} ({r['n_ranked_full']}), "
                  f"cited {r['root_in_ranking_cited']} ({r['n_ranked_cited']}); reason: {r['skip_reason'][:160]}")
    summ = "\n".join(L_) + "\n"

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "r1_identity.csv", [{"row": r, "case_id": c, "N": recs[(r, c)]["N"], "n_ledger": len(recs[(r, c)]["ledger"]),
                                          "rounds": recs[(r, c)]["rounds"], "ok": recs[(r, c)]["r1_ok"],
                                          "why": recs[(r, c)]["r1_why"]} for r in LOOP_ROWS for c in cases])
    write_csv(out / "revision_rounds.csv", rounds_rows)
    write_csv(out / "revision_cases.csv", case_rows)
    write_csv(out / "revision_by_row.csv", by_row)
    write_csv(out / "hint_by_row.csv", hint_rows)
    write_csv(out / "pushed_back_wrong.csv", pbw)
    write_csv(out / "budget.csv", budget)
    write_csv(out / "skip_cause.csv", skip_rows)
    (out / "structure.txt").write_text("\n".join(S) + "\n")
    (out / "summary.md").write_text(summ)
    print("\n" + summ)
    print(f"wrote {out}/ (8 CSVs + structure.txt + summary.md)")


if __name__ == "__main__":
    main()
