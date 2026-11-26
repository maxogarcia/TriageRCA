#!/usr/bin/env python3
"""Critic attribution (post hoc, Addendum A13): onset-screen baseline and paired tests between rows.

Read-only. No LLM calls. Reads the saved held-out runs and writes CSVs + summary.md to a NEW --out.

Inputs (repo-relative defaults):
  --runs       experiment folders from run_ablation.py (default runs/heldout runs/heldout_baselines).
               Each needs experiment.json, scores_cases.csv and <row>/<config>/<case_id>.json.
  --grounding  results/grounding_heldout/grounding_secondary_cases.csv (judge status, grounding_expl)

Step 0 (before any comparison):
  master table, one line per (row, case) for 7 rows x 60 cases, rebuilt from the raw JSONs, then
  reconciled against scores_cases.csv case by case and against every count already reported
  (EXPECTED below, with the doc each number comes from). Any mismatch stops the script with
  nothing written. To go on after a mismatch has been explained, pass --ack <check> ... and
  --ack-note "<explanation>"; both are recorded in reconcile.csv and summary.md.

Heuristics (A13):
  H_E001a  first service under "Top 12 services" in E001 (metric_overview). Primary baseline.
           E001 must be identical (source, args, content) across the 6 telemetry rows per case.
  H_E001b  top of src.evidence.onset_shift_ranking over E001 only. Descriptive.
  H_final  top of onset_shift_ranking over the row's own full final ledger (needs >= 2 ranked
           services). Diagnostic only, not a baseline.

Tests (A13): exact two-sided McNemar, family "critic attribution" (9 comparisons), Holm.

Outputs (--out, must be new or empty):
  master_cases.csv       one line per (row, case): the per-case source of truth for later phases
  reconcile.csv          every check: expected, got, status (ok / ack)
  h_e001_by_row.csv      agreement + case categories per row, variants a and b, all / SS / TT
  h_e001_changes.csv     corrections and regressions vs H_E001a, every telemetry row
  h_final_by_row.csv     H_final agreement and AC@1 when agree vs disagree
  mcnemar.csv            family tests (Holm) + descriptive H_E001b tests + vs-ReAct reproduction
  skip_split.csv         hybrid vs code_rule discordant cases by code_rule rule skip and system
  summary.md             numbers for the fix-plan Results block

Example (from repo root):
  python scripts/critic_attribution.py --out results/critic_attribution
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evidence import EvidenceItem, EvidenceLedger, onset_shift_ranking  # noqa: E402

LOOP_ROWS = ["hybrid", "llm_only", "hybrid_7b", "code_rule"]          # critic rows with a real critic
ROWS = LOOP_ROWS + ["no_critic", "react", "retrieval_only"]
TELEMETRY_ROWS = [r for r in ROWS if r != "retrieval_only"]          # rows that get E001
BASELINE_ROWS = {"react", "retrieval_only"}                           # no critic loop at all
LIST_ROWS = ["hybrid", "code_rule", "llm_only", "react"]              # A13: lists shown in summary
SYSTEMS = ["SS", "TT"]
PUSH_BACK = {"reject", "need_more_evidence"}
RULE_SKIP = "code onset rule skipped:"
PARSE_FAIL = "critic output could not be parsed"
UNCHECKED = "accepted unchecked"
NOT_ACCEPTED_ERR = "did not accept within"
STATUSES = {"ok", "parse_fail_label", "no_report"}
THRESHOLD = 0.75          # A12 high grounding
READING_CUTOFF = 45       # A13 reading rule: hybrid == H_E001a in >= 45 of 60 cases
EPS = 1e-9

# A13 family "critic attribution": (A, B). "H_E001a" is the heuristic as a pseudo-row.
FAMILY = [("H_E001a", "hybrid"), ("H_E001a", "code_rule"), ("H_E001a", "llm_only"),
          ("H_E001a", "react"), ("H_E001a", "no_critic"),
          ("hybrid", "code_rule"), ("hybrid", "llm_only"), ("hybrid", "hybrid_7b"),
          ("llm_only", "code_rule")]
DESCRIPTIVE_B = [("H_E001b", r) for r in ("hybrid", "code_rule", "llm_only", "react", "no_critic")]
REPRO_REACT = [(r, "react") for r in ("no_critic", "code_rule", "llm_only", "hybrid_7b", "hybrid")]

# ------------------------------------------------------------------ expected (already reported)
# Sources: H = heldout-baselines-results.md, G = heldout-grounding-results.md,
#          R = fix-plan-critic-review.md (review findings, Nov 17).
_AC = {"retrieval_only": (5, 0), "no_critic": (10, 5), "react": (17, 19), "code_rule": (22, 15),
       "llm_only": (21, 21), "hybrid_7b": (21, 21), "hybrid": (25, 23)}
_2X2 = {  # scored, C+H, C+L, W+H, W+L, excl parse fail, excl zero expl claims, excl no report
    "react": (52, 22, 10, 4, 16, 8, 0, 0), "retrieval_only": (55, 2, 3, 32, 18, 0, 4, 1),
    "no_critic": (56, 8, 7, 19, 22, 4, 0, 0), "code_rule": (59, 22, 15, 12, 10, 1, 0, 0),
    "hybrid_7b": (60, 22, 20, 8, 10, 0, 0, 0), "llm_only": (55, 25, 15, 11, 4, 5, 0, 0),
    "hybrid": (52, 26, 16, 3, 7, 8, 0, 0)}
_LLM_TIME = {"hybrid": 124.5, "llm_only": 119.5, "hybrid_7b": 51.8, "code_rule": 20.7,
             "no_critic": 13.9, "react": 8.6, "retrieval_only": 3.5}


def _expected() -> dict:
    e = {}
    for row, (ss, tt) in _AC.items():
        e[f"ac1/{row}"] = (ss + tt, "H table")
        e[f"ac1_SS/{row}"] = (ss, "H table")
        e[f"ac1_TT/{row}"] = (tt, "H table")
    e["accepted/hybrid"] = (39, "R concern 1")
    for row, n in {"hybrid": 2, "llm_only": 7, "hybrid_7b": 11, "code_rule": 23}.items():
        e[f"wrong_accepted/{row}"] = (n, "R concern 1")
    e["never_accepted/hybrid"] = (21, "R concern 1; H Daniel's checks")
    e["never_accepted_correct/hybrid"] = (11, "R concern 1")
    e["never_accepted_wrong/hybrid"] = (10, "R concern 1")
    e["did_not_accept_error/hybrid"] = (21, "H Daniel's checks (grep)")
    for row, (fx, br) in {"code_rule": (23, 1), "hybrid": (34, 1), "llm_only": (27, 0),
                          "hybrid_7b": (28, 1)}.items():
        e[f"fixed_vs_no_critic/{row}"] = (fx, "R concern 3")
        e[f"broke_vs_no_critic/{row}"] = (br, "R concern 3")
    for row, (a, b) in {"no_critic": (3, 24), "code_rule": (11, 10), "llm_only": (14, 8),
                        "hybrid_7b": (12, 6), "hybrid": (17, 5)}.items():
        e[f"vs_react_row_only/{row}"] = (a, "H McNemar table")
        e[f"vs_react_react_only/{row}"] = (b, "H McNemar table")
    e["vs_react_p/hybrid"] = ("0.017", "H McNemar table (3 dp)")
    e["vs_react_p/code_rule"] = ("1.0", "H McNemar table (1 dp)")
    e["vs_react_p/llm_only"] = ("0.29", "H McNemar table (2 dp)")
    e["vs_react_p/hybrid_7b"] = ("0.24", "H McNemar table (2 dp)")
    e["vs_react_p/no_critic"] = ("<0.001", "H McNemar table")
    names = ["scored", "correct_high", "correct_low", "wrong_high", "wrong_low",
             "excl_parse_fail", "excl_zero_expl", "excl_no_report"]
    for row, vals in _2X2.items():
        for k, v in zip(names, vals):
            e[f"2x2_{k}/{row}"] = (v, "G 2x2 table")
    for k, v in {"ok": 393, "parse_fail_label": 26, "no_report": 1}.items():
        e[f"judge_status_{k}/all"] = (v, "G run status")
    for row, n in {"hybrid": 11, "hybrid_7b": 17, "llm_only": 10, "code_rule": 29}.items():
        e[f"zero_diff_vs_no_critic/{row}"] = (n, "G A11 zero-difference pairs")
    e["r1_pushback/hybrid"] = (49, "R concern 3")
    e["r1_pushback/code_rule"] = (31, "R concern 3")
    for row, (n, w) in {"hybrid": (23, 4), "llm_only": (18, 4), "hybrid_7b": (11, 4)}.items():
        e[f"correct_pushed_back/{row}"] = (n, "R concern 1")
        e[f"correct_pushed_back_final_wrong/{row}"] = (w, "R concern 1")
    e["unchecked_accept/hybrid"] = (3, "R concern 1")
    e["unchecked_accept_correct/hybrid"] = (3, "R concern 1")
    e["unchecked_accept/critic_rows"] = (6, "R concern 1 (6 across rows)")
    e["rule_skip_cases/code_rule"] = (14, "R concern 3")
    e["rule_skip_cases_SS/code_rule"] = (2, "R concern 3")
    e["rule_skip_cases_TT/code_rule"] = (12, "R concern 3")
    e["rule_skip_wrong/code_rule"] = (13, "R concern 3 (13 of 23 wrong accepts)")
    for row, t in _LLM_TIME.items():
        e[f"mean_llm_time_s/{row}"] = (f"{t:.1f}", "H table (1 dp)")
    e["truth_vs_case_id_mismatch/all"] = (0, "consistency (manifest truth vs case id)")
    e["truth_differs_across_rows/all"] = (0, "consistency (scores_cases.csv)")
    e["system_differs_across_rows/all"] = (0, "consistency (scores_cases.csv)")
    e["hyp_verdict_len_mismatch/all"] = (0, "consistency (RunRecord)")
    e["final_ne_last_hypothesis/all"] = (0, "consistency (RunRecord)")
    e["accept_not_last_round/all"] = (0, "consistency (loop stops on accept)")
    return e


# ------------------------------------------------------------------ helpers

def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def norm(s) -> str:
    """AC@1 scorer rule (score_ablation.norm): strip + lowercase, exact match."""
    return str(s).strip().lower() if s is not None else ""


SYSTEM_NAME = {"ss": "SS", "tt": "TT", "ob": "OB"}


def root_from_case_id(case_id: str) -> str:
    head, _, rest = case_id.partition("_")
    parts = rest.rsplit("_", 2)
    if not head.startswith("re2") or len(parts) != 3 or not parts[0]:
        die(f"cannot parse case id {case_id!r}")
    return parts[0]


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
        die(f"internal: nothing to write for {path.name}")
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar = binomial(n=b+c, 0.5) on the discordant pairs; 1.0 if none."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ E001 / heuristics

_TOP_HEADER = "Top 12 services"
_OVERVIEW_LINE = re.compile(r"^\s*-\s*(?P<svc>[A-Za-z0-9\-]+):\s*(?P<col>[^,]+),.*?shift=(?P<shift>[-+\d.eE]+)")


def e001_top_a(content: str) -> tuple[str, str, bool, int]:
    """-> (top service, top shift as printed, tie at top, number of listed services)."""
    lines = content.splitlines()
    idx = next((i for i, l in enumerate(lines) if l.strip().startswith(_TOP_HEADER)), None)
    if idx is None:
        raise ValueError("E001 has no 'Top 12 services' line")
    listed = []
    for l in lines[idx + 1:]:
        m = _OVERVIEW_LINE.match(l)
        if not m:
            break
        listed.append((m["svc"], m["shift"]))
    if not listed:
        return "", "", False, 0
    tie = len(listed) > 1 and float(listed[0][1]) == float(listed[1][1])
    return listed[0][0], listed[0][1], tie, len(listed)


def to_ledger(items: list[dict]) -> EvidenceLedger:
    out = []
    for d in items:
        try:
            out.append(EvidenceItem(id=d["id"], source=d["source"], args=d.get("args") or {},
                                    content=d["content"], created_at=d.get("created_at", 0.0)))
        except KeyError as e:
            raise ValueError(f"ledger item without {e}") from None
    return EvidenceLedger(items=out)


def ranking_top(items: list[dict], need_two: bool) -> tuple[str, int]:
    ranking = onset_shift_ranking(to_ledger(items))
    if not ranking or (need_two and len(ranking) < 2):
        return "", len(ranking)
    return ranking[0][0], len(ranking)


# ------------------------------------------------------------------ loading

def load_experiments(run_dirs: list[Path]):
    rows_at, case_list = {}, None
    for d in run_dirs:
        exp_path = d / "experiment.json"
        if not exp_path.exists():
            die(f"{exp_path} not found")
        exp = json.loads(exp_path.read_text())
        cases = list(exp["cases"])
        if case_list is None:
            case_list = cases
        elif sorted(cases) != sorted(case_list):
            die(f"case list in {exp_path} differs from the first --runs folder; not paired")
        for row in exp["rows"]:
            if (d / row).is_dir():
                if row in rows_at:
                    die(f"row {row} has a folder in both {rows_at[row]} and {d}")
                rows_at[row] = d
    missing = [r for r in ROWS if r not in rows_at]
    extra = sorted(set(rows_at) - set(ROWS))
    if missing or extra:
        die(f"expected rows {ROWS}; missing {missing}, unexpected {extra}")
    if len(set(case_list)) != len(case_list):
        die("duplicate case ids in experiment.json")
    return rows_at, sorted(case_list)


def find_json(row_dir: Path, case_id: str) -> Path | None:
    hits = [p for p in row_dir.glob(f"*/{case_id}.json") if p.parent.name != "llm_cache"]
    if len(hits) > 1:
        die(f"more than one JSON for {case_id} in {row_dir}: {hits}")
    return hits[0] if hits else None


def load_scores(run_dirs: list[Path]) -> dict:
    out = {}
    for d in run_dirs:
        for r in read_csv(d / "scores_cases.csv"):
            k = (r["row"], r["case_id"])
            if k in out:
                die(f"duplicate {k} in scores_cases.csv files")
            out[k] = r
    return out


def load_grounding(path: Path) -> dict:
    rows = read_csv(path)
    need = ["row", "case_id", "system", "status", "n_claims_expl", "grounding_expl"]
    miss = [c for c in need if c not in rows[0]]
    if miss:
        die(f"{path} is missing columns {miss}")
    out = {}
    for r in rows:
        k = (r["row"], r["case_id"])
        if k in out:
            die(f"duplicate {k} in {path}")
        if r["status"] not in STATUSES:
            die(f"{k}: unexpected judge status {r['status']!r} in {path}")
        out[k] = r
    return out


def build_case(row: str, case_id: str, truth: str, system: str, path: Path | None) -> tuple[dict, list[dict]]:
    """-> (master line, ledger items)."""
    m = {"row": row, "case_id": case_id, "system": system, "truth": truth,
         "missing_json": path is None}
    rec = json.loads(path.read_text()) if path else {}
    if path and rec.get("case_id") != case_id:
        die(f"{path} has case_id {rec.get('case_id')!r}, expected {case_id!r}")
    report = rec.get("report") or {}
    ranked = report.get("ranked_services") or []
    hyps = rec.get("hypotheses") or []
    verds = rec.get("critic_verdicts") or []
    ledger = rec.get("evidence_ledger") or []
    calls = rec.get("llm_calls") or []
    errors = [str(e) for e in (rec.get("errors") or [])]
    final = ranked[0] if ranked else ""
    m["final_root"] = final
    m["no_report"] = not ranked
    m["correct"] = bool(ranked) and norm(final) == norm(truth)
    m["rounds"] = rec.get("iteration_count") or 0
    m["n_hypotheses"], m["n_verdicts"] = len(hyps), len(verds)
    roots = [((h.get("ranked_services") or [""])[0] if isinstance(h, dict) else "") for h in hyps]
    decs = [str(v.get("decision", "")) for v in verds]
    for i in range(3):
        m[f"r{i + 1}_root"] = roots[i] if i < len(roots) else ""
        m[f"r{i + 1}_decision"] = decs[i] if i < len(decs) else ""
    m["round_path"] = " -> ".join(
        f"r{i + 1} {roots[i] or '?'}" + (f" ({decs[i]})" if i < len(decs) else "") for i in range(len(roots)))
    m["r1_correct"] = bool(roots) and norm(roots[0]) == norm(truth)
    loop = row not in BASELINE_ROWS
    last = decs[-1] if decs else ""
    m["last_decision"] = last if loop else ""
    m["accepted"] = (last == "accept") if loop else ""
    m["accepted_round"] = (len(decs) if last == "accept" else "") if loop else ""
    m["never_accepted"] = (bool(decs) and last != "accept") if loop else ""
    m["wrong_accepted"] = (last == "accept" and not m["correct"]) if loop else ""
    m["unchecked_accept"] = (last == "accept" and any(UNCHECKED in str(r) for r in (verds[-1].get("reasons") or []))) if loop else ""
    m["rule_skip"] = any(RULE_SKIP in str(r) for v in verds for r in (v.get("reasons") or [])) if loop else ""
    m["rule_skip_verdicts"] = sum(any(RULE_SKIP in str(r) for r in (v.get("reasons") or [])) for v in verds) if loop else ""
    m["parse_fail_verdicts"] = sum(any(PARSE_FAIL in str(r) for r in (v.get("reasons") or [])) for v in verds) if loop else ""
    m["parse_fail_unchecked"] = sum(any(PARSE_FAIL in str(r) for r in (v.get("reasons") or []))
                                    and any(UNCHECKED in str(r) for r in (v.get("reasons") or [])) for v in verds) if loop else ""
    m["r1_pushback"] = (bool(decs) and decs[0] in PUSH_BACK) if loop else ""
    m["correct_pushed_back"] = any(i < len(roots) and norm(roots[i]) == norm(truth) and d != "accept"
                                   for i, d in enumerate(decs)) if loop else ""
    m["n_ledger"] = len(ledger)
    m["n_calls"] = len(calls)
    m["llm_time_s"] = round(sum(float(c.get("latency_s") or 0) for c in calls), 3)
    m["critic_time_s"] = round(sum(float(c.get("latency_s") or 0) for c in calls if c.get("role") == "critic"), 3)
    m["has_errors"] = bool(errors)
    m["did_not_accept_error"] = any(NOT_ACCEPTED_ERR in e for e in errors)
    m["errors"] = " | ".join(errors)
    m["_hyp_verdict_len_mismatch"] = loop and bool(path) and len(hyps) != len(verds)
    m["_final_ne_last_hyp"] = bool(ranked) and bool(roots) and norm(final) != norm(roots[-1])
    m["_accept_not_last"] = loop and any(d == "accept" for d in decs[:-1])
    return m, ledger


# ------------------------------------------------------------------ reconcile

def reconcile(master: list[dict], by: dict, scores: dict, ground: dict, cases: list[str]) -> list[dict]:
    got: dict[str, object] = {}

    def rows_of(row):
        return [by[(row, c)] for c in cases]

    for row in ROWS:
        ms = rows_of(row)
        got[f"ac1/{row}"] = sum(m["correct"] for m in ms)
        for s in SYSTEMS:
            got[f"ac1_{s}/{row}"] = sum(m["correct"] for m in ms if m["system"].upper() == s)
        ran = [m for m in ms if not m["missing_json"]]
        got[f"mean_llm_time_s/{row}"] = f"{sum(m['llm_time_s'] for m in ran) / len(ran):.1f}" if ran else "n/a"
    h = rows_of("hybrid")
    got["accepted/hybrid"] = sum(m["accepted"] is True for m in h)
    for row in LOOP_ROWS:
        got[f"wrong_accepted/{row}"] = sum(m["wrong_accepted"] is True for m in rows_of(row))
    na = [m for m in h if m["never_accepted"] is True]
    got["never_accepted/hybrid"] = len(na)
    got["never_accepted_correct/hybrid"] = sum(m["correct"] for m in na)
    got["never_accepted_wrong/hybrid"] = sum(not m["correct"] for m in na)
    got["did_not_accept_error/hybrid"] = sum(m["did_not_accept_error"] for m in h)
    for row in LOOP_ROWS:
        pairs = [(by[(row, c)], by[("no_critic", c)]) for c in cases]
        got[f"fixed_vs_no_critic/{row}"] = sum(a["correct"] and not b["correct"] for a, b in pairs)
        got[f"broke_vs_no_critic/{row}"] = sum(b["correct"] and not a["correct"] for a, b in pairs)
    for row in ["no_critic", "code_rule", "llm_only", "hybrid_7b", "hybrid"]:
        pairs = [(by[(row, c)], by[("react", c)]) for c in cases]
        b_ = sum(a["correct"] and not r["correct"] for a, r in pairs)
        c_ = sum(r["correct"] and not a["correct"] for a, r in pairs)
        got[f"vs_react_row_only/{row}"], got[f"vs_react_react_only/{row}"] = b_, c_
        p = mcnemar_exact(b_, c_)
        dp = {"hybrid": 3, "code_rule": 1, "llm_only": 2, "hybrid_7b": 2}.get(row)
        got[f"vs_react_p/{row}"] = "<0.001" if dp is None and p < 0.001 else (
            f"{round(p, dp)}" if dp is not None else f"{p:.4f}")
    # 2x2 (A12 rule) and judge status
    status = Counter()
    for row in ROWS:
        cell = Counter()
        for c in cases:
            g = ground[(row, c)]
            status[g["status"]] += 1
            ok = by[(row, c)]["correct"]
            if g["status"] == "ok":
                if g["grounding_expl"] == "":
                    cell["excl_zero_expl"] += 1
                else:
                    cell["scored"] += 1
                    hi = float(g["grounding_expl"]) >= THRESHOLD - EPS
                    cell[("correct_" if ok else "wrong_") + ("high" if hi else "low")] += 1
            elif g["status"] == "parse_fail_label":
                cell["excl_parse_fail"] += 1
            else:
                cell["excl_no_report"] += 1
        for k in ["scored", "correct_high", "correct_low", "wrong_high", "wrong_low",
                  "excl_parse_fail", "excl_zero_expl", "excl_no_report"]:
            got[f"2x2_{k}/{row}"] = cell[k]
    for k in ["ok", "parse_fail_label", "no_report"]:
        got[f"judge_status_{k}/all"] = status[k]
    for row in LOOP_ROWS:
        z = 0
        for c in cases:
            a, b = ground[(row, c)], ground[("no_critic", c)]
            if a["status"] == b["status"] == "ok" and a["grounding_expl"] != "" and b["grounding_expl"] != "":
                z += abs(float(a["grounding_expl"]) - float(b["grounding_expl"])) < EPS
        got[f"zero_diff_vs_no_critic/{row}"] = z
    for row in ("hybrid", "code_rule"):
        got[f"r1_pushback/{row}"] = sum(m["r1_pushback"] is True for m in rows_of(row))
    for row in ("hybrid", "llm_only", "hybrid_7b"):
        cp = [m for m in rows_of(row) if m["correct_pushed_back"] is True]
        got[f"correct_pushed_back/{row}"] = len(cp)
        got[f"correct_pushed_back_final_wrong/{row}"] = sum(not m["correct"] for m in cp)
    uc = [m for m in h if m["unchecked_accept"] is True]
    got["unchecked_accept/hybrid"] = len(uc)
    got["unchecked_accept_correct/hybrid"] = sum(m["correct"] for m in uc)
    got["unchecked_accept/critic_rows"] = sum(m["unchecked_accept"] is True for r in LOOP_ROWS for m in rows_of(r))
    sk = [m for m in rows_of("code_rule") if m["rule_skip"] is True]
    got["rule_skip_cases/code_rule"] = len(sk)
    for s in SYSTEMS:
        got[f"rule_skip_cases_{s}/code_rule"] = sum(m["system"].upper() == s for m in sk)
    got["rule_skip_wrong/code_rule"] = sum(not m["correct"] for m in sk)
    got["truth_vs_case_id_mismatch/all"] = len({m["case_id"] for m in master
                                                if norm(root_from_case_id(m["case_id"])) != norm(m["truth"])})
    got["truth_differs_across_rows/all"] = sum(len({norm(by[(r, c)]["truth"]) for r in ROWS}) > 1 for c in cases)
    got["system_differs_across_rows/all"] = sum(len({by[(r, c)]["system"] for r in ROWS}) > 1 for c in cases)
    got["hyp_verdict_len_mismatch/all"] = sum(m["_hyp_verdict_len_mismatch"] for m in master)
    got["final_ne_last_hypothesis/all"] = sum(m["_final_ne_last_hyp"] for m in master)
    got["accept_not_last_round/all"] = sum(m["_accept_not_last"] for m in master)

    lines = []
    for key, (exp, src) in EXPECTED.items():
        g = got.get(key, "MISSING")
        lines.append({"check": key, "expected": exp, "got": g, "source": src,
                      "status": "ok" if str(g) == str(exp) else "MISMATCH"})
    # case-by-case vs scores_cases.csv
    fields = [("pred", "final_root", "str"), ("truth", "truth", "norm"), ("correct", "correct", "bool"),
              ("missing_json", "missing_json", "bool"), ("missing_report", "no_report", "bool"),
              ("rounds", "rounds", "int"), ("has_errors", "has_errors", "bool"),
              ("llm_time_s", "llm_time_s", "float")]
    critic_fields = [("critic_time_s", "critic_time_s", "float"), ("last_decision", "last_decision", "str"), ("wrong_accepted", "wrong_accepted", "bool"),
                     ("correct_rejected", "correct_pushed_back", "bool"),
                     ("rule_skips", "rule_skip_verdicts", "int"), ("parse_fails", "parse_fail_verdicts", "int"),
                     ("parse_fails_unchecked", "parse_fail_unchecked", "int")]
    bad = defaultdict(list)
    for m in master:
        s = scores.get((m["row"], m["case_id"]))
        if s is None:
            bad["scores_cases/missing_line"].append(f"{m['row']}/{m['case_id']}")
            continue
        if s.get("system") != m["system"]:
            bad["scores_cases/system"].append(f"{m['row']}/{m['case_id']}")
        flist = fields + ([] if m["row"] in BASELINE_ROWS else critic_fields)
        for sk_, mk, kind in flist:
            if sk_ not in s:
                bad[f"scores_cases/{sk_}"].append("column missing")
                continue
            sv, mv = s[sk_], m[mk]
            if kind == "bool":
                same = sv == str(bool(mv))
            elif kind == "int":
                same = sv != "" and int(float(sv)) == int(mv)
            elif kind == "float":
                same = sv != "" and abs(float(sv) - float(mv)) <= 0.0011
            elif kind == "norm":
                same = norm(sv) == norm(mv)
            else:
                same = sv == str(mv)
            if not same:
                bad[f"scores_cases/{sk_}"].append(f"{m['row']}/{m['case_id']} csv={sv!r} json={mv!r}")
    extra = set(scores) - {(m["row"], m["case_id"]) for m in master}
    if extra:
        bad["scores_cases/extra_lines"] = [f"{r}/{c}" for r, c in sorted(extra)]
    for key in ["scores_cases/missing_line", "scores_cases/system", "scores_cases/extra_lines"] + \
            [f"scores_cases/{f[0]}" for f in fields + critic_fields]:
        n = len(bad.get(key, []))
        lines.append({"check": key, "expected": 0, "got": n, "source": "scores_cases.csv, case by case",
                      "status": "ok" if n == 0 else "MISMATCH",
                      "examples": "; ".join(bad.get(key, [])[:5])})
    return lines


# ------------------------------------------------------------------ analyses

def categorize(pred: str, h: str, truth: str) -> str:
    pr, hr = norm(pred) == norm(truth) and pred != "", norm(h) == norm(truth) and h != ""
    if pr and hr:
        return "both_right"
    if pr:
        return "correction"
    if hr:
        return "regression"
    return "both_wrong_same" if pred != "" and norm(pred) == norm(h) else "both_wrong_diff"


CATS = ["both_right", "both_wrong_same", "both_wrong_diff", "correction", "regression"]


def h_e001_tables(by, cases, heur):
    out, changes = [], []
    for variant in ("a", "b"):
        for row in TELEMETRY_ROWS:
            for group in ["all"] + SYSTEMS:
                cs = [c for c in cases if group == "all" or by[(row, c)]["system"].upper() == group]
                cat = Counter()
                agree = r1_agree = 0
                for c in cs:
                    m, h = by[(row, c)], heur[c][variant]
                    cat[categorize(m["final_root"], h, m["truth"])] += 1
                    agree += h != "" and norm(m["final_root"]) == norm(h)
                    r1_agree += h != "" and norm(m["r1_root"]) == norm(h)
                    if variant == "a" and group == "all":
                        k = categorize(m["final_root"], h, m["truth"])
                        if k in ("correction", "regression"):
                            changes.append({"row": row, "category": k, "case_id": c, "system": m["system"],
                                            "truth": m["truth"], "h_e001a": h, "final_root": m["final_root"],
                                            "r1_root": m["r1_root"], "r1_eq_h_e001a": norm(m["r1_root"]) == norm(h),
                                            "rounds": m["rounds"], "round_path": m["round_path"],
                                            "accepted": m["accepted"], "unchecked_accept": m["unchecked_accept"],
                                            "rule_skip": m["rule_skip"], "h_final": m["h_final"],
                                            "final_eq_h_final": m["final_eq_h_final"],
                                            "listed_in_summary": row in LIST_ROWS})
                n = len(cs)
                out.append({"variant": variant, "row": row, "group": group, "n": n,
                            "h_correct": sum(norm(heur[c][variant]) == norm(by[(row, c)]["truth"]) and heur[c][variant] != "" for c in cs),
                            "row_correct": sum(by[(row, c)]["correct"] for c in cs),
                            "agree": agree, "agree_rate": round(agree / n, 4) if n else "",
                            "r1_agree": r1_agree, **{k: cat[k] for k in CATS}})
    return out, changes


def h_final_table(by, cases):
    out = []
    for row in ROWS:
        ms = [by[(row, c)] for c in cases]
        have = [m for m in ms if m["h_final"] != ""]
        ag = [m for m in have if m["final_eq_h_final"] is True]
        dis = [m for m in have if m["final_eq_h_final"] is False]
        out.append({"row": row, "n": len(ms), "with_h_final": len(have), "no_h_final": len(ms) - len(have),
                    "agree": len(ag), "agree_rate": round(len(ag) / len(have), 4) if have else "",
                    "correct_when_agree": sum(m["correct"] for m in ag),
                    "ac1_when_agree": round(sum(m["correct"] for m in ag) / len(ag), 4) if ag else "",
                    "disagree": len(dis), "correct_when_disagree": sum(m["correct"] for m in dis),
                    "ac1_when_disagree": round(sum(m["correct"] for m in dis) / len(dis), 4) if dis else "",
                    "h_final_correct": sum(m["h_final_correct"] is True for m in ms)})
    return out


def pred_correct(name: str, c: str, by, heur) -> bool:
    if name.startswith("H_E001"):
        h = heur[c][name[-1]]
        return h != "" and norm(h) == norm(by[("hybrid", c)]["truth"])
    return by[(name, c)]["correct"]


def mcnemar_table(by, cases, heur):
    def one(a, b, family):
        pa = [pred_correct(a, c, by, heur) for c in cases]
        pb = [pred_correct(b, c, by, heur) for c in cases]
        x = sum(i and not j for i, j in zip(pa, pb))
        y = sum(j and not i for i, j in zip(pa, pb))
        return {"family": family, "A": a, "B": b, "n": len(cases),
                "A_correct": sum(pa), "B_correct": sum(pb),
                "both_right": sum(i and j for i, j in zip(pa, pb)), "A_only": x, "B_only": y,
                "both_wrong": sum(not i and not j for i, j in zip(pa, pb)),
                "p_exact": mcnemar_exact(x, y)}
    fam = [one(a, b, "critic_attribution") for a, b in FAMILY]
    for r, adj in zip(fam, holm([r["p_exact"] for r in fam])):
        r["p_holm"] = adj
        r["significant_holm_0.05"] = adj < 0.05
    rest = [one(a, b, "descriptive_H_E001b") for a, b in DESCRIPTIVE_B] + \
           [one(a, b, "reproduction_vs_react") for a, b in REPRO_REACT]
    for r in rest:
        r["p_holm"], r["significant_holm_0.05"] = "", ""
    out = fam + rest
    for r in out:
        r["p_exact"] = round(r["p_exact"], 6)
        if r["p_holm"] != "":
            r["p_holm"] = round(r["p_holm"], 6)
    return out


def skip_split(by, cases):
    out = []
    for group in ["all"] + SYSTEMS:
        for skipped in (True, False):
            cs = [c for c in cases if (group == "all" or by[("hybrid", c)]["system"].upper() == group)
                  and (by[("code_rule", c)]["rule_skip"] is True) == skipped]
            h_only = sum(by[("hybrid", c)]["correct"] and not by[("code_rule", c)]["correct"] for c in cs)
            c_only = sum(by[("code_rule", c)]["correct"] and not by[("hybrid", c)]["correct"] for c in cs)
            out.append({"group": group, "code_rule_rule_skip": skipped, "n": len(cs),
                        "hybrid_only_correct": h_only, "code_rule_only_correct": c_only,
                        "both_right": sum(by[("hybrid", c)]["correct"] and by[("code_rule", c)]["correct"] for c in cs),
                        "both_wrong": sum(not by[("hybrid", c)]["correct"] and not by[("code_rule", c)]["correct"] for c in cs),
                        "cases_hybrid_only": " ".join(c for c in cs if by[("hybrid", c)]["correct"] and not by[("code_rule", c)]["correct"]),
                        "cases_code_rule_only": " ".join(c for c in cs if by[("code_rule", c)]["correct"] and not by[("hybrid", c)]["correct"])})
    return out


# ------------------------------------------------------------------ summary

def summary_md(recon, acks, ack_note, heur_stats, he, changes, hf, mc, sk, cases) -> str:
    L = ["# Critic attribution (post hoc, A13): summary", "",
         f"Cases: {len(cases)}. Rows: {', '.join(ROWS)}. Generated by scripts/critic_attribution.py.", ""]
    n_ok = sum(r["status"] == "ok" for r in recon)
    L.append(f"## Reconcile\n{n_ok}/{len(recon)} checks ok" + (f"; acknowledged: {', '.join(acks)} ({ack_note})" if acks else "; none acknowledged") + "\n")
    L.append(f"## H_E001 AC@1 (all {len(cases)} / SS / TT)")
    for v in ("a", "b"):
        s = heur_stats[v]
        L.append(f"- {v}: {s['all']} / {s['SS']} / {s['TT']}")
    L.append(f"- a != b in {heur_stats['a_ne_b']} cases; a empty {heur_stats['a_empty']}, b empty {heur_stats['b_empty']}; "
             f"tie at top of E001 in {heur_stats['ties']} cases\n")
    L.append(f"## Case by case vs H_E001a (all {len(cases)} cases)")
    L.append("| row | agree | both right | both wrong same | both wrong diff | corrections | regressions | r1 = H |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in he:
        if r["variant"] == "a" and r["group"] == "all":
            L.append(f"| {r['row']} | {r['agree']} ({r['agree_rate']:.0%}) | {r['both_right']} | {r['both_wrong_same']} "
                     f"| {r['both_wrong_diff']} | {r['correction']} | {r['regression']} | {r['r1_agree']} |")
    hy = next(r for r in he if r["variant"] == "a" and r["group"] == "all" and r["row"] == "hybrid")
    L.append(f"\nReading rule (A13, cutoff {READING_CUTOFF}/60): hybrid agrees in {hy['agree']}/{len(cases)} -> "
             + ("MET: \"largely converges on the onset screen's top service\" allowed." if hy["agree"] >= READING_CUTOFF
                else "NOT met: report the agreement rate without \"largely\".") + "\n")
    for row in LIST_ROWS:
        for k in ("correction", "regression"):
            cs = [c for c in changes if c["row"] == row and c["category"] == k]
            L.append(f"**{row} {k}s ({len(cs)})**" + ("" if cs else ": none"))
            for c in cs:
                L.append(f"- {c['case_id']}: H={c['h_e001a']}, final={c['final_root']}, {c['round_path']}")
        L.append("")
    L.append("## H_final agreement (diagnostic)")
    L.append("| row | with H_final | agree | AC@1 agree | AC@1 disagree | H_final correct |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in hf:
        L.append(f"| {r['row']} | {r['with_h_final']} | {r['agree']} | {r['correct_when_agree']}/{r['agree']} "
                 f"| {r['correct_when_disagree']}/{r['disagree']} | {r['h_final_correct']} |")
    L.append("\n## McNemar (exact, two-sided); family critic_attribution, Holm")
    L.append("| family | A | B | A correct | B correct | A only | B only | p | p_holm |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in mc:
        ph = "" if r["p_holm"] == "" else fmt_p(r["p_holm"]) + (" *" if r["significant_holm_0.05"] else "")
        L.append(f"| {r['family']} | {r['A']} | {r['B']} | {r['A_correct']} | {r['B_correct']} | {r['A_only']} "
                 f"| {r['B_only']} | {fmt_p(r['p_exact'])} | {ph} |")
    L.append("\n## Skip split: hybrid vs code_rule discordant cases")
    for r in sk:
        L.append(f"- {r['group']}, code_rule skip={r['code_rule_rule_skip']}: n={r['n']}, hybrid only {r['hybrid_only_correct']}, "
                 f"code_rule only {r['code_rule_only_correct']}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

EXPECTED = _expected()


def main():
    global EXPECTED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs/heldout", "runs/heldout_baselines"])
    ap.add_argument("--grounding", default="results/grounding_heldout/grounding_secondary_cases.csv")
    ap.add_argument("--out", default="results/critic_attribution")
    ap.add_argument("--ack", nargs="*", default=[], help="reconcile checks to accept after a written explanation")
    ap.add_argument("--ack-note", default="", help="the explanation (required with --ack)")
    ap.add_argument("--expected", default=None, help=argparse.SUPPRESS)  # tests only: JSON replacing EXPECTED
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    if a.ack and not a.ack_note.strip():
        die("--ack needs --ack-note with the explanation")
    if a.expected:
        EXPECTED = {k: tuple(v) for k, v in json.loads(Path(a.expected).read_text()).items()}
        print(f"NOTE: expected values from {a.expected} (test mode), not the held-out numbers")

    run_dirs = [Path(p) for p in a.runs]
    rows_at, cases = load_experiments(run_dirs)
    scores = load_scores(run_dirs)
    ground = load_grounding(Path(a.grounding))
    print(f"{len(cases)} cases, rows: " + ", ".join(f"{r} ({rows_at[r]})" for r in ROWS))

    master, by, ledgers = [], {}, {}
    for row in ROWS:
        for c in cases:
            s = scores.get((row, c))
            if s is None:
                die(f"{row}/{c} not in scores_cases.csv")
            m, led = build_case(row, c, s["truth"], s["system"], find_json(rows_at[row] / row, c))
            master.append(m)
            by[(row, c)] = m
            ledgers[(row, c)] = led
            g = ground.get((row, c))
            if g is None:
                die(f"{row}/{c} not in {a.grounding}")
            if g["system"] != m["system"]:
                die(f"{row}/{c}: system {m['system']!r} vs {g['system']!r} in grounding")
            m["judge_status"] = g["status"]
            m["n_claims_expl"] = g["n_claims_expl"]
            m["grounding_expl"] = g["grounding_expl"]
    only_g = set(ground) - set(by)
    if only_g:
        die(f"{len(only_g)} (row, case) only in {a.grounding}, e.g. {sorted(only_g)[:2]}")

    # E001 identity across telemetry rows, then heuristics
    heur, heur_stats = {}, {}
    for c in cases:
        firsts = {}
        for row in TELEMETRY_ROWS:
            led = ledgers[(row, c)]
            if not led or led[0].get("id") != "E001" or led[0].get("source") != "metric_overview":
                die(f"{row}/{c}: first ledger item is not E001 metric_overview")
            firsts[row] = (led[0]["source"], json.dumps(led[0].get("args"), sort_keys=True), led[0]["content"])
        if len(set(firsts.values())) != 1:
            diff = [r for r in TELEMETRY_ROWS if firsts[r] != firsts[TELEMETRY_ROWS[0]]]
            die(f"{c}: E001 differs across rows ({TELEMETRY_ROWS[0]} vs {diff}); H_E001 would not be one baseline")
        e001 = ledgers[(TELEMETRY_ROWS[0], c)][0]
        try:
            top_a, shift_a, tie, n_listed = e001_top_a(e001["content"])
            top_b, n_b = ranking_top([e001], need_two=False)
        except ValueError as e:
            die(f"{c}: {e}")
        heur[c] = {"a": top_a, "b": top_b, "a_shift": shift_a, "tie": tie, "n_listed": n_listed, "n_b": n_b}
    for m in master:
        h = heur[m["case_id"]]
        tele = m["row"] in TELEMETRY_ROWS
        m["e001_top_a"] = h["a"] if tele else ""
        m["e001_top_a_shift"] = h["a_shift"] if tele else ""
        m["e001_top_tie"] = h["tie"] if tele else ""
        m["e001_top_b"] = h["b"] if tele else ""
        m["e001_n_ranked_b"] = h["n_b"] if tele else ""
        m["h_e001a_correct"] = (h["a"] != "" and norm(h["a"]) == norm(m["truth"])) if tele else ""
        m["h_e001b_correct"] = (h["b"] != "" and norm(h["b"]) == norm(m["truth"])) if tele else ""
        m["final_eq_h_e001a"] = (h["a"] != "" and norm(m["final_root"]) == norm(h["a"])) if tele else ""
        m["final_eq_h_e001b"] = (h["b"] != "" and norm(m["final_root"]) == norm(h["b"])) if tele else ""
        m["r1_eq_h_e001a"] = (h["a"] != "" and norm(m["r1_root"]) == norm(h["a"])) if tele else ""
        try:
            hf, n_rank = ranking_top(ledgers[(m["row"], m["case_id"])], need_two=True)
        except ValueError as e:
            die(f"{m['row']}/{m['case_id']}: {e}")
        m["h_final"], m["h_final_n_ranked"] = hf, n_rank
        m["h_final_correct"] = (norm(hf) == norm(m["truth"])) if hf else ""
        m["final_eq_h_final"] = (norm(m["final_root"]) == norm(hf)) if hf else ""
    for v in ("a", "b"):
        heur_stats[v] = {"all": sum(heur[c][v] != "" and norm(heur[c][v]) == norm(by[("hybrid", c)]["truth"]) for c in cases)}
        for s in SYSTEMS:
            heur_stats[v][s] = sum(heur[c][v] != "" and norm(heur[c][v]) == norm(by[("hybrid", c)]["truth"])
                                   for c in cases if by[("hybrid", c)]["system"].upper() == s)
    heur_stats["a_ne_b"] = sum(norm(heur[c]["a"]) != norm(heur[c]["b"]) for c in cases)
    heur_stats["a_empty"] = sum(heur[c]["a"] == "" for c in cases)
    heur_stats["b_empty"] = sum(heur[c]["b"] == "" for c in cases)
    heur_stats["ties"] = sum(heur[c]["tie"] for c in cases)

    # Step 0 reconcile: stop on mismatch
    recon = reconcile(master, by, scores, ground, cases)
    mism = [r["check"] for r in recon if r["status"] == "MISMATCH"]
    stale = [k for k in a.ack if k not in mism]
    if stale:
        die(f"--ack for checks that did not fail: {stale}; remove them")
    print(f"\nreconcile: {len(recon) - len(mism)}/{len(recon)} ok")
    if mism:
        for r in recon:
            if r["status"] == "MISMATCH":
                ex = f"  e.g. {r['examples']}" if r.get("examples") else ""
                tag = "ACK " if r["check"] in a.ack else ""
                print(f"  {tag}MISMATCH {r['check']}: expected {r['expected']}, got {r['got']} ({r['source']}){ex}")
        open_ = [k for k in mism if k not in a.ack]
        if open_:
            sys.exit(f"STOP: {len(open_)} reconcile mismatch(es); nothing written. Explain each before going on "
                     f"(then --ack <check> ... --ack-note \"...\").")
        for r in recon:
            if r["check"] in a.ack:
                r["status"] = "ack"
                r["ack_note"] = a.ack_note

    he, changes = h_e001_tables(by, cases, heur)
    hft = h_final_table(by, cases)
    mc = mcnemar_table(by, cases, heur)
    sk = skip_split(by, cases)
    summ = summary_md(recon, a.ack, a.ack_note, heur_stats, he, changes, hft, mc, sk, cases)

    for m in master:
        for k in [k for k in m if k.startswith("_")]:
            del m[k]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "master_cases.csv", master)
    write_csv(out / "reconcile.csv", recon)
    write_csv(out / "h_e001_by_row.csv", he)
    write_csv(out / "h_e001_changes.csv", changes or [{"row": "", "category": "none"}])
    write_csv(out / "h_final_by_row.csv", hft)
    write_csv(out / "mcnemar.csv", mc)
    write_csv(out / "skip_split.csv", sk)
    (out / "summary.md").write_text(summ)
    print("\n" + summ)
    print(f"wrote {out}/ (7 CSVs + summary.md)")


if __name__ == "__main__":
    main()
