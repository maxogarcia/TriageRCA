#!/usr/bin/env python3
"""Grounding review (post hoc, Addendum A13.2): paired CIs, claim counts, judge error analysis,
wrong + high grounding examples.

Read-only. No LLM calls. Reads saved CSVs and writes CSVs + summary.md to a NEW --out.

Inputs (repo-relative defaults):
  --secondary    results/grounding_heldout/grounding_secondary_cases.csv   (per-case scores, A10 bounds)
  --claims       results/grounding_heldout/grounding_claims.csv            (judge per-claim labels)
  --tests        results/grounding_heldout/grounding_tests.csv             (A11; used as a check only)
  --master       results/critic_attribution/master_cases.csv               (Phase 1 source of truth)
  --cal-key      results/grounding_dev/calibration_key.csv                 (judge labels, 18 claims)
  --cal-labeled  the 3 pass-1 human label files, one per labeler          (required)
  --cal-names    labeler names in the same order                          (required; aeon daniel max)

Step 0 (before any analysis): reproduce the reported numbers (EXPECTED below, each with its source
doc). Any mismatch stops the script with nothing written.

Analyses (A13.2):
  1. paired bootstrap 95% CI of the mean difference (zero differences kept), each row vs react and
     each critic row vs no_critic, grounding_expl + supported_rate_expl; A10 bound sensitivity.
  2. explanation length: n_claims_expl per row, paired difference vs no_critic / react, claim-type
     mix, raw match mix (yes / partly / no / absent), grounding vs length.
  3. judge error analysis on the dev calibration: confusion matrices, false support, per labeler
     and vs the 3-human majority, all claims and explanation claims.
  4. wrong + high grounding examples (llm_only, code_rule), fixed pick rule.

Outputs (--out, must be new or empty):
  checks.csv, ci_paired.csv, length_by_row.csv, length_paired.csv, length_hist.csv,
  claim_mix.csv, length_vs_grounding.csv, cal_agreement.csv, cal_confusion.csv, cal_claims.csv,
  cal_false_support.csv, wrong_high_cases.csv, wrong_high_claims.csv, summary.md

Example (from repo root):
  python scripts/grounding_review.py --out results/grounding_review \\
      --cal-labeled results/grounding_dev/calibration_labeled.csv \\
                    results/grounding_dev/calibration_labeled_daniel.csv \\
                    results/grounding_dev/calibration_labeled_max.csv \\
      --cal-names aeon daniel max
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROWS = ["hybrid", "llm_only", "hybrid_7b", "code_rule", "no_critic", "react", "retrieval_only"]
VS_REACT = ["hybrid", "hybrid_7b", "llm_only", "code_rule", "no_critic", "retrieval_only"]
VS_NC = ["hybrid", "hybrid_7b", "llm_only", "code_rule"]
NOT_TESTED_IN_A11 = {"retrieval_only"}
METRICS = [("grounding_expl", "primary"), ("supported_rate_expl", "secondary")]
LABELS = ["Supported", "Partially Supported", "Unsupported"]
SHORT = {"Supported": "S", "Partially Supported": "P", "Unsupported": "U"}
MATCHES = ["yes", "partly", "no", "absent"]
CLAIM_TYPES = ["root-cause", "observational", "causal"]
FAIL = "PARSE_FAIL"
STATUSES = {"ok", "parse_fail_label", "no_report"}
THRESHOLD = 0.75           # A12 high grounding
EPS = 1e-9
BINS = [(1, 2, "1-2"), (3, 4, "3-4"), (5, 6, "5-6"), (7, 10 ** 6, "7+")]
WRONG_HIGH_ROWS = ["llm_only", "code_rule"]
N_PICK = 5
KEY = ("row", "case_id", "claim_id")
# Fixed structured root-cause claims (judge_grounding.structured_claims); everything else = explanation.
STRUCTURED_RE = [re.compile(r"^.+ is the root-cause service of this incident\.$"),
                 re.compile(r"^The fault category is .+\.$"),
                 re.compile(r"^The anomaly indicator is .+\.$")]

# ------------------------------------------------------------------ expected (already reported)
# G = heldout-grounding-results.md, H = heldout-baselines-results.md, A = addendum A2 / A7,
# P1 = fix-plan Phase 1 results.
_MEAN_EXPL = {"retrieval_only": "0.771", "llm_only": "0.764", "hybrid": "0.742", "react": "0.718",
              "hybrid_7b": "0.714", "code_rule": "0.711", "no_critic": "0.670"}
_MEAN_SR = {"retrieval_only": "0.698", "llm_only": "0.677", "hybrid": "0.668", "react": "0.628",
            "hybrid_7b": "0.607", "code_rule": "0.606", "no_critic": "0.554"}
_OK_FAIL = {"retrieval_only": (59, 0), "llm_only": (55, 5), "hybrid": (52, 8), "react": (52, 8),
            "hybrid_7b": (60, 0), "code_rule": (59, 1), "no_critic": (56, 4)}
_AC = {"hybrid": 48, "llm_only": 42, "hybrid_7b": 42, "code_rule": 37, "no_critic": 15, "react": 36,
       "retrieval_only": 5}


def _expected() -> dict:
    e = {}
    for k, v in {"ok": 393, "parse_fail_label": 26, "no_report": 1}.items():
        e[f"status_{k}/all"] = (v, "G run status")
    for row in ROWS:
        e[f"mean_expl/{row}"] = (_MEAN_EXPL[row], "G per-row table (3 dp)")
        e[f"mean_sr/{row}"] = (_MEAN_SR[row], "G per-row table (3 dp)")
        e[f"n_ok/{row}"] = (_OK_FAIL[row][0], "G per-row table")
        e[f"n_parse_fail/{row}"] = (_OK_FAIL[row][1], "G per-row table")
        e[f"ac1/{row}"] = (_AC[row], "H / P1 master table")
    e["expl_scored/retrieval_only"] = (55, "G note (4 ok cases have no explanation claims)")
    e["parse_fail_claims/all"] = (43, "G run status")
    e["parse_fail_claims_expl/all"] = (34, "G run status")
    for row, n in {"hybrid": 11, "hybrid_7b": 17, "llm_only": 10, "code_rule": 29}.items():
        e[f"zero_diff_vs_no_critic/{row}"] = (n, "G A11 zero-difference pairs")
    for row, d in {"llm_only": "0.107", "hybrid": "0.087", "hybrid_7b": "0.037", "code_rule": "0.040"}.items():
        e[f"a11_mean_diff_primary_vs_no_critic/{row}"] = (d, "G A11 (3 dp)")
    for row, d in {"hybrid": "0.137", "llm_only": "0.136"}.items():
        e[f"a11_mean_diff_secondary_vs_no_critic/{row}"] = (d, "G A11 (3 dp)")
    e["wrong_high/llm_only"] = (11, "G 2x2")
    e["wrong_high/code_rule"] = (12, "G 2x2")
    # calibration (A2), names as passed with --cal-names
    e["cal_claims/all"] = (18, "A2")
    e["cal_cases/all"] = (6, "A2")
    e["cal_claims_expl/all"] = (11, "A2 diagnostic")
    e["cal_no_majority/all"] = (1, "A2 reference label")
    for nm, (k, ag) in {"aeon": ("-0.075", "6/18"), "daniel": ("0.118", "8/18"), "max": ("0.130", "8/18")}.items():
        e[f"cal_kappa3_all/{nm}"] = (k, "A2 table (3 dp)")
        e[f"cal_agree_all/{nm}"] = (ag, "A2 table")
    for nm, (k3, kb, agb) in {"aeon": ("-0.055", "0.127", "6/11"), "daniel": ("0.312", "0.633", "9/11"),
                              "max": ("0.321", "0.814", "10/11"), "majority": ("0.219", "0.600", "8/10")}.items():
        e[f"cal_kappa3_expl/{nm}"] = (k3, "A2 diagnostic (3 dp)")
        e[f"cal_kappa_svn_expl/{nm}"] = (kb, "A2 diagnostic (3 dp)")
        e[f"cal_agree_svn_expl/{nm}"] = (agb, "A2 diagnostic")
    e["cal_fleiss/humans"] = ("0.636", "A2 table (3 dp)")
    e["cal_all_agree/humans"] = ("14/18", "A2 table")
    return e


EXPECTED = _expected()


# ------------------------------------------------------------------ helpers

def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def norm(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def read_csv(path: Path, need: list[str] | None = None) -> list[dict]:
    if not path.exists():
        die(f"missing {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die(f"{path} is empty")
    if need:
        miss = [c for c in need if c not in rows[0]]
        if miss:
            die(f"{path} is missing columns {miss}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        rows = [{"note": "none"}]
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def f3(x) -> str:
    if x is None or x == "":
        return "n/a"
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


def mean(xs):
    return statistics.mean(xs) if xs else None


def pctl(sorted_xs: list[float], q: float) -> float:
    """Percentile with linear interpolation (numpy default)."""
    n = len(sorted_xs)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def boot_ci(diffs: list[float], name: str, B: int, seed: int) -> tuple[float | None, float | None]:
    n = len(diffs)
    if n == 0:
        return None, None
    rng = random.Random(seed + zlib.crc32(name.encode()))
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return pctl(means, 0.025), pctl(means, 0.975)


def kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return None if abs(pe - 1) < EPS else (po - pe) / (1 - pe)


def fleiss(items: list[list[str]]) -> float | None:
    N = len(items)
    if not N:
        return None
    n = len(items[0])
    counts = [Counter(it) for it in items]
    p_bar = sum((sum(c[l] ** 2 for l in LABELS) - n) / (n * (n - 1)) for c in counts) / N
    p_j = [sum(c[l] for c in counts) / (N * n) for l in LABELS]
    pe = sum(p ** 2 for p in p_j)
    return None if abs(pe - 1) < EPS else (p_bar - pe) / (1 - pe)


def svn(label: str) -> str:
    return "S" if label == "Supported" else "notS"


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if sx == 0 or sy == 0 else sxy / (sx * sy)


def is_structured(claim: str, claim_type: str) -> bool:
    return claim_type == "root-cause" and any(p.match(claim.strip()) for p in STRUCTURED_RE)


def fl(x: str) -> float | None:
    return None if x == "" else float(x)


# ------------------------------------------------------------------ loading

def load_secondary(path: Path) -> dict:
    need = ["row", "case_id", "system", "status", "n_claims_expl", "n_fail_expl", "grounding_expl",
            "supported_rate_expl", "grounding_expl_lo", "grounding_expl_hi",
            "supported_rate_expl_lo", "supported_rate_expl_hi"]
    out = {}
    for r in read_csv(path, need):
        k = (r["row"], r["case_id"])
        if k in out:
            die(f"duplicate {k} in {path}")
        if r["status"] not in STATUSES:
            die(f"{k}: unexpected judge status {r['status']!r} in {path}")
        out[k] = r
    rows = {r for r, _ in out}
    if rows != set(ROWS):
        die(f"{path}: rows {sorted(rows)} != expected {sorted(ROWS)}")
    return out


def load_claims(path: Path) -> dict:
    need = ["row", "case_id", "claim_id", "claim_type", "origin", "claim", "match", "label", "verified_label"]
    out = defaultdict(list)
    seen = set()
    for r in read_csv(path, need):
        k = (r["row"], r["case_id"], r["claim_id"])
        if k in seen:
            die(f"duplicate claim {k} in {path}")
        seen.add(k)
        if r["origin"] not in ("structured", "explanation"):
            die(f"{k}: unexpected origin {r['origin']!r}")
        if r["verified_label"] not in LABELS + [FAIL]:
            die(f"{k}: unexpected verified_label {r['verified_label']!r}")
        if r["verified_label"] != FAIL and r["match"] not in MATCHES:
            die(f"{k}: unexpected match {r['match']!r}")
        out[(r["row"], r["case_id"])].append(r)
    return out


def load_master(path: Path) -> dict:
    need = ["row", "case_id", "system", "truth", "final_root", "correct", "e001_top_a"]
    out = {}
    for r in read_csv(path, need):
        k = (r["row"], r["case_id"])
        if k in out:
            die(f"duplicate {k} in {path}")
        if r["correct"] not in ("True", "False"):
            die(f"{k}: correct={r['correct']!r} in {path}")
        out[k] = r
    return out


# ------------------------------------------------------------------ step 0

def step0(sec, claims, master, tests_path: Path | None) -> tuple[dict, list[str]]:
    got: dict[str, object] = {}
    notes = []
    st = Counter(r["status"] for r in sec.values())
    for k in ("ok", "parse_fail_label", "no_report"):
        got[f"status_{k}/all"] = st[k]
    for row in ROWS:
        rs = [r for (rw, _), r in sec.items() if rw == row]
        got[f"n_ok/{row}"] = sum(r["status"] == "ok" for r in rs)
        got[f"n_parse_fail/{row}"] = sum(r["status"] == "parse_fail_label" for r in rs)
        ge = [float(r["grounding_expl"]) for r in rs if r["grounding_expl"] != ""]
        sr = [float(r["supported_rate_expl"]) for r in rs if r["supported_rate_expl"] != ""]
        got[f"mean_expl/{row}"] = f3(mean(ge))
        got[f"mean_sr/{row}"] = f3(mean(sr))
        if row == "retrieval_only":
            got["expl_scored/retrieval_only"] = len(ge)
        ms = [m for (rw, _), m in master.items() if rw == row]
        got[f"ac1/{row}"] = sum(m["correct"] == "True" for m in ms)
    allc = [c for cl in claims.values() for c in cl]
    fails = [c for c in allc if c["verified_label"] == FAIL]
    got["parse_fail_claims/all"] = len(fails)
    got["parse_fail_claims_expl/all"] = sum(c["origin"] == "explanation" for c in fails)
    for row in VS_NC:
        z = 0
        for (rw, cid), r in sec.items():
            if rw != row:
                continue
            b = sec.get(("no_critic", cid))
            if b and r["grounding_expl"] != "" and b["grounding_expl"] != "":
                z += abs(float(r["grounding_expl"]) - float(b["grounding_expl"])) < EPS
        got[f"zero_diff_vs_no_critic/{row}"] = z
    for metric, role in METRICS:
        for row in VS_NC:
            d = paired(sec, row, "no_critic", metric)
            got[f"a11_mean_diff_{role}_vs_no_critic/{row}"] = f3(mean([x for x, _ in d]))
    for row in WRONG_HIGH_ROWS:
        got[f"wrong_high/{row}"] = sum(
            1 for (rw, cid), r in sec.items() if rw == row and r["status"] == "ok" and r["grounding_expl"] != ""
            and float(r["grounding_expl"]) >= THRESHOLD - EPS and master[(rw, cid)]["correct"] == "False")
    # consistency: case sets, claims vs secondary
    bad = defaultdict(list)
    if set(sec) != set(master):
        bad["case_sets_secondary_vs_master"] = [str(k) for k in sorted(set(sec) ^ set(master))][:5]
    for k, r in sec.items():
        m = master.get(k)
        if m and m["system"] != r["system"]:
            bad["system_secondary_vs_master"].append(str(k))
        cl = claims.get(k, [])
        n_expl = sum(c["origin"] == "explanation" for c in cl)
        if r["status"] in ("ok", "parse_fail_label") and str(n_expl) != r["n_claims_expl"]:
            bad["n_claims_expl_vs_claims_csv"].append(f"{k} csv={r['n_claims_expl']} claims={n_expl}")
        if r["status"] == "no_report" and cl:
            bad["no_report_with_claims"].append(str(k))
        if r["status"] == "ok" and r["grounding_expl"] != "":
            ex = [c for c in cl if c["origin"] == "explanation"]
            w = {"Supported": 1.0, "Partially Supported": 0.5, "Unsupported": 0.0}
            g = sum(w[c["verified_label"]] for c in ex) / len(ex)
            if abs(g - float(r["grounding_expl"])) > 1e-4:
                bad["grounding_expl_vs_claims_csv"].append(f"{k} csv={r['grounding_expl']} claims={g:.4f}")
    orphan = set(claims) - set(sec)
    if orphan:
        bad["claims_without_case"] = [str(k) for k in sorted(orphan)][:5]
    for name in ["case_sets_secondary_vs_master", "system_secondary_vs_master", "n_claims_expl_vs_claims_csv",
                 "no_report_with_claims", "grounding_expl_vs_claims_csv", "claims_without_case"]:
        EXPECTED.setdefault(f"consistency/{name}", (0, "internal consistency"))
        got[f"consistency/{name}"] = len(bad.get(name, []))
        if bad.get(name):
            notes.append(f"{name}: " + "; ".join(bad[name][:3]))
    # A11 tests file: every comparison's n_pairs and mean_diff must match exactly
    if tests_path is not None:
        tr = read_csv(tests_path, ["metric", "family", "row", "reference", "n_pairs", "n_zero", "mean_diff"])
        mism = []
        for t in tr:
            d = paired(sec, t["row"], t["reference"], t["metric"])
            md = round(statistics.mean([x for x, _ in d]), 4) if d else ""
            nz = sum(abs(x) < EPS for x, _ in d)
            if str(len(d)) != t["n_pairs"] or str(nz) != t["n_zero"] or \
                    (md != "" and abs(md - float(t["mean_diff"])) > 1e-4):
                mism.append(f"{t['metric']} {t['row']} vs {t['reference']}: n {len(d)}/{t['n_pairs']}, "
                            f"zero {nz}/{t['n_zero']}, mean {md}/{t['mean_diff']}")
        EXPECTED.setdefault("consistency/a11_tests_file", (0, "grounding_tests.csv, every comparison"))
        got["consistency/a11_tests_file"] = len(mism)
        EXPECTED.setdefault("consistency/a11_tests_rows", (18, "grounding_tests.csv rows (2 metrics x 9)"))
        got["consistency/a11_tests_rows"] = len(tr)
        if mism:
            notes.append("a11: " + "; ".join(mism[:3]))
    return got, notes


def paired(sec, row, ref, metric) -> list[tuple[float, str]]:
    out = []
    for (rw, cid), r in sorted(sec.items()):
        if rw != row:
            continue
        b = sec.get((ref, cid))
        if b is None:
            die(f"{ref}/{cid} missing (pairing {row})")
        if r[metric] != "" and b[metric] != "":
            out.append((float(r[metric]) - float(b[metric]), cid))
    return out


def paired_bounds(sec, row, ref, metric, mode) -> list[float]:
    out = []
    for (rw, cid), r in sorted(sec.items()):
        if rw != row:
            continue
        b = sec[(ref, cid)]
        if r[f"{metric}_lo"] == "" or b[f"{metric}_lo"] == "":
            continue
        if mode == "pessimistic":
            out.append(float(r[f"{metric}_lo"]) - float(b[f"{metric}_hi"]))
        else:
            out.append(float(r[f"{metric}_hi"]) - float(b[f"{metric}_lo"]))
    return out


# ------------------------------------------------------------------ analyses

def ci_table(sec, B, seed) -> list[dict]:
    out = []
    comps = [("vs_react", r, "react") for r in VS_REACT] + [("vs_no_critic", r, "no_critic") for r in VS_NC]
    for metric, role in METRICS:
        for fam, row, ref in comps:
            d = [x for x, _ in paired(sec, row, ref, metric)]
            lo, hi = boot_ci(d, f"{metric}|{row}|{ref}|scored", B, seed)
            base = {"metric": metric, "role": role, "family": fam, "row": row, "reference": ref,
                    "in_a11": row not in NOT_TESTED_IN_A11}
            out.append({**base, "kind": "scored (primary)", "n_pairs": len(d),
                        "n_zero": sum(abs(x) < EPS for x in d),
                        "mean_diff": round(mean(d), 4) if d else "",
                        "ci_lo": round(lo, 4) if lo is not None else "",
                        "ci_hi": round(hi, 4) if hi is not None else "",
                        "ci_contains_0": (lo <= 0 <= hi) if lo is not None else ""})
            for mode in ("pessimistic", "optimistic"):
                d2 = paired_bounds(sec, row, ref, metric, mode)
                lo2, hi2 = boot_ci(d2, f"{metric}|{row}|{ref}|{mode}", B, seed)
                out.append({**base, "kind": f"A10 bounds, {mode} (sensitivity)", "n_pairs": len(d2),
                            "n_zero": sum(abs(x) < EPS for x in d2),
                            "mean_diff": round(mean(d2), 4) if d2 else "",
                            "ci_lo": round(lo2, 4) if lo2 is not None else "",
                            "ci_hi": round(hi2, 4) if hi2 is not None else "",
                            "ci_contains_0": (lo2 <= 0 <= hi2) if lo2 is not None else ""})
    return out


def judged(r) -> bool:
    return r["status"] in ("ok", "parse_fail_label")


def length_tables(sec, claims, B, seed):
    by_row, hist, paired_rows, mix, lvg = [], [], [], [], []
    for row in ROWS:
        rs = [r for (rw, _), r in sorted(sec.items()) if rw == row and judged(r)]
        ns = [int(r["n_claims_expl"]) for r in rs]
        by_row.append({"row": row, "n_judged_cases": len(rs), "mean": round(mean(ns), 3) if ns else "",
                       "median": statistics.median(ns) if ns else "", "min": min(ns) if ns else "",
                       "max": max(ns) if ns else "", "n_zero": sum(n == 0 for n in ns),
                       "total_expl_claims": sum(ns)})
        c = Counter(ns)
        for k in range(0, max(ns + [0]) + 1):
            hist.append({"row": row, "n_claims_expl": k, "cases": c[k]})
        ex = [cl for (rw, _), cls in claims.items() if rw == row for cl in cls if cl["origin"] == "explanation"]
        for t in CLAIM_TYPES + ["ALL"]:
            sub = [x for x in ex if t == "ALL" or x["claim_type"] == t]
            vl = Counter(x["verified_label"] for x in sub)
            mt = Counter(x["match"] for x in sub if x["verified_label"] != FAIL)
            q0 = sum(x.get("quote_ok", "") == "0" for x in sub)
            mix.append({"row": row, "claim_type": t, "n": len(sub),
                        "share": round(len(sub) / len(ex), 3) if ex and t != "ALL" else "",
                        "verified_S": vl["Supported"], "verified_P": vl["Partially Supported"],
                        "verified_U": vl["Unsupported"], "parse_fail": vl[FAIL],
                        "match_yes": mt["yes"], "match_partly": mt["partly"], "match_no": mt["no"],
                        "match_absent": mt["absent"], "quote_failed_to_U": q0})
        sc = [r for r in rs if r["status"] == "ok" and r["grounding_expl"] != ""]
        xs = [int(r["n_claims_expl"]) for r in sc]
        ys = [float(r["grounding_expl"]) for r in sc]
        rho = spearman(xs, ys)
        rec = {"row": row, "n_scored": len(sc), "spearman_n_claims_vs_grounding": round(rho, 3) if rho is not None else ""}
        for lo, hi, lab in BINS:
            g = [y for x, y in zip(xs, ys) if lo <= x <= hi]
            rec[f"bin_{lab}_n"] = len(g)
            rec[f"bin_{lab}_mean_grounding"] = round(mean(g), 3) if g else ""
        lvg.append(rec)
    # pooled
    sc = [r for r in sec.values() if r["status"] == "ok" and r["grounding_expl"] != ""]
    rho = spearman([int(r["n_claims_expl"]) for r in sc], [float(r["grounding_expl"]) for r in sc])
    rec = {"row": "ALL (pooled)", "n_scored": len(sc), "spearman_n_claims_vs_grounding": round(rho, 3) if rho is not None else ""}
    for lo, hi, lab in BINS:
        g = [float(r["grounding_expl"]) for r in sc if lo <= int(r["n_claims_expl"]) <= hi]
        rec[f"bin_{lab}_n"] = len(g)
        rec[f"bin_{lab}_mean_grounding"] = round(mean(g), 3) if g else ""
    lvg.append(rec)
    comps = [(r, "no_critic") for r in VS_NC] + [(r, "react") for r in VS_REACT]
    for row, ref in comps:
        d = []
        for (rw, cid), r in sorted(sec.items()):
            if rw == row and judged(r) and judged(sec[(ref, cid)]):
                d.append(int(r["n_claims_expl"]) - int(sec[(ref, cid)]["n_claims_expl"]))
        lo, hi = boot_ci([float(x) for x in d], f"nclaims|{row}|{ref}", B, seed)
        paired_rows.append({"row": row, "reference": ref, "n_pairs": len(d), "n_zero": sum(x == 0 for x in d),
                            "n_row_longer": sum(x > 0 for x in d), "n_ref_longer": sum(x < 0 for x in d),
                            "mean_diff": round(mean(d), 3) if d else "",
                            "ci_lo": round(lo, 3) if lo is not None else "",
                            "ci_hi": round(hi, 3) if hi is not None else ""})
    return by_row, hist, paired_rows, mix, lvg


# ------------------------------------------------------------------ calibration

def load_calibration(key_path: Path, labeled: list[Path], names: list[str]):
    key = {}
    for r in read_csv(key_path, ["row", "case_id", "claim_id", "claim_type", "claim",
                                 "judge_label", "judge_verified_label", "judge_match"]):
        k = tuple(r[c] for c in KEY)
        if k in key:
            die(f"duplicate {k} in {key_path}")
        for col, allowed in (("judge_label", LABELS), ("judge_verified_label", LABELS), ("judge_match", MATCHES)):
            if r[col] not in allowed:
                die(f"{key_path} {k}: {col}={r[col]!r}")
        key[k] = r
    human = {}
    for nm, p in zip(names, labeled):
        rows = read_csv(p, ["row", "case_id", "claim_id", "claim", "human_label"])
        by = {}
        for r in rows:
            k = tuple(r[c] for c in KEY)
            if k in by:
                die(f"duplicate {k} in {p}")
            if r["human_label"] not in LABELS:
                die(f"{p} {k}: human_label {r['human_label']!r}")
            if k not in key:
                die(f"{p} {k}: not in {key_path}")
            if r["claim"].strip() != key[k]["claim"].strip():
                die(f"{p} {k}: claim text differs from the key")
            by[k] = r
        if set(by) != set(key):
            die(f"{p}: labels {len(by)} claims, key has {len(key)}; every labeler must label the same set")
        human[nm] = by
    return key, human


def majority(labels: list[str]) -> str | None:
    c = Counter(labels).most_common()
    return c[0][0] if c[0][1] >= 2 else None


def calibration(key, human, names):
    ks = sorted(key, key=lambda k: (k[0], k[1], int(k[2]) if k[2].isdigit() else k[2]))
    expl = {k for k in ks if not is_structured(key[k]["claim"], key[k]["claim_type"])}
    maj = {k: majority([human[nm][k]["human_label"] for nm in names]) for k in ks}
    refs = {nm: {k: human[nm][k]["human_label"] for k in ks} for nm in names}
    refs["majority"] = {k: v for k, v in maj.items() if v is not None}
    got = {"cal_claims/all": len(ks), "cal_cases/all": len({k[:2] for k in ks}),
           "cal_claims_expl/all": len(expl), "cal_no_majority/all": sum(v is None for v in maj.values())}
    items = [[human[nm][k]["human_label"] for nm in names] for k in ks]
    got["cal_fleiss/humans"] = f3(fleiss(items))
    got["cal_all_agree/humans"] = f"{sum(len(set(i)) == 1 for i in items)}/{len(ks)}"
    agreement, confusion, fs_list = [], [], []
    for ref, labs in refs.items():
        for scope in ("all", "explanation"):
            kk = [k for k in ks if k in labs and (scope == "all" or k in expl)]
            for jkind, col in (("verified", "judge_verified_label"), ("raw", "judge_label")):
                pairs = [(labs[k], key[k][col]) for k in kk]
                bp = [(svn(a), svn(b)) for a, b in pairs]
                k3, kb = kappa(pairs, LABELS), kappa(bp, ["S", "notS"])
                hu = [k for k in kk if labs[k] == "Unsupported"]
                fs = [k for k in hu if key[k][col] == "Supported"]
                oc = [k for k in kk if labs[k] == "Partially Supported" and key[k][col] == "Supported"]
                uc = [k for k in kk if labs[k] == "Supported" and key[k][col] != "Supported"]
                agreement.append({"reference": ref, "scope": scope, "judge": jkind, "n": len(kk),
                                  "kappa_3": round(k3, 3) if k3 is not None else "",
                                  "agree_3": f"{sum(a == b for a, b in pairs)}/{len(pairs)}",
                                  "kappa_S_vs_not": round(kb, 3) if kb is not None else "",
                                  "agree_S_vs_not": f"{sum(a == b for a, b in bp)}/{len(bp)}",
                                  "human_U": len(hu), "false_support": len(fs),
                                  "false_support_rate": f"{len(fs)}/{len(hu)}" if hu else "0/0",
                                  "over_credit_P_to_S": len(oc), "under_credit_S_to_notS": len(uc),
                                  "false_support_claims": " ".join(f"{k[1]}#{k[2]}" for k in fs)})
                cm = Counter(pairs)
                for h in LABELS:
                    for j in LABELS:
                        confusion.append({"reference": ref, "scope": scope, "judge": jkind,
                                          "human": h, "judge_label": j, "count": cm[(h, j)]})
                if jkind == "verified":
                    if scope == "all":
                        got[f"cal_kappa3_all/{ref}"] = f3(k3)
                        got[f"cal_agree_all/{ref}"] = f"{sum(a == b for a, b in pairs)}/{len(pairs)}"
                    else:
                        got[f"cal_kappa3_expl/{ref}"] = f3(k3)
                        got[f"cal_kappa_svn_expl/{ref}"] = f3(kb)
                        got[f"cal_agree_svn_expl/{ref}"] = f"{sum(a == b for a, b in bp)}/{len(bp)}"
                    if scope == "all":
                        for k in fs:
                            fs_list.append({"reference": ref, "case_id": k[1], "claim_id": k[2],
                                            "claim_type": key[k]["claim_type"],
                                            "explanation_claim": k in expl, "claim": key[k]["claim"],
                                            "human_label": labs[k], "judge_verified": key[k]["judge_verified_label"],
                                            "judge_raw": key[k]["judge_label"], "judge_match": key[k]["judge_match"],
                                            "judge_quote_ok": key[k].get("judge_quote_ok", ""),
                                            "judge_evidence": key[k].get("judge_evidence", "")})
            # human x raw match (4 values), all claims and explanation
            cm4 = Counter((labs[k], key[k]["judge_match"]) for k in kk)
            for h in LABELS:
                for j in MATCHES:
                    confusion.append({"reference": ref, "scope": scope, "judge": "raw_match",
                                      "human": h, "judge_label": j, "count": cm4[(h, j)]})
    claims_rows = []
    for k in ks:
        r = {"case_id": k[1], "claim_id": k[2], "claim_type": key[k]["claim_type"],
             "explanation_claim": k in expl, "claim": key[k]["claim"]}
        for nm in names:
            r[f"{nm}"] = SHORT[human[nm][k]["human_label"]]
        r["majority"] = SHORT[maj[k]] if maj[k] else "none"
        r["judge_verified"] = SHORT[key[k]["judge_verified_label"]]
        r["judge_raw"] = SHORT[key[k]["judge_label"]]
        r["judge_match"] = key[k]["judge_match"]
        r["judge_quote_ok"] = key[k].get("judge_quote_ok", "")
        claims_rows.append(r)
    return got, agreement, confusion, claims_rows, fs_list


# ------------------------------------------------------------------ wrong + high

def wrong_high(sec, claims, master):
    cands = []
    for row in WRONG_HIGH_ROWS:
        for (rw, cid), r in sorted(sec.items()):
            if rw != row or r["status"] != "ok" or r["grounding_expl"] == "":
                continue
            m = master[(rw, cid)]
            if m["correct"] == "False" and float(r["grounding_expl"]) >= THRESHOLD - EPS:
                cands.append({"row": row, "case_id": cid, "system": r["system"], "truth": m["truth"],
                              "final_root": m["final_root"], "h_e001a": m["e001_top_a"],
                              "final_eq_h_e001a": norm(m["final_root"]) == norm(m["e001_top_a"]),
                              "grounding_expl": float(r["grounding_expl"]),
                              "supported_rate_expl": r["supported_rate_expl"],
                              "n_claims_expl": int(r["n_claims_expl"])})
    order = {row: sorted([c for c in cands if c["row"] == row],
                         key=lambda c: (-c["grounding_expl"], -c["n_claims_expl"], c["case_id"]))
             for row in WRONG_HIGH_ROWS}
    picked, i = [], 0
    while len(picked) < N_PICK and any(order.values()):
        row = WRONG_HIGH_ROWS[i % len(WRONG_HIGH_ROWS)]
        if order[row]:
            picked.append(order[row].pop(0))
        i += 1
    keys = {(c["row"], c["case_id"]) for c in picked}
    for n, c in enumerate(picked, 1):
        c["pick_order"] = n
    for c in cands:
        c["picked"] = (c["row"], c["case_id"]) in keys
        c.setdefault("pick_order", "")
    claim_rows = []
    for c in picked:
        for cl in sorted(claims[(c["row"], c["case_id"])], key=lambda x: int(x["claim_id"])):
            claim_rows.append({"pick_order": c["pick_order"], "row": c["row"], "case_id": c["case_id"],
                               "claim_id": cl["claim_id"], "origin": cl["origin"], "claim_type": cl["claim_type"],
                               "verified_label": cl["verified_label"], "match": cl["match"],
                               "quote_ok": cl.get("quote_ok", ""), "evidence": cl.get("evidence", ""),
                               "claim": cl["claim"], "evidence_says": (cl.get("evidence_says", "") or "")[:300]})
    return cands, picked, claim_rows


# ------------------------------------------------------------------ summary

def summary_md(checks, ci, lb, lp, mix, lvg, agr, fsl, cands, picked, wclaims, args, names) -> str:
    L = ["# Grounding review (post hoc, A13.2): summary", "",
         f"Generated by scripts/grounding_review.py; bootstrap B={args.boot}, seed {args.seed}, "
         "percentile 95% CI, case resampling, zero differences kept. Unadjusted intervals, not tests.", ""]
    acks = [c["check"] for c in checks if c["status"] == "ack"]
    L.append(f"## Step 0\n{sum(c['status'] == 'ok' for c in checks)}/{len(checks)} checks ok"
             + (f"; acknowledged: {', '.join(acks)} ({args.ack_note})" if acks else "; none acknowledged") + "\n")
    med = statistics.median([float(r["median"]) for r in lb if r["median"] != "" and r["row"] != "retrieval_only"])
    L.append(f"Scale aid: median explanation claims per case across non-retrieval rows ~{med:g}; "
             f"one claim moving S -> U changes a {med:g}-claim case score by {1 / med:.2f}.\n")
    for metric, role in METRICS:
        L.append(f"## Paired CIs, {metric} ({role})")
        L.append("| family | row | ref | kind | pairs | zero | mean diff | 95% CI |")
        L.append("|---|---|---|---|---:|---:|---:|---|")
        for r in ci:
            if r["metric"] == metric:
                tag = "" if r["in_a11"] else " (not in A11)"
                L.append(f"| {r['family']} | {r['row']}{tag} | {r['reference']} | {r['kind']} | {r['n_pairs']} "
                         f"| {r['n_zero']} | {r['mean_diff']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |"
                         if r["mean_diff"] != "" else f"| {r['family']} | {r['row']} | {r['reference']} | {r['kind']} | 0 | | | |")
        L.append("")
    L.append("## Explanation length (n_claims_expl, judged cases; exploratory)")
    L.append("| row | cases | mean | median | min | max | zero |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in lb:
        L.append(f"| {r['row']} | {r['n_judged_cases']} | {r['mean']} | {r['median']} | {r['min']} | {r['max']} | {r['n_zero']} |")
    L.append("\nPaired difference in n_claims_expl (row minus ref):")
    L.append("| row | ref | pairs | row longer | ref longer | same | mean diff | 95% CI |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in lp:
        L.append(f"| {r['row']} | {r['reference']} | {r['n_pairs']} | {r['n_row_longer']} | {r['n_ref_longer']} "
                 f"| {r['n_zero']} | {r['mean_diff']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] |")
    L.append("\nClaim-type mix and raw match (explanation claims; no = contradicted, absent = not in evidence):")
    L.append("| row | n | root-cause | observational | causal | S | P | U | fail | yes | partly | no | absent | quote fail -> U |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in ROWS:
        a = next(m for m in mix if m["row"] == row and m["claim_type"] == "ALL")
        t = {m["claim_type"]: m["n"] for m in mix if m["row"] == row}
        L.append(f"| {row} | {a['n']} | {t['root-cause']} | {t['observational']} | {t['causal']} | {a['verified_S']} "
                 f"| {a['verified_P']} | {a['verified_U']} | {a['parse_fail']} | {a['match_yes']} | {a['match_partly']} "
                 f"| {a['match_no']} | {a['match_absent']} | {a['quote_failed_to_U']} |")
    L.append("\nGrounding vs length (ok scored cases): Spearman rho, mean grounding_expl by bin (n):")
    L.append("| row | n | rho | 1-2 | 3-4 | 5-6 | 7+ |")
    L.append("|---|---:|---:|---|---|---|---|")
    for r in lvg:
        cells = [f"{r[f'bin_{lab}_mean_grounding']} ({r[f'bin_{lab}_n']})" for _, _, lab in BINS]
        L.append(f"| {r['row']} | {r['n_scored']} | {r['spearman_n_claims_vs_grounding']} | " + " | ".join(cells) + " |")
    L.append("\n## Judge error analysis (dev calibration, 18 claims, verified labels)")
    L.append("| reference | scope | n | kappa 3 | agree 3 | kappa S/not | agree S/not | human U | false support | P->S | S->not S |")
    L.append("|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|")
    for r in agr:
        if r["judge"] == "verified":
            L.append(f"| {r['reference']} | {r['scope']} | {r['n']} | {r['kappa_3']} | {r['agree_3']} | {r['kappa_S_vs_not']} "
                     f"| {r['agree_S_vs_not']} | {r['human_U']} | {r['false_support']} | {r['over_credit_P_to_S']} "
                     f"| {r['under_credit_S_to_notS']} |")
    L.append("\nRaw-label rows (no quote check) are in cal_agreement.csv; confusion matrices in cal_confusion.csv.")
    L.append(f"\nFalse-support claims (human U, judge verified S), all claims: {len(fsl)} (reference, claim)")
    for f in fsl:
        L.append(f"- {f['reference']}: {f['case_id']} #{f['claim_id']} [{f['claim_type']}] {f['claim']!r} "
                 f"(judge raw {f['judge_raw']}, match {f['judge_match']}, quote_ok {f['judge_quote_ok'] or '-'})")
    per_row = Counter(c["row"] for c in cands)
    L.append(f"\n## Wrong + high grounding (llm_only, code_rule; A12 cut 0.75): {len(cands)} candidates ("
             + ", ".join(f"{r} {per_row[r]}" for r in WRONG_HIGH_ROWS) + ")")
    for c in picked:
        L.append(f"\n**{c['pick_order']}. {c['row']} {c['case_id']}**: truth {c['truth']}, final {c['final_root']}, "
                 f"H_E001a {c['h_e001a']}, grounding {c['grounding_expl']:.3f}, {c['n_claims_expl']} expl claims")
        for cl in wclaims:
            if cl["row"] == c["row"] and cl["case_id"] == c["case_id"]:
                L.append(f"- [{cl['origin'][:4]}/{cl['claim_type']}] {SHORT.get(cl['verified_label'], cl['verified_label'])} "
                         f"({cl['match']}): {cl['claim']}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main():
    global EXPECTED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secondary", default="results/grounding_heldout/grounding_secondary_cases.csv")
    ap.add_argument("--claims", default="results/grounding_heldout/grounding_claims.csv")
    ap.add_argument("--tests", default="results/grounding_heldout/grounding_tests.csv")
    ap.add_argument("--master", default="results/critic_attribution/master_cases.csv")
    ap.add_argument("--cal-key", default="results/grounding_dev/calibration_key.csv")
    ap.add_argument("--cal-labeled", nargs="+", required=True)
    ap.add_argument("--cal-names", nargs="+", required=True)
    ap.add_argument("--out", default="results/grounding_review")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ack", nargs="*", default=[], help="step-0 checks to accept after a written explanation")
    ap.add_argument("--ack-note", default="", help="the explanation (required with --ack)")
    ap.add_argument("--expected", default=None, help=argparse.SUPPRESS)  # tests only
    a = ap.parse_args()
    if a.ack and not a.ack_note.strip():
        die("--ack needs --ack-note with the explanation")

    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    if len(a.cal_names) != len(a.cal_labeled):
        die("--cal-names needs one name per --cal-labeled file")
    if len(set(a.cal_names)) != len(a.cal_names) or "majority" in a.cal_names:
        die(f"labeler names must be unique and not 'majority': {a.cal_names}")
    if len(a.cal_names) != 3:
        die("A2 majority needs exactly 3 labelers")
    if a.boot < 1000:
        die("--boot must be >= 1000")
    if a.expected:
        EXPECTED = {k: tuple(v) for k, v in json.loads(Path(a.expected).read_text()).items()}
        print(f"NOTE: expected values from {a.expected} (test mode), not the held-out numbers")

    sec = load_secondary(Path(a.secondary))
    claims = load_claims(Path(a.claims))
    master = load_master(Path(a.master))
    tests = Path(a.tests)
    if not tests.exists():
        die(f"missing {tests} (needed as a check)")
    key, human = load_calibration(Path(a.cal_key), [Path(p) for p in a.cal_labeled], a.cal_names)
    print(f"{len(sec)} (row, case) lines, {sum(len(v) for v in claims.values())} claims, "
          f"{len(key)} calibration claims, labelers {', '.join(a.cal_names)}")

    got, notes = step0(sec, claims, master, tests)
    cal_got, agr, conf, cal_claims, fsl = calibration(key, human, a.cal_names)
    got.update(cal_got)
    checks = []
    for k, (exp, src) in EXPECTED.items():
        g = got.get(k, "MISSING")
        checks.append({"check": k, "expected": exp, "got": g, "source": src,
                       "status": "ok" if str(g) == str(exp) else "MISMATCH"})
    mism = [c for c in checks if c["status"] != "ok"]
    stale = [k for k in a.ack if k not in {c["check"] for c in mism}]
    if stale:
        die(f"--ack for checks that did not fail: {stale}; remove them")
    print(f"step 0: {len(checks) - len(mism)}/{len(checks)} checks ok")
    if mism:
        for c in mism:
            tag = "ACK " if c["check"] in a.ack else ""
            print(f"  {tag}MISMATCH {c['check']}: expected {c['expected']}, got {c['got']} ({c['source']})")
        for n in notes:
            print(f"  detail: {n}")
        open_ = [c for c in mism if c["check"] not in a.ack]
        if open_:
            sys.exit(f"STOP: {len(open_)} mismatch(es); nothing written. Explain each before going on "
                     f"(then --ack <check> ... --ack-note \"...\").")
        for c in mism:
            c["status"], c["ack_note"] = "ack", a.ack_note

    print(f"bootstrap: B={a.boot}, seed={a.seed} ...")
    ci = ci_table(sec, a.boot, a.seed)
    lb, hist, lp, mix, lvg = length_tables(sec, claims, a.boot, a.seed)
    cands, picked, wclaims = wrong_high(sec, claims, master)
    summ = summary_md(checks, ci, lb, lp, mix, lvg, agr, fsl, cands, picked, wclaims, a, a.cal_names)

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("checks.csv", checks), ("ci_paired.csv", ci), ("length_by_row.csv", lb),
                       ("length_paired.csv", lp), ("length_hist.csv", hist), ("claim_mix.csv", mix),
                       ("length_vs_grounding.csv", lvg), ("cal_agreement.csv", agr),
                       ("cal_confusion.csv", conf), ("cal_claims.csv", cal_claims),
                       ("cal_false_support.csv", fsl), ("wrong_high_cases.csv", cands),
                       ("wrong_high_claims.csv", wclaims)]:
        write_csv(out / name, rows)
    (out / "summary.md").write_text(summ)
    print("\n" + summ)
    print(f"wrote {out}/ (13 CSVs + summary.md)")


if __name__ == "__main__":
    main()
