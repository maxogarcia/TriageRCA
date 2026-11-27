#!/usr/bin/env python3
"""Diagnosis (post hoc, descriptive, Addendum A15). Answers the professor's Nov 23 questions 1-3.

Read-only. No LLM calls. No src/ imports. Writes CSVs + summary.md to a NEW --out.

  (a) A15.1  single-spike vs multi-service cases (from E001), AC@1 per group, where hybrid's
             corrections / regressions vs the onset screen fall
  (b) A15.2  what the round-1 Executor named in the cases where it did not pick E001's top service
  (c) A15.3  hybrid's false rejects (push back on a correct root), per decision, by category

Inputs (repo-relative defaults):
  --master  results/critic_attribution/master_cases.csv   (A13 source of truth)
  --runs    runs/heldout runs/heldout_baselines            (<row>/<config>/<case_id>.json)

Step 0: known counts and consistency checks (A15). Any mismatch stops with nothing written.

Example (from repo root):
  python scripts/diagnosis.py --out results/diagnosis
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

ROWS_SHOWN = ["H_E001a", "hybrid", "no_critic", "llm_only", "react"]
NEEDED_ROWS = ["hybrid", "no_critic", "llm_only", "react"]
PUSH_BACK = {"reject", "need_more_evidence"}
RATIO = 2.0                                   # A15.1
MIN_READ = 10                                 # A15 reading rule
GATEWAYS = {"SS": ["front-end"], "TT": ["ts-ui-dashboard", "ts-gateway-service"]}   # A15.2
HINT_RE = re.compile(r"\binvestigate\s+([A-Za-z0-9][A-Za-z0-9\-]*)\s*:", re.I)    # A13.1
TOP_HEADER = "Top 12 services"
LINE_RE = re.compile(r"^\s*-\s*(?P<svc>[A-Za-z0-9\-]+):\s*(?P<col>[^,]+),.*?shift=(?P<shift>[-+\d.eE]+)")
TAGS = [  # A15.3, fixed
    ("onset", re.compile(r"onset", re.I)),
    ("indicator", re.compile(r"indicator", re.I)),
    ("fault_type", re.compile(r"fault\s+type|fault\s+category|category", re.I)),
    ("contradicted", re.compile(r"contradict", re.I)),
    ("unsupported", re.compile(r"unsupported|not\s+supported|partial", re.I)),
]
CATEGORIES = ["screen (= H_E001a)", "gateway", "E001 rank 2-3", "E001 rank 4-12", "not in E001 top 12", "no answer"]
EXPECTED = {"ac1/hybrid": 48, "ac1/no_critic": 15, "ac1/H_E001a": 48, "r1_eq_screen": 15,
            "hybrid_decisions": 140, "false_rejects": 28, "pushed_back_final_wrong": 4,
            "corrections": 6, "regressions": 6, "e001_top_ties": 6}


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def norm(s) -> str:
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


def tf(s: str) -> bool:
    if s not in ("True", "False"):
        die(f"master_cases.csv: expected True/False, got {s!r}")
    return s == "True"


def strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings_in(v)


def hint_of(verdict) -> str:
    for s in strings_in(verdict):
        m = HINT_RE.search(s)
        if m:
            return m.group(1)
    return ""


def root_of(h) -> str:
    rs = (h.get("ranked_services") or []) if isinstance(h, dict) else []
    return str(rs[0]) if rs else ""


def parse_e001(content: str) -> list[tuple[str, float]]:
    lines = content.splitlines()
    i = next((k for k, l in enumerate(lines) if l.strip().startswith(TOP_HEADER)), None)
    if i is None:
        raise ValueError("E001 has no 'Top 12 services' line")
    out = []
    for l in lines[i + 1:]:
        m = LINE_RE.match(l)
        if not m:
            break
        out.append((m["svc"], float(m["shift"])))
    return out


def split_group(listed) -> tuple[str, float | str, bool]:
    """A15.1 -> (group, ratio s1/s2, tie at top)."""
    mags = sorted((abs(s) for _, s in listed), reverse=True)
    if len(mags) < 2:
        return "single", "", False
    s1, s2 = mags[0], mags[1]
    tie = abs(float(listed[0][1])) == abs(float(listed[1][1])) if len(listed) > 1 else False
    if s2 == 0:
        return "single", "inf", tie
    ratio = s1 / s2
    if s1 == s2:
        return "multi", 1.0, tie
    return ("single" if ratio >= RATIO else "multi"), round(ratio, 4), tie


def find_json(run_dirs, row, case_id) -> Path:
    hits = []
    for d in run_dirs:
        hits += [p for p in (d / row).glob(f"*/{case_id}.json") if p.parent.name != "llm_cache"]
    if len(hits) != 1:
        die(f"expected 1 JSON for {row}/{case_id}, found {len(hits)}")
    return hits[0]


def r1_category(r1, ranked, system, screen) -> str:
    if not r1:
        return "no answer"
    if norm(r1) == norm(screen):
        return "screen (= H_E001a)"
    if norm(r1) in {norm(g) for g in GATEWAYS.get(system.upper(), [])}:
        return "gateway"
    names = [norm(s) for s, _ in ranked]
    if norm(r1) in names:
        pos = names.index(norm(r1)) + 1
        return "E001 rank 2-3" if pos <= 3 else "E001 rank 4-12"
    return "not in E001 top 12"


def pct(n, d) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else f"{n}/0"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="results/critic_attribution/master_cases.csv")
    ap.add_argument("--runs", nargs="+", default=["runs/heldout", "runs/heldout_baselines"])
    ap.add_argument("--out", default="results/diagnosis")
    ap.add_argument("--expected", default=None, help=argparse.SUPPRESS)   # tests only
    a = ap.parse_args()

    global EXPECTED
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f"{out} exists and is not empty; pick a new --out")
    if a.expected:
        EXPECTED = json.loads(Path(a.expected).read_text())
        print(f"NOTE: expected values from {a.expected} (test mode), not the held-out numbers")
    run_dirs = [Path(p) for p in a.runs]

    master = {}
    for r in read_csv(Path(a.master)):
        k = (r["row"], r["case_id"])
        if k in master:
            die(f"duplicate {k} in {a.master}")
        master[k] = r
    need_cols = ["row", "case_id", "system", "truth", "final_root", "correct", "r1_root",
                 "e001_top_a", "e001_top_tie", "h_e001a_correct", "correct_pushed_back"]
    miss = [c for c in need_cols if c not in next(iter(master.values()))]
    if miss:
        die(f"{a.master} is missing columns {miss}")
    cases = sorted(c for (r, c) in master if r == "hybrid")
    for row in NEEDED_ROWS:
        have = sorted(c for (r, c) in master if r == row)
        if have != cases:
            die(f"row {row} has a different case set from hybrid in {a.master}")
    M = lambda row, c: master[(row, c)]  # noqa: E731

    # ---------- load JSONs
    info, bad = {}, defaultdict(list)
    for c in cases:
        hm = M("hybrid", c)
        rec = json.loads(find_json(run_dirs, "hybrid", c).read_text())
        if rec.get("case_id") != c:
            die(f"hybrid/{c}: JSON case_id {rec.get('case_id')!r}")
        led = rec.get("evidence_ledger") or []
        if not led or led[0].get("id") != "E001" or led[0].get("source") != "metric_overview":
            die(f"hybrid/{c}: first ledger item is not E001 metric_overview")
        try:
            listed = parse_e001(led[0].get("content", ""))
        except ValueError as e:
            die(f"{c}: {e}")
        if not listed:
            die(f"{c}: no service lines under '{TOP_HEADER}'")
        group, ratio, tie = split_group(listed)
        screen = listed[0][0]
        if norm(screen) != norm(hm["e001_top_a"]):
            bad["e001_top_a"].append(f"{c}: json {screen!r} vs master {hm['e001_top_a']!r}")
        if tie != tf(hm["e001_top_tie"]):
            bad["e001_top_tie"].append(f"{c}: json {tie} vs master {hm['e001_top_tie']}")
        hyps = rec.get("hypotheses") or []
        verds = rec.get("critic_verdicts") or []
        if len(hyps) != len(verds):
            die(f"hybrid/{c}: {len(hyps)} hypotheses vs {len(verds)} verdicts")
        r1 = M("no_critic", c)["final_root"]
        if norm(root_of(hyps[0]) if hyps else "") != norm(r1):
            bad["r1_identity"].append(f"{c}: hybrid r1 {root_of(hyps[0]) if hyps else ''!r} vs no_critic {r1!r}")
        if norm(hm["r1_root"]) != norm(r1):
            bad["r1_master"].append(f"{c}: master hybrid r1_root {hm['r1_root']!r} vs no_critic {r1!r}")
        info[c] = {"system": hm["system"].upper(), "truth": hm["truth"], "listed": listed, "group": group,
                   "ratio": ratio, "tie": tie, "screen": screen, "hyps": hyps, "verds": verds, "r1": r1}
    if bad:
        for k, ex in bad.items():
            print(f"  MISMATCH {k}: {len(ex)} cases, e.g. {ex[:3]}")
        die("E001 / round-1 consistency check failed; nothing written")

    # ---------- per case predictions
    def correct(name, c):
        if name == "H_E001a":
            return norm(info[c]["screen"]) == norm(info[c]["truth"])
        return tf(M(name, c)["correct"])

    case_rows, fr_rows = [], []
    for c in cases:
        x = info[c]
        h_ok, s_ok = correct("hybrid", c), correct("H_E001a", c)
        change = "correction" if h_ok and not s_ok else ("regression" if s_ok and not h_ok else "")
        cat = r1_category(x["r1"], x["listed"], x["system"], x["screen"])
        pos = next((i + 1 for i, (s, _) in enumerate(x["listed"]) if norm(s) == norm(x["r1"])), "")
        # false rejects (A15.3)
        n_fr = 0
        for i, (h, v) in enumerate(zip(x["hyps"], x["verds"])):
            dec = str(v.get("decision", ""))
            root = root_of(h)
            if dec in PUSH_BACK and norm(root) == norm(x["truth"]):
                n_fr += 1
                hint = hint_of(v)
                text = " | ".join(strings_in(v))
                tags = [t for t, rx in TAGS if rx.search(text)]
                me = v.get("missing_evidence") or []
                me = me if isinstance(me, list) else [me]
                if any(not HINT_RE.search(str(e)) for e in me if str(e).strip()):
                    tags.append("missing_evidence_listed")
                fr_rows.append({
                    "case_id": c, "system": x["system"], "group": x["group"], "round": i + 1,
                    "decision": dec, "claimed_root": root, "truth": x["truth"],
                    "onset_hint": hint, "hint_eq_h_e001a": (norm(hint) == norm(x["screen"])) if hint else "",
                    "primary": f"{dec} + {'hint' if hint else 'no hint'}",
                    "tags": " ".join(tags) if tags else "none",
                    "case_final_correct": h_ok, "outcome": "recovered" if h_ok else "ended wrong",
                    "final_root": M("hybrid", c)["final_root"], "rounds": len(x["verds"]),
                    "reason_text": text[:1500]})
        case_rows.append({
            "case_id": c, "system": x["system"], "truth": x["truth"], "group": x["group"],
            "ratio_s1_s2": x["ratio"], "e001_top_tie": x["tie"], "screen": x["screen"],
            "screen_shift": x["listed"][0][1], "second": x["listed"][1][0] if len(x["listed"]) > 1 else "",
            "second_shift": x["listed"][1][1] if len(x["listed"]) > 1 else "",
            "n_listed": len(x["listed"]),
            **{f"{n}_correct": correct(n, c) for n in ROWS_SHOWN},
            "hybrid_final": M("hybrid", c)["final_root"], "hybrid_vs_screen": change,
            "r1_root": x["r1"], "r1_e001_rank": pos, "r1_category": cat,
            "r1_correct": norm(x["r1"]) == norm(x["truth"]),
            "hybrid_false_rejects": n_fr,
            "correct_pushed_back_final_wrong": tf(M("hybrid", c)["correct_pushed_back"]) and not h_ok})

    # ---------- Step 0
    got = {
        "ac1/hybrid": sum(r["hybrid_correct"] for r in case_rows),
        "ac1/no_critic": sum(r["no_critic_correct"] for r in case_rows),
        "ac1/H_E001a": sum(r["H_E001a_correct"] for r in case_rows),
        "r1_eq_screen": sum(r["r1_category"] == CATEGORIES[0] for r in case_rows),
        "hybrid_decisions": sum(len(info[c]["verds"]) for c in cases),
        "false_rejects": len(fr_rows),
        "pushed_back_final_wrong": sum(r["correct_pushed_back_final_wrong"] for r in case_rows),
        "corrections": sum(r["hybrid_vs_screen"] == "correction" for r in case_rows),
        "regressions": sum(r["hybrid_vs_screen"] == "regression" for r in case_rows),
        "e001_top_ties": sum(r["e001_top_tie"] for r in case_rows),
    }
    fr_wrong_cases = {r["case_id"] for r in fr_rows if r["outcome"] == "ended wrong"}
    pbw_cases = {r["case_id"] for r in case_rows if r["correct_pushed_back_final_wrong"]}
    checks = [{"check": k, "expected": EXPECTED.get(k), "got": v,
               "status": "ok" if EXPECTED.get(k) == v else "MISMATCH"} for k, v in got.items()]
    checks.append({"check": "fr_ended_wrong_cases == pushed_back_final_wrong cases", "expected": sorted(pbw_cases),
                   "got": sorted(fr_wrong_cases), "status": "ok" if fr_wrong_cases == pbw_cases else "MISMATCH"})
    n_bad = sum(r["status"] != "ok" for r in checks)
    print(f"Step 0: {len(checks) - n_bad}/{len(checks)} checks ok")
    if n_bad:
        for r in checks:
            if r["status"] != "ok":
                print(f"  MISMATCH {r['check']}: expected {r['expected']}, got {r['got']}")
        die("Step 0 failed; nothing written")

    # ---------- (a) groups
    groups = ["single", "multi"]
    g_rows = []
    for g in ["all"] + groups:
        for s in ["all", "SS", "TT"]:
            cs = [r for r in case_rows if (g == "all" or r["group"] == g) and (s == "all" or r["system"] == s)]
            line = {"group": g, "system": s, "n": len(cs)}
            for n in ROWS_SHOWN:
                line[f"{n}_correct"] = sum(r[f"{n}_correct"] for r in cs)
            line["r1_eq_screen"] = sum(r["r1_category"] == CATEGORIES[0] for r in cs)
            line["corrections"] = sum(r["hybrid_vs_screen"] == "correction" for r in cs)
            line["regressions"] = sum(r["hybrid_vs_screen"] == "regression" for r in cs)
            line["ties"] = sum(r["e001_top_tie"] for r in cs)
            rs = [float(r["ratio_s1_s2"]) for r in cs if r["ratio_s1_s2"] not in ("", "inf")]
            line["ratio_median"] = round(statistics.median(rs), 3) if rs else ""
            line["ratio_min"] = round(min(rs), 3) if rs else ""
            line["ratio_max"] = round(max(rs), 3) if rs else ""
            line["interpret"] = len(cs) >= MIN_READ
            g_rows.append(line)

    # ---------- (b) round-1 categories
    cat_rows = []
    for split in ["all", "SS", "TT", "single", "multi"]:
        for cat in CATEGORIES:
            cs = [r for r in case_rows if r["r1_category"] == cat and
                  (split == "all" or r["system"] == split or r["group"] == split)]
            cat_rows.append({"split": split, "category": cat, "n": len(cs),
                             "r1_correct": sum(r["r1_correct"] for r in cs),
                             "hybrid_final_correct": sum(r["hybrid_correct"] for r in cs),
                             "screen_correct": sum(r["H_E001a_correct"] for r in cs),
                             "cases": " ".join(r["case_id"] for r in cs)})
    gw_seen = Counter()
    for c in cases:
        names = {norm(s) for s, _ in info[c]["listed"]} | {norm(info[c]["r1"])}
        for sysname, gws in GATEWAYS.items():
            for gname in gws:
                if norm(gname) in names:
                    gw_seen[gname] += 1

    # ---------- (c) false rejects
    fr_counts = []
    for key in ["primary", "tags"]:
        cnt = Counter()
        for r in fr_rows:
            vals = [r[key]] if key == "primary" else r["tags"].split()
            for v in vals:
                cnt[(v, r["outcome"])] += 1
        for v in sorted({k for k, _ in cnt}):
            fr_counts.append({"kind": key, "value": v, "recovered": cnt[(v, "recovered")],
                              "ended_wrong": cnt[(v, "ended wrong")],
                              "total": cnt[(v, "recovered")] + cnt[(v, "ended wrong")]})
    fr_cases = Counter(r["case_id"] for r in fr_rows)

    # ---------- summary
    L = ["# Diagnosis (post hoc, descriptive, A15): summary", "",
         f"Cases: {len(cases)} (SS+TT). Generated by scripts/diagnosis.py. Step 0: {len(checks)}/{len(checks)} checks ok.",
         f"Reading rule: groups / categories with fewer than {MIN_READ} cases are reported, not interpreted.", "",
         f"## (a) Single-spike vs multi-service (A15.1: single if top shift >= {RATIO:g}x second; ties = multi)", "",
         "| group | system | n | screen | hybrid | no_critic | llm_only | react | r1 = screen | corrections | regressions | ties | s1/s2 median [min, max] |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in g_rows:
        L.append(f"| {r['group']} | {r['system']} | {r['n']} | {r['H_E001a_correct']} | {r['hybrid_correct']} | "
                 f"{r['no_critic_correct']} | {r['llm_only_correct']} | {r['react_correct']} | {r['r1_eq_screen']} | "
                 f"{r['corrections']} | {r['regressions']} | {r['ties']} | {r['ratio_median']} [{r['ratio_min']}, {r['ratio_max']}] |")
    L.append("\nCorrections / regressions (hybrid vs screen):")
    for r in case_rows:
        if r["hybrid_vs_screen"]:
            L.append(f"- {r['hybrid_vs_screen']}: {r['case_id']} ({r['group']}, s1/s2 {r['ratio_s1_s2']}, "
                     f"screen {r['screen']}, hybrid {r['hybrid_final']}, truth {r['truth']})")
    L += ["", "## (b) Round-1 pick categories (A15.2)", "",
          "| category | n | r1 correct | hybrid final correct | screen correct | SS | TT | single | multi |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by = {(r["split"], r["category"]): r for r in cat_rows}
    for cat in CATEGORIES:
        a_ = by[("all", cat)]
        L.append(f"| {cat} | {a_['n']} | {a_['r1_correct']} | {a_['hybrid_final_correct']} | {a_['screen_correct']} | "
                 + " | ".join(str(by[(s, cat)]["n"]) for s in ["SS", "TT", "single", "multi"]) + " |")
    L.append(f"\nGateway names seen in E001 lists or round-1 roots (cases): "
             + ", ".join(f"{g}={gw_seen[g]}" for gws in GATEWAYS.values() for g in gws))
    L += ["", f"## (c) Hybrid false rejects (A15.3): {len(fr_rows)} of {got['hybrid_decisions']} decisions, "
              f"in {len(fr_cases)} cases; {len(fr_wrong_cases)} cases ended wrong", "",
          "| kind | value | recovered | ended wrong | total |", "|---|---|---:|---:|---:|"]
    for r in fr_counts:
        L.append(f"| {r['kind']} | {r['value']} | {r['recovered']} | {r['ended_wrong']} | {r['total']} |")
    hinted = [r for r in fr_rows if r["onset_hint"]]
    L.append(f"\nHinted false rejects: {len(hinted)}; hint = screen in {sum(r['hint_eq_h_e001a'] is True for r in hinted)}.")
    L.append(f"False rejects per case: {dict(Counter(fr_cases.values()))} (number of false rejects: cases)")
    summ = "\n".join(L) + "\n"

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "checks.csv", checks)
    write_csv(out / "cases.csv", case_rows)
    write_csv(out / "groups.csv", g_rows)
    write_csv(out / "r1_categories.csv", cat_rows)
    write_csv(out / "false_rejects.csv", fr_rows)
    write_csv(out / "false_reject_counts.csv", fr_counts)
    (out / "summary.md").write_text(summ)
    print("\n" + summ)
    print(f"wrote {out}/ (6 CSVs + summary.md)")


if __name__ == "__main__":
    main()
