"""Score an ablation experiment written by scripts/run_ablation.py.

Reads <out>/experiment.json (case list, manifest path + sha256, row names),
then every case JSON at <out>/<row>/<system dir>/<case_id>.json.

Outputs:
  - markdown table per system (dev = ob, held-out = ss, tt, plus ss+tt) to stdout
  - <out>/scores_cases.csv    one line per (row, case)
  - <out>/scores_summary.csv  one line per (row, group)

Scoring rules:
  - prediction = report.ranked_services[0]; truth = manifest root_cause_service
  - missing JSON or missing report counts as wrong (case stays in the denominator)
  - cases with nonempty `errors` are kept (e.g. 3 rejects still leave a report)
  - wrong accepted  = last verdict is accept AND final answer wrong
  - correct rejected = case where some round's top service was correct and that
    round's verdict was not accept (reject or need_more_evidence)
  - LLM time = sum of latency_s over all calls (cached calls keep their original
    latency, so this is fair across rows); critic time = same for role "critic";
    wall clock is secondary (lower for rows with cache hits)

Usage:
  python scripts/score_ablation.py --out runs/heldout
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RULE_SKIP = "code onset rule skipped:"
PARSE_FAIL = "critic output could not be parsed"
UNCHECKED = "accepted unchecked"
DEV_SYSTEMS = {"ob"}

CASE_FIELDS = [
    "row", "case_id", "system", "truth", "pred", "correct", "missing_json", "missing_report",
    "last_decision", "wrong_accepted", "correct_rejected", "rounds", "n_calls", "n_cached",
    "llm_time_s", "critic_time_s", "wall_clock_s", "rule_skips", "parse_fails",
    "parse_fails_unchecked", "has_errors",
]


def norm(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reasons_contain(verdict: dict, text: str) -> bool:
    return any(text in str(r) for r in (verdict.get("reasons") or []))


def find_case_json(row_dir: Path, case_id: str) -> Path | None:
    hits = [p for p in row_dir.glob(f"*/{case_id}.json") if p.parent.name != "llm_cache"]
    if len(hits) > 1:
        sys.exit(f"ERROR: more than one JSON for {case_id} in {row_dir}: {hits}")
    return hits[0] if hits else None


def score_case(row: str, case_id: str, label: dict, path: Path | None) -> dict:
    truth = label["root_cause_service"]
    out = {
        "row": row, "case_id": case_id, "system": label["system"], "truth": truth,
        "pred": "", "correct": False, "missing_json": path is None, "missing_report": True,
        "last_decision": "", "wrong_accepted": False, "correct_rejected": False,
        "rounds": 0, "n_calls": 0, "n_cached": 0, "llm_time_s": 0.0, "critic_time_s": 0.0,
        "wall_clock_s": 0.0, "rule_skips": 0, "parse_fails": 0, "parse_fails_unchecked": 0,
        "has_errors": False,
    }
    if path is None:
        return out
    try:
        r = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: unreadable {path}: {e}; counted as wrong", file=sys.stderr)
        out["missing_json"] = True
        return out
    if r.get("case_id") != case_id:
        sys.exit(f"ERROR: {path} has case_id {r.get('case_id')!r}, expected {case_id!r}")

    report = r.get("report") or {}
    ranked = report.get("ranked_services") or []
    out["missing_report"] = not ranked
    out["pred"] = ranked[0] if ranked else ""
    out["correct"] = bool(ranked) and norm(ranked[0]) == norm(truth)

    verdicts = r.get("critic_verdicts") or []
    hyps = r.get("hypotheses") or []
    if verdicts:
        out["last_decision"] = verdicts[-1].get("decision", "")
    out["wrong_accepted"] = out["last_decision"] == "accept" and not out["correct"]
    for i, v in enumerate(verdicts):
        top = (hyps[i].get("ranked_services") or [None])[0] if i < len(hyps) else None
        if top is not None and norm(top) == norm(truth) and v.get("decision") != "accept":
            out["correct_rejected"] = True
        out["rule_skips"] += reasons_contain(v, RULE_SKIP)
        if reasons_contain(v, PARSE_FAIL):
            out["parse_fails"] += 1
            out["parse_fails_unchecked"] += reasons_contain(v, UNCHECKED)

    calls = r.get("llm_calls") or []
    out["rounds"] = r.get("iteration_count") or 0
    out["n_calls"] = len(calls)
    out["n_cached"] = sum(1 for c in calls if c.get("cached"))
    out["llm_time_s"] = sum(float(c.get("latency_s") or 0) for c in calls)
    out["critic_time_s"] = sum(float(c.get("latency_s") or 0) for c in calls if c.get("role") == "critic")
    out["wall_clock_s"] = float(r.get("wall_clock_s") or 0)
    out["has_errors"] = bool(r.get("errors"))
    return out


def summarize(row: str, group: str, cases: list[dict]) -> dict:
    n = len(cases)

    def total(k):
        return sum(c[k] for c in cases)

    ran = [c for c in cases if not c["missing_json"]]

    def mean(k):  # rounds and timing: only cases that produced a JSON (missing = 0 would dilute)
        return sum(c[k] for c in ran) / len(ran) if ran else 0.0

    return {
        "row": row, "group": group, "n": n,
        "correct": total("correct"), "accuracy": total("correct") / n if n else 0.0,
        "wrong_accepted": total("wrong_accepted"), "correct_rejected": total("correct_rejected"),
        "missing": sum(1 for c in cases if c["missing_json"] or c["missing_report"]),
        "with_errors": total("has_errors"),
        "mean_rounds": mean("rounds"), "mean_llm_time_s": mean("llm_time_s"),
        "mean_critic_time_s": mean("critic_time_s"), "mean_wall_clock_s": mean("wall_clock_s"),
        "cached_calls": total("n_cached"), "total_calls": total("n_calls"),
        "rule_skips": total("rule_skips"), "parse_fails": total("parse_fails"),
        "parse_fails_unchecked": total("parse_fails_unchecked"),
    }


def group_name(g: str) -> str:
    return f"{g} (dev)" if g in DEV_SYSTEMS else g


def markdown(summaries: list[dict], group: str) -> str:
    head = ("| row | correct | wrong accepted | correct rejected | missing | errors | rounds "
            "| LLM s | critic s | wall s | cached/calls | rule skips | parse fails (unchecked) |")
    lines = [f"### {group_name(group)}", "", head, "|" + "---|" * 13]
    for s in summaries:
        if s["group"] != group:
            continue
        lines.append(
            f"| {s['row']} | {s['correct']}/{s['n']} ({s['accuracy']:.0%}) | {s['wrong_accepted']} "
            f"| {s['correct_rejected']} | {s['missing']} | {s['with_errors']} | {s['mean_rounds']:.2f} "
            f"| {s['mean_llm_time_s']:.1f} | {s['mean_critic_time_s']:.1f} | {s['mean_wall_clock_s']:.1f} "
            f"| {s['cached_calls']}/{s['total_calls']} | {s['rule_skips']} "
            f"| {s['parse_fails']} ({s['parse_fails_unchecked']}) |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/heldout", help="experiment folder from run_ablation.py")
    ap.add_argument("--manifest", default=None,
                    help="override the manifest path in experiment.json (sha256 must still match)")
    ap.add_argument("--rows", nargs="*", default=None, help="subset of rows (default: all in experiment.json)")
    args = ap.parse_args()

    out = Path(args.out)
    exp_path = out / "experiment.json"
    if not exp_path.exists():
        sys.exit(f"ERROR: {exp_path} not found")
    exp = json.loads(exp_path.read_text())

    manifest = Path(args.manifest or exp["manifest"])
    if not manifest.exists():
        sys.exit(f"ERROR: manifest {manifest} not found (use --manifest)")
    if sha256(manifest) != exp["manifest_sha256"]:
        sys.exit(f"ERROR: {manifest} sha256 differs from experiment.json; scoring would use other labels")
    with open(manifest, newline="") as f:
        labels = {r["case"]: r for r in csv.DictReader(f)}

    cases = list(exp["cases"])
    unknown = [c for c in cases if c not in labels]
    if unknown:
        sys.exit(f"ERROR: cases not in manifest: {unknown[:5]}")

    rows = list(exp["rows"].keys())
    if args.rows:
        bad = [r for r in args.rows if r not in rows]
        if bad:
            sys.exit(f"ERROR: unknown rows {bad}; experiment has {rows}")
        rows = [r for r in rows if r in args.rows]

    case_rows = []
    for row in rows:
        for case_id in cases:
            case_rows.append(score_case(row, case_id, labels[case_id], find_case_json(out / row, case_id)))

    systems = sorted({c["system"] for c in case_rows}, key=lambda s: (s not in DEV_SYSTEMS, s))
    groups = list(systems)
    held = [s for s in systems if s not in DEV_SYSTEMS]
    if len(held) > 1:
        groups.append("+".join(held))

    summaries = []
    for g in groups:
        members = set(g.split("+"))
        for row in rows:
            subset = [c for c in case_rows if c["row"] == row and c["system"] in members]
            summaries.append(summarize(row, g, subset))

    print(f"experiment: {out}  code {exp.get('code_version', '?')[:12]}  cases {len(cases)}  rows {len(rows)}\n")
    for g in groups:
        print(markdown(summaries, g) + "\n")

    write_csv(out / "scores_cases.csv", case_rows, CASE_FIELDS)
    write_csv(out / "scores_summary.csv", summaries, list(summaries[0].keys()))
    print(f"wrote {out / 'scores_cases.csv'} and {out / 'scores_summary.csv'}")


if __name__ == "__main__":
    main()
