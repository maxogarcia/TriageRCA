"""Build the paper's main results table: one line per (model, row) across ablation experiments.

Reads one or more experiment folders written by scripts/run_ablation.py (e.g. the Qwen critic rows,
the Qwen baselines, the Gemini rows), checks they cover the same cases with the same manifest, and
scores every row folder found. Optionally adds the code-only reference row (largest metric shift
at onset, scripts/run_deterministic_baseline.py) on the same cases.

Columns (per group: each system, plus the held-out union, e.g. ss+tt):
  AC@1/3/5, MRR    rank of the true root-cause service in report.ranked_services; a missing JSON or
                   report counts as wrong (the case stays in the denominator)
  fault acc        report.fault_category == manifest fault
  indicator acc    only over cases with a verified indicator label ("n/a" when there are none)
  grounding        mean grounding_expl (primary score, spec v1.1) from scripts/judge_grounding.py's
                   grounding_cases.csv (--grounding), matched to each case by run_path; "pending"
                   when the row was not judged; n = judged cases with a score
  cost/case        mean RunRecord.cost_usd (list price, cached calls included, so rows with cache
                   hits are not cheaper); local models price at $0
  LLM s/case       mean sum of llm_calls[].latency_s (cached calls keep their original latency)
  wall s/case      mean wall_clock_s (lower for rows that hit the shared cache)

Outputs: markdown to stdout and <out>.md, per-line summary <out>.csv, per-case <out>_cases.csv.

Usage:
  python scripts/results_table.py runs/heldout runs/heldout_pipeline_model --deterministic \\
      --grounding results/grounding/grounding_cases.csv --out results/main_table
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.labels import ManifestLabels  # noqa: E402

KS = (1, 3, 5)
DEV_SYSTEMS = {"ob"}
DETERMINISTIC = ("code-only", "largest_onset_shift")


def norm(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_names(models_path: Path = REPO / "configs/models.yaml") -> dict[str, str]:
    import yaml

    models = yaml.safe_load(models_path.read_text())
    return {k: v.get("name", k) for k, v in models.items() if isinstance(v, dict)}


def load_grounding(paths: list[Path]) -> dict[Path, float | None]:
    """run_path -> grounding_expl from judge_grounding.py's grounding_cases.csv (None = judged, no score)."""
    out: dict[Path, float | None] = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                run = Path(r["run_path"])
                run = (run if run.is_absolute() else REPO / run).resolve()
                out[run] = float(r["grounding_expl"]) if r.get("grounding_expl", "") != "" else None
    return out


def score_prediction(case_id: str, labels: ManifestLabels, record: dict | None) -> dict:
    truth = labels.label(case_id)
    report = (record or {}).get("report") or {}
    ranked = [norm(s) for s in (report.get("ranked_services") or [])]
    t = norm(truth.root_cause_service)
    rank = ranked.index(t) + 1 if t in ranked else None
    indicator_known = truth.indicator != "unknown"
    calls = (record or {}).get("llm_calls") or []
    return {
        "case_id": case_id,
        "system": str(labels.row(case_id)["system"]),
        "truth": truth.root_cause_service,
        "pred": ranked[0] if ranked else "",
        "rank": rank or "",
        **{f"ac{k}": int(rank is not None and rank <= k) for k in KS},
        "rr": 1 / rank if rank else 0.0,
        "fault_ok": int(norm(report.get("fault_category")) == norm(truth.fault_category)),
        "indicator_ok": (int(norm(report.get("indicator")) == norm(truth.indicator)) if indicator_known else ""),
        "missing": int(record is None or not ranked),
        "has_errors": int(bool((record or {}).get("errors"))),
        "cost_usd": float((record or {}).get("cost_usd") or 0.0),
        "llm_s": sum(float(c.get("latency_s") or 0) for c in calls),
        "wall_s": float((record or {}).get("wall_clock_s") or 0.0),
        "n_calls": len(calls),
    }


def find_record(row_dir: Path, case_id: str) -> Path | None:
    hits = [p for p in row_dir.glob(f"*/{case_id}.json") if p.parent.name != "llm_cache"]
    if len(hits) > 1:
        sys.exit(f"ERROR: more than one JSON for {case_id} in {row_dir}: {hits}")
    return hits[0] if hits else None


def load_record(path: Path | None, case_id: str) -> dict | None:
    if path is None:
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"WARNING: unreadable {path}: {e}; counted as missing", file=sys.stderr)
        return None
    if record.get("case_id") != case_id:
        sys.exit(f"ERROR: {path} has case_id {record.get('case_id')!r}, expected {case_id!r}")
    return record


def deterministic_records(cases: list[str], labels: ManifestLabels) -> dict[str, dict | None]:
    """Run the largest-onset-shift baseline on the cases, shaped like RunRecords (report only)."""
    from scripts.run_deterministic_baseline import make_prediction  # needs RCAEval + downloaded data

    out: dict[str, dict | None] = {}
    for case_id in cases:
        row = labels.row(case_id).to_dict() | {"case": case_id}
        try:
            pred = make_prediction(row)
        except Exception as e:  # noqa: BLE001 -- a failed case counts as missing, like a crashed run
            print(f"WARNING: deterministic baseline failed on {case_id}: {e}", file=sys.stderr)
            out[case_id] = None
            continue
        ranked = list(dict.fromkeys(pred["ranked_services"]))[:5]
        indicator = pred["ranked_indicators"][0] if pred["ranked_indicators"] else None
        out[case_id] = {"case_id": case_id, "report": {"ranked_services": ranked, "indicator": indicator}}
    return out


def summarize(lines: list[dict], group: str) -> dict:
    n = len(lines)
    ran = [c for c in lines if not c["missing"]]
    ind = [c for c in lines if c["indicator_ok"] != ""]
    grounded = [c for c in lines if c.get("grounding") not in (None, "")]

    def mean(rows, k):
        return sum(c[k] for c in rows) / len(rows) if rows else 0.0

    return {
        "group": group, "n": n,
        **{f"AC@{k}": mean(lines, f"ac{k}") for k in KS},
        "MRR": mean(lines, "rr"),
        "fault_acc": mean(lines, "fault_ok"),
        "indicator_acc": mean(ind, "indicator_ok") if ind else None,
        "indicator_n": len(ind),
        "grounding": mean(grounded, "grounding") if grounded else None,
        "grounding_n": len(grounded),
        "missing": sum(c["missing"] for c in lines),
        "with_errors": sum(c["has_errors"] for c in lines),
        # cost and time over cases that produced a record (a missing case would read as free and instant)
        "cost_per_case_usd": mean(ran, "cost_usd"),
        "llm_s_per_case": mean(ran, "llm_s"),
        "wall_s_per_case": mean(ran, "wall_s"),
        "calls_per_case": mean(ran, "n_calls"),
    }


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def grounding_cell(s: dict) -> str:
    if s["grounding"] is None:
        return "pending"
    return f"{s['grounding']:.2f} (n={s['grounding_n']})"


def markdown(summaries: list[dict], group: str, code_only: bool) -> str:
    head = ("| model | row | AC@1 | AC@3 | AC@5 | MRR | fault acc | indicator acc | grounding "
            "| cost/case | LLM s/case | wall s/case | missing | errors |")
    lines = [f"### {group}{' (dev, diagnostic only)' if group in DEV_SYSTEMS else ''}", "", head, "|" + "---|" * 14]
    for s in summaries:
        if s["group"] != group:
            continue
        is_code = (s["model"], s["row"]) == DETERMINISTIC
        lines.append(
            f"| {s['model']} | {s['row']} | {pct(s['AC@1'])} | {pct(s['AC@3'])} | {pct(s['AC@5'])} "
            f"| {s['MRR']:.3f} | {'n/a' if is_code else pct(s['fault_acc'])} | {pct(s['indicator_acc'])} "
            f"| {'n/a' if is_code else grounding_cell(s)} "
            f"| {'n/a' if is_code else '$' + format(s['cost_per_case_usd'], '.4f')} "
            f"| {'n/a' if is_code else format(s['llm_s_per_case'], '.1f')} "
            f"| {'n/a' if is_code else format(s['wall_s_per_case'], '.1f')} | {s['missing']} | {s['with_errors']} |"
        )
    if code_only:
        lines.append("")
        lines.append("code-only = largest pre/post-onset metric shift (scripts/run_deterministic_baseline.py), no LLM.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", type=Path, help="experiment folders from run_ablation.py")
    ap.add_argument("--deterministic", action="store_true", help="add the code-only largest-onset-shift row")
    ap.add_argument("--grounding", nargs="*", type=Path, default=[],
                    help="grounding_cases.csv file(s) from scripts/judge_grounding.py")
    ap.add_argument("--out", type=Path, default=Path("results/main_table"), help="output path prefix")
    ap.add_argument("--allow-mixed-code", action="store_true",
                    help="allow experiments run at different commits (default: refuse, the sweep must be frozen)")
    args = ap.parse_args()

    exps = []
    for d in args.experiments:
        p = d / "experiment.json"
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        exps.append((d, json.loads(p.read_text())))

    first = exps[0][1]
    for d, e in exps[1:]:
        if e["cases"] != first["cases"]:
            sys.exit(f"ERROR: {d} covers different cases than {exps[0][0]}; rows would not be comparable")
        if e["manifest_sha256"] != first["manifest_sha256"]:
            sys.exit(f"ERROR: {d} used a different manifest than {exps[0][0]}")
    versions = {e["code_version"] for _, e in exps}
    if len(versions) > 1:
        msg = "experiments ran at different code versions: " + ", ".join(sorted(v[:12] for v in versions))
        if not args.allow_mixed_code:
            sys.exit(f"ERROR: {msg} (pass --allow-mixed-code for a non-final table)")
        print(f"WARNING: {msg}", file=sys.stderr)

    manifest = Path(first["manifest"])
    manifest = manifest if manifest.is_absolute() else REPO / manifest
    if sha256(manifest) != first["manifest_sha256"]:
        sys.exit(f"ERROR: {manifest} sha256 differs from experiment.json; scoring would use other labels")
    labels = ManifestLabels(manifest)
    cases = list(first["cases"])
    names = model_names()
    grounding = load_grounding(args.grounding)

    per_case: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for d, e in exps:
        for row, cfg in e["rows"].items():
            if not (d / row).is_dir():
                continue  # experiment.json lists every row, including ones never run into this folder
            model = names.get(cfg["model"], cfg["model"])
            if (model, row) in seen:
                sys.exit(f"ERROR: ({model}, {row}) appears in more than one experiment folder")
            seen.add((model, row))
            for c in cases:
                path = find_record(d / row, c)
                line = score_prediction(c, labels, load_record(path, c))
                line["grounding"] = grounding.get(path.resolve(), "") if path else ""
                per_case.append({"model": model, "row": row} | line)

    if args.deterministic:
        for c, rec in deterministic_records(cases, labels).items():
            line = score_prediction(c, labels, rec) | {"grounding": ""}
            per_case.append({"model": DETERMINISTIC[0], "row": DETERMINISTIC[1]} | line)

    if not per_case:
        sys.exit("ERROR: no row folders found in the given experiments")

    systems = sorted({c["system"] for c in per_case}, key=lambda s: (s not in DEV_SYSTEMS, s))
    groups = list(systems)
    held = [s for s in systems if s not in DEV_SYSTEMS]
    if len(held) > 1:
        groups.append("+".join(held))

    keys = list(dict.fromkeys((c["model"], c["row"]) for c in per_case))
    summaries = []
    for g in groups:
        members = set(g.split("+"))
        for model, row in keys:
            subset = [c for c in per_case if c["model"] == model and c["row"] == row and c["system"] in members]
            summaries.append({"model": model, "row": row} | summarize(subset, g))

    # Headline group first: the held-out union if there is one, else the last system.
    order = groups[-1:] + groups[:-1]
    md = "\n\n".join(markdown(summaries, g, args.deterministic) for g in order)
    header = (f"cases: {len(cases)}  code: {', '.join(sorted(v[:12] for v in versions))}  "
              f"manifest sha256: {first['manifest_sha256'][:12]}\n\n")
    print(header + md)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(header + md + "\n")
    for path, rows in ((args.out.with_suffix(".csv"), summaries),
                       (args.out.with_name(args.out.name + "_cases.csv"), per_case)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: round(v, 4) if isinstance(v, float) else ("" if v is None else v)
                            for k, v in r.items()})
    print(f"\nwrote {args.out.with_suffix('.md')}, {args.out.with_suffix('.csv')}, "
          f"{args.out.with_name(args.out.name + '_cases.csv')}")


if __name__ == "__main__":
    main()
