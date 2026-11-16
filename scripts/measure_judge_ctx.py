#!/usr/bin/env python3
"""Measure the grounding judge's Step B prompt size for every run, with no LLM calls (too_long check).

Read-only. For each run JSON it rebuilds exactly what judge_grounding.py feeds Step B
(LABEL_SYSTEM + "EVIDENCE:" + incident query + full ledger + longest claim) and applies the judge's
own pre-check: est = chars / CHARS_PER_TOKEN, too_long if est + NUM_PREDICT > num_ctx.
It reads only the sizes of ledgers and explanations. It never reads ranks, ground truth, critic
verdicts or scores, so running it on held-out folders is not held-out tuning.

The longest claim is only known after the LLM decompose step, so each run gets a range:
  est_lo: longest claim = longest fixed structured claim (decompose claims are usually short)
  est_hi: longest claim = the whole explanation (strict upper bound; no claim is longer)
The judge's real est is always between the two. The ledger dominates, so the range is narrow.

Optional dev-only check (--calibrate DIR, a judged results folder with grounding_cases.csv and
judge_cache/):
  1. bounds check: est_lo <= judge's recorded prompt_tokens_B <= est_hi for every judged case
     (proves this script rebuilds the judge's prompt correctly);
  2. real tokens: rebuilds each Step B request body, finds its cache file, and reads Ollama's
     prompt_eval_count, giving actual chars/token for gemma3 vs the judge's 3.2 assumption.

Examples:
  python scripts/measure_judge_ctx.py --out runs/heldout runs/heldout_baselines
  python scripts/measure_judge_ctx.py --out runs/heldout runs/heldout_baselines \\
      --csv results/judge_ctx/heldout_sizes.csv
  python scripts/measure_judge_ctx.py --calibrate results/grounding_dev
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_grounding import (  # noqa: E402
    CHARS_PER_TOKEN, CITE_RE, DEFAULT_MANIFEST, LABEL_SCHEMA, LABEL_SYSTEM, NUM_PREDICT,
    discover, format_evidence, incident_query, load_manifest, repo_path, structured_claims,
)

DEFAULT_CTX = [16384, 20480, 24576, 32768]
CPT = CHARS_PER_TOKEN  # --cpt overrides for the size report only


def prefix_for(run: dict, m: dict) -> str:
    """Same string the judge builds in judge_case Step B."""
    evidence = format_evidence(incident_query(m), run.get("evidence_ledger") or [])
    return f"EVIDENCE:\n\n{evidence}\n\nCLAIM TO CHECK:\n"


def measure(run: dict, m: dict) -> dict:
    report = run.get("report") or {}
    fixed = structured_claims(report)
    explanation = CITE_RE.sub("", report.get("explanation") or "").strip()
    base = len(LABEL_SYSTEM) + len(prefix_for(run, m))
    lo_claim = max((len(t) for t in fixed), default=0)
    hi_claim = max(lo_claim, len(explanation))
    ledger = run.get("evidence_ledger") or []
    return {
        "n_evidence": len(ledger),
        "ledger_chars": sum(len(str(i.get("content") or "")) for i in ledger),
        "expl_chars": len(explanation),
        "est_lo": int((base + lo_claim) / CPT),
        "est_hi": int((base + hi_claim) / CPT),
    }


def pct(xs: list[float], q: int) -> int | str:
    """q-th percentile (nearest-rank), '' if empty."""
    if not xs:
        return ""
    s = sorted(xs)
    k = max(0, min(len(s) - 1, -(-q * len(s) // 100) - 1))
    return int(s[k])


def load_runs(out_dirs: list[str], rows: set[str] | None):
    for out in out_dirs:
        for row, p in discover([out], rows):
            try:
                run = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                yield Path(out).name, row, p, None
                continue
            yield Path(out).name, row, p, run


# ---------------------------------------------------------------- held-out size report


def size_report(args, manifest):
    ctxs = sorted(set(args.ctx))
    recs, skipped = [], defaultdict(int)
    for out, row, p, run in load_runs(args.out, set(args.rows) if args.rows else None):
        if not isinstance(run, dict) or "report" not in run or "evidence_ledger" not in run:
            skipped["unreadable_or_not_a_run"] += 1
            continue
        if not run.get("report"):
            skipped["no_report (judge: status no_report, not a size issue)"] += 1
            continue
        cid = run.get("case_id", p.stem)
        if cid not in manifest:
            sys.exit(f"{cid} not in manifest {args.manifest}")
        m = manifest[cid]
        recs.append({"out": out, "row": row, "system": m.get("system", ""), "case_id": cid,
                     **measure(run, m), "run_path": str(p)})
    if not recs:
        print("no measurable runs. What each --out holds:")
        for out in args.out:
            base = Path(out)
            if not base.is_dir():
                print(f"  {out}: folder does not exist (relative to {Path.cwd()})")
                continue
            n_match = len(discover([out], set(args.rows) if args.rows else None))
            n_any = sum(1 for _ in base.rglob("*.json"))
            subs = sorted(p.name for p in base.iterdir())[:12]
            print(f"  {out}: {n_match} JSONs at <row>/<sub>/<case>.json, {n_any} JSONs anywhere; "
                  f"top level: {', '.join(subs) or '(empty)'}")
        for k, v in skipped.items():
            print(f"  skipped {v}: {k}")
        sys.exit(1)

    groups = defaultdict(list)
    for r in recs:
        groups[(r["out"], r["row"], r["system"])].append(r)
        groups[(r["out"], r["row"], "ALL")].append(r)
        groups[("*", "*", r["system"])].append(r)
        groups[("*", "*", "ALL")].append(r)

    def over(rs, key, ctx):
        return sum(r[key] + NUM_PREDICT > ctx for r in rs)

    tag = "judge pre-check" if CPT == CHARS_PER_TOKEN else f"REAL-TOKEN view (--cpt {CPT}; judge uses {CHARS_PER_TOKEN})"
    print(f"{tag}: est = chars / {CPT}; too_long if est + {NUM_PREDICT} > num_ctx")
    print("counts below are too_long cases at each num_ctx, as lo/hi (claim-length bounds)\n")
    hdr = f"{'out':<20} {'row':<16} {'sys':<4} {'n':>3} {'max_lo':>7} {'max_hi':>7} {'p90_hi':>7} {'p50_hi':>7}"
    hdr += "".join(f" {('>' + str(c // 1024) + 'k'):>8}" for c in ctxs)
    print(hdr)
    summary = []
    for key in sorted(groups, key=lambda k: (k[0] == "*", k[0], k[1], k[2] == "ALL", k[2])):
        rs = groups[key]
        his = [r["est_hi"] for r in rs]
        line = {"out": key[0], "row": key[1], "system": key[2], "n": len(rs),
                "max_lo": max(r["est_lo"] for r in rs), "max_hi": max(his),
                "p90_hi": pct(his, 90), "p50_hi": pct(his, 50)}
        for c in ctxs:
            line[f"too_long_lo_{c}"] = over(rs, "est_lo", c)
            line[f"too_long_hi_{c}"] = over(rs, "est_hi", c)
        summary.append(line)
        cells = "".join(f" {str(line[f'too_long_lo_{c}']) + '/' + str(line[f'too_long_hi_{c}']):>8}"
                        for c in ctxs)
        print(f"{key[0]:<20} {key[1]:<16} {key[2]:<4} {len(rs):>3} {line['max_lo']:>7} "
              f"{line['max_hi']:>7} {line['p90_hi']:>7} {line['p50_hi']:>7}{cells}")
        if key[2] == "ALL" and key[0] != "*":
            print()

    need = max(r["est_hi"] for r in recs) + NUM_PREDICT
    print(f"\nlargest est_hi + NUM_PREDICT = {need} tokens (judge estimate, not Ollama's count)")
    worst = sorted(recs, key=lambda r: -r["est_hi"])[: args.top]
    print(f"top {len(worst)} by est_hi:")
    for r in worst:
        print(f"  {r['est_hi']:>6} (lo {r['est_lo']:>6})  n_ev={r['n_evidence']:>3}  "
              f"{r['out']}/{r['row']}/{r['case_id']}")
    for k, v in skipped.items():
        print(f"skipped {v}: {k}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        sp = path.with_name(path.stem + "_summary.csv")
        with open(sp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"\nper-run CSV -> {path}\nsummary CSV -> {sp}")


# ---------------------------------------------------------------- dev calibration


def calibrate(args, manifest):
    d = Path(args.calibrate)
    cases_csv, claims_csv, cache = d / "grounding_cases.csv", d / "grounding_claims.csv", d / "judge_cache"
    for f in (cases_csv, cache):
        if not f.exists():
            sys.exit(f"missing {f}")
    with open(cases_csv, newline="") as f:
        cases = list(csv.DictReader(f))
    claims_by_case = defaultdict(list)
    if claims_csv.exists():
        with open(claims_csv, newline="") as f:
            for c in csv.DictReader(f):
                claims_by_case[(c["row"], c["case_id"])].append(c)

    runs, bound_bad, n_bound = {}, 0, 0
    print(f"1) bounds check vs judge's prompt_tokens_B ({cases_csv})")
    for c in cases:
        p = Path(c["run_path"])
        if not p.exists():
            p = repo_path(c["run_path"])
        try:
            run = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"  cannot read {c['run_path']} (run this from the folder you ran the judge in)")
            continue
        m = manifest[c["case_id"]]
        runs[(c["row"], c["case_id"])] = (run, m)
        rec = c.get("prompt_tokens_B", "").lstrip("~")
        if not rec:
            continue
        n_bound += 1
        mm = measure(run, m)
        ok = mm["est_lo"] - 1 <= int(rec) <= mm["est_hi"] + 1  # +-1 for int() rounding
        bound_bad += not ok
        if not ok or c["status"] == "too_long":
            print(f"  {'OUT OF BOUNDS' if not ok else 'too_long'}: {c['row']}/{c['case_id']} "
                  f"judge={rec} lo={mm['est_lo']} hi={mm['est_hi']} status={c['status']}")
    print(f"  {n_bound} cases with prompt_tokens_B, {bound_bad} out of bounds\n")

    print(f"2) real Ollama token counts from {cache}/ (model={args.model}, num_ctx={args.num_ctx})")
    ratios, first_ratio, found, missing = [], [], 0, 0
    for (row, cid), cl in claims_by_case.items():
        if (row, cid) not in runs:
            continue
        run, m = runs[(row, cid)]
        prefix = prefix_for(run, m)
        per_case = []
        for c in sorted(cl, key=lambda x: int(x["claim_id"])):
            body = {
                "model": args.model,
                "messages": [{"role": "system", "content": LABEL_SYSTEM},
                             {"role": "user", "content": prefix + c["claim"]}],
                "stream": False,
                "format": LABEL_SCHEMA,
                "options": {"temperature": 0, "seed": 0, "num_ctx": args.num_ctx, "num_predict": NUM_PREDICT},
            }
            key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            f = cache / f"{key}.json"
            if not f.exists():
                missing += 1
                continue
            n_tok = json.loads(f.read_text()).get("prompt_eval_count") or 0
            if not n_tok:
                continue
            found += 1
            chars = len(LABEL_SYSTEM) + len(prefix) + len(c["claim"])
            per_case.append((int(c["claim_id"]), chars, n_tok))
        if per_case:
            # Ollama may report only the tokens it did not reuse from the prompt cache, so later
            # claims of a case can show small counts. Use the largest count per case.
            _, chars, n_tok = max(per_case, key=lambda t: t[2])
            ratios.append(chars / n_tok)
            small = sum(t[2] < 0.5 * n_tok for t in per_case)
            first_ratio.append(small)
    print(f"  cache hits {found}, misses {missing} (misses: claim text/model/num_ctx differ from that run)")
    if ratios:
        print(f"  real chars/token over {len(ratios)} cases: min {min(ratios):.2f} "
              f"median {statistics.median(ratios):.2f} max {max(ratios):.2f}  (judge assumes {CHARS_PER_TOKEN})")
        print(f"  cases where some claim's prompt_eval_count < half the case max: "
              f"{sum(x > 0 for x in first_ratio)}/{len(first_ratio)}"
              f"  (>0 means Ollama reports only non-reused tokens; the judge's truncated check "
              f"then sees small counts on claims 2+)")
        print(f"  lowest ratio {min(ratios):.2f}: true tokens = est x {CHARS_PER_TOKEN / min(ratios):.2f} "
              f"in the worst dev case")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", nargs="+", help="run output dir(s), same as the judge's --out")
    ap.add_argument("--rows", nargs="*", help="only these rows (first dir under --out)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="repo-relative or absolute")
    ap.add_argument("--ctx", nargs="+", type=int, default=DEFAULT_CTX, help="num_ctx values to test")
    ap.add_argument("--top", type=int, default=10, help="list the N largest runs")
    ap.add_argument("--cpt", type=float, default=CHARS_PER_TOKEN,
                    help="chars/token for the size report (default: judge's value; use the --calibrate min for real tokens)")
    ap.add_argument("--csv", help="optional per-run CSV path (+ <stem>_summary.csv)")
    ap.add_argument("--calibrate", help="dev results dir with grounding_cases.csv + judge_cache/")
    ap.add_argument("--model", default="gemma3-12b-16k", help="--calibrate: model the judge ran with")
    ap.add_argument("--num-ctx", type=int, default=16384, help="--calibrate: num_ctx the judge ran with")
    args = ap.parse_args()
    if not args.out and not args.calibrate:
        ap.error("give --out and/or --calibrate")
    manifest = load_manifest(repo_path(args.manifest))
    if args.calibrate and args.cpt != CHARS_PER_TOKEN:
        ap.error("--cpt is for the size report; run --calibrate without it")
    if args.calibrate:
        calibrate(args, manifest)
        if args.out:
            print("\n" + "-" * 80 + "\n")
    if args.out:
        global CPT
        CPT = args.cpt
        size_report(args, manifest)


if __name__ == "__main__":
    main()
